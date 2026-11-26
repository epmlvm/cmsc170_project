"""Build the NumPy weather dataset cache from extensionless ``MUN_*`` CSV files.

Run manually whenever the source dataset or schema changes::

    python build_cache.py
    python build_cache.py --limit 2 --cache scratch/smoke_cache

Future training/evaluation code should only read the cache via ``weather_cache.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from weather_cache import (
    BUILDING_MARKER,
    CACHE_FORMAT,
    COMPASS_16,
    CSV_COLUMN_MAP,
    MANIFEST_NAME,
    MISSING_CODE,
    NUMERIC_FEATURES,
    REPORT_NAME,
    SCHEMA_VERSION,
    SPLITS,
    UNITS_DESC,
    WEATHER_CLASSES,
    normalize_weather_class,
)

_MAX_EXAMPLES = 20
_MISSING_TOKENS = frozenset({"", "-", "nan", "none", "null", "na", "n/a"})

_NUM_PAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_RANGE_RE = re.compile(rf"^\s*({_NUM_PAT})\s*-\s*({_NUM_PAT})\s*$")
_FILENAME_RE = re.compile(r"^MUN_(.+)_([^_]+)$")
_FOLDER_INIT_RE = re.compile(r"_(\d{10})_")

_EPOCH = datetime(1970, 1, 1)
_NAT_INT64 = np.int64(-(1 << 63))

_COMPASS_TO_CODE = {name: idx for idx, name in enumerate(COMPASS_16)}
_WEATHER_TO_CODE = {name: idx for idx, name in enumerate(WEATHER_CLASSES)}
_FEATURE_TO_IDX = {name: idx for idx, name in enumerate(NUMERIC_FEATURES)}


class IssueTracker:
    """Accumulates issue counts and up to ``_MAX_EXAMPLES`` example strings per issue."""

    def __init__(self, max_examples: int = _MAX_EXAMPLES):
        self.max_examples = max_examples
        self.counts: Counter[str] = Counter()
        self.examples: dict[str, list[str]] = {}

    def add(self, key: str, example: str | None = None) -> None:
        self.counts[key] += 1
        if example is not None:
            bucket = self.examples.setdefault(key, [])
            if len(bucket) < self.max_examples:
                bucket.append(example)

    def merge(self, counts: dict[str, int], examples: dict[str, list[str]]) -> None:
        for key, n in counts.items():
            self.counts[key] += n
        for key, exs in examples.items():
            bucket = self.examples.setdefault(key, [])
            for ex in exs:
                if len(bucket) >= self.max_examples:
                    break
                bucket.append(ex)


def parse_numeric_cell(raw: str | int | float | None) -> tuple[float, float, bool, str | None]:
    """Parse a scalar or ``lo-hi`` range cell into ``(lo, hi, is_range, issue)``.

    ``issue`` is one of ``None``, ``"missing"``, ``"bad"``, or ``"range_swapped"``.
    """
    if raw is None:
        return math.nan, math.nan, False, "missing"

    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        val = float(raw)
        if not math.isfinite(val):
            return math.nan, math.nan, False, "missing"
        return val, val, False, None

    text = str(raw).strip()
    if text.lower() in _MISSING_TOKENS:
        return math.nan, math.nan, False, "missing"

    # Try scalar float first so negative scalars like "-5.2" don't trip range logic.
    try:
        val = float(text)
    except ValueError:
        val = None

    if val is not None:
        if not math.isfinite(val):
            return math.nan, math.nan, False, "missing"
        return val, val, False, None

    m = _RANGE_RE.match(text)
    if m:
        try:
            lo = float(m.group(1))
            hi = float(m.group(2))
        except ValueError:
            return math.nan, math.nan, False, "bad"
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return math.nan, math.nan, False, "bad"
        if lo > hi:
            return hi, lo, True, "range_swapped"
        return lo, hi, True, None

    return math.nan, math.nan, False, "bad"


def parse_filename(name: str) -> tuple[str, str, bool]:
    """Extract ``(municipality, province, matched)`` from ``MUN_<Municipality>_<Province>``."""
    m = _FILENAME_RE.match(name)
    if m:
        return m.group(1).replace("_", " "), m.group(2), True
    return name, "", False


def parse_folder_init_time(folder_name: str) -> datetime | None:
    """Extract initialization datetime from a folder name like ``..._2018040102_pst``."""
    m = _FOLDER_INIT_RE.search(folder_name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H")
    except ValueError:
        return None


def parse_timestamp_sec(raw: str, cache: dict[str, int]) -> int:
    """Parse ``YYYY-MM-DD HH:MM:SS`` into naive UTC-epoch seconds (no local-TZ shift)."""
    text = raw.strip()
    cached = cache.get(text)
    if cached is not None:
        return cached
    if not text:
        sec = int(_NAT_INT64)
    else:
        try:
            dt = datetime.fromisoformat(text)
            sec = int((dt - _EPOCH).total_seconds())
        except ValueError:
            sec = int(_NAT_INT64)
    cache[text] = sec
    return sec


def encode_weather(raw: str) -> tuple[int, str | None]:
    """Map raw weather code to ``0=Clear, 1=Cloud, 2=Rain`` or ``-1`` with an issue key."""
    cleaned = (raw or "").strip()
    if not cleaned:
        return MISSING_CODE, "missing:weather"
    cls = normalize_weather_class(cleaned)
    if cls is None:
        return MISSING_CODE, "weather_unknown"
    return _WEATHER_TO_CODE[cls], None


def _parse_noah_info(run_dir: Path, issues: IssueTracker) -> datetime | None:
    noah_path = run_dir / "NoahInfo"
    if not noah_path.is_file():
        return None
    try:
        text = noah_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        issues.add("noahinfo_unreadable", f"{noah_path}: {exc}")
        return None
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    issues.add("noahinfo_bad_timestamp", f"{noah_path}: {text!r}")
    return None


def _discover_runs(dataset_dir: Path) -> list[tuple[str, str, str, Path]]:
    """Return sorted ``(split, month, run_name, run_dir)`` under ``training/`` and ``testing/``."""
    discovered: list[tuple[str, str, str, Path]] = []
    for split in SPLITS:
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            continue
        for month_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for run_dir in sorted(p for p in month_dir.iterdir() if p.is_dir()):
                discovered.append((split, month_dir.name, run_dir.name, run_dir))
    return discovered


def _process_run(
    task: tuple[str, str, str, str],
) -> tuple[dict | None, dict[str, np.ndarray] | None, list[str], dict[str, int], dict[str, list[str]]]:
    """Parse all extensionless location CSVs inside one outlook folder."""
    split, month, run_name, run_dir_str = task
    run_dir = Path(run_dir_str)
    issues = IssueTracker()

    folder_dt = parse_folder_init_time(run_name)
    noah_dt = _parse_noah_info(run_dir, issues)
    if folder_dt is not None and noah_dt is not None and folder_dt != noah_dt:
        issues.add(
            "init_time_mismatch",
            f"{run_dir}: folder={folder_dt.isoformat()} noah={noah_dt.isoformat()}",
        )

    init_dt = folder_dt if folder_dt is not None else noah_dt
    if init_dt is None:
        issues.add("init_time_missing", str(run_dir))
        init_arr = np.array("NaT", dtype="datetime64[s]")
        init_iso: str | None = None
    else:
        init_sec = int((init_dt - _EPOCH).total_seconds())
        init_arr = np.array(init_sec, dtype="datetime64[s]")
        init_iso = init_dt.isoformat()

    csv_files = sorted(
        p for p in run_dir.iterdir()
        if p.is_file()
        and p.name != "NoahInfo"
        and not p.name.endswith(".json")
        and not p.name.startswith(".")
    )

    if not csv_files:
        issues.add("empty_run", str(run_dir))
        return None, None, [], dict(issues.counts), issues.examples

    time_cache: dict[str, int] = {}
    extra_wind_labels: list[str] = []
    extra_wind_map: dict[str, int] = {}

    parsed_locs: list[dict] = []
    expected_fields = ("date",) + NUMERIC_FEATURES + ("wind_dir", "weather")

    for path in csv_files:
        try:
            with open(path, encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    issues.add("file_unreadable", f"{path}: empty file / missing header")
                    continue

                field_col: dict[str, int] = {}
                for idx, col_name in enumerate(header):
                    mapped = CSV_COLUMN_MAP.get(col_name.strip())
                    if mapped is not None and mapped not in field_col:
                        field_col[mapped] = idx

                missing_cols = [fld for fld in expected_fields if fld not in field_col]
                for fld in missing_cols:
                    issues.add(f"missing_column:{fld}", str(path))

                n_hdr = len(header)
                date_idx = field_col.get("date")
                wind_idx = field_col.get("wind_dir")
                weather_idx = field_col.get("weather")
                num_col_indices = [field_col.get(feat) for feat in NUMERIC_FEATURES]

                loc_times: list[int] = []
                loc_vals: list[list[tuple[float, float]]] = []
                loc_ranges: list[list[bool]] = []
                loc_wind: list[int] = []
                loc_weather: list[int] = []

                prev_valid_t: int | None = None
                not_inc_logged = False

                for row_num, raw_row in enumerate(reader, start=2):
                    if not raw_row or (len(raw_row) == 1 and not raw_row[0].strip()):
                        continue

                    if len(raw_row) < n_hdr:
                        issues.add("short_row", f"{path}:{row_num}: {len(raw_row)}<{n_hdr}")
                    elif len(raw_row) > n_hdr:
                        issues.add("long_row", f"{path}:{row_num}: {len(raw_row)}>{n_hdr}")

                    if date_idx is not None and date_idx < len(raw_row):
                        t_sec = parse_timestamp_sec(raw_row[date_idx], time_cache)
                        if t_sec == int(_NAT_INT64):
                            issues.add("bad:date", f"{path}:{row_num}: {raw_row[date_idx]!r}")
                        else:
                            if prev_valid_t is not None and t_sec <= prev_valid_t and not not_inc_logged:
                                issues.add("times_not_increasing", f"{path}:{row_num}: {raw_row[date_idx]!r}")
                                not_inc_logged = True
                            prev_valid_t = t_sec
                    else:
                        t_sec = int(_NAT_INT64)
                        issues.add("missing:date", f"{path}:{row_num}")

                    row_vals: list[tuple[float, float]] = []
                    row_ranges: list[bool] = []
                    for feat_idx, feat_name in enumerate(NUMERIC_FEATURES):
                        c_idx = num_col_indices[feat_idx]
                        raw_cell = raw_row[c_idx] if (c_idx is not None and c_idx < len(raw_row)) else None
                        lo, hi, is_rng, issue = parse_numeric_cell(raw_cell)
                        if issue is not None:
                            issues.add(f"{issue}:{feat_name}", f"{path}:{row_num}: {raw_cell!r}")
                        row_vals.append((lo, hi))
                        row_ranges.append(is_rng)

                    if wind_idx is not None and wind_idx < len(raw_row):
                        w_dir = raw_row[wind_idx].strip().upper()
                    else:
                        w_dir = ""
                    if not w_dir:
                        w_code = MISSING_CODE
                        issues.add("missing:wind_dir", f"{path}:{row_num}")
                    elif w_dir in _COMPASS_TO_CODE:
                        w_code = _COMPASS_TO_CODE[w_dir]
                    else:
                        w_code = extra_wind_map.get(w_dir)
                        if w_code is None:
                            w_code = len(COMPASS_16) + len(extra_wind_labels)
                            extra_wind_map[w_dir] = w_code
                            extra_wind_labels.append(w_dir)
                        issues.add("wind_dir_nonstandard", f"{path}:{row_num}: {w_dir!r}")

                    if weather_idx is not None and weather_idx < len(raw_row):
                        raw_wx = raw_row[weather_idx]
                    else:
                        raw_wx = ""
                    wx_code, wx_issue = encode_weather(raw_wx)
                    if wx_issue is not None:
                        issues.add(wx_issue, f"{path}:{row_num}: {raw_wx!r}")

                    loc_times.append(t_sec)
                    loc_vals.append(row_vals)
                    loc_ranges.append(row_ranges)
                    loc_wind.append(w_code)
                    loc_weather.append(wx_code)

        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            issues.add("file_unreadable", f"{path}: {exc}")
            continue

        muni, prov, name_ok = parse_filename(path.name)
        if not name_ok:
            issues.add("filename_unexpected", str(path))

        parsed_locs.append({
            "location": path.name,
            "municipality": muni,
            "province": prov,
            "times": loc_times,
            "values": loc_vals,
            "is_range": loc_ranges,
            "wind_dir": loc_wind,
            "weather": loc_weather,
        })

    if not parsed_locs:
        issues.add("empty_run", str(run_dir))
        return None, None, extra_wind_labels, dict(issues.counts), issues.examples

    n_locs = len(parsed_locs)
    n_steps_arr = np.array([len(loc["times"]) for loc in parsed_locs], dtype=np.int32)
    max_steps = int(n_steps_arr.max()) if n_locs > 0 else 0

    values = np.full((n_locs, max_steps, len(NUMERIC_FEATURES), 2), np.nan, dtype=np.float32)
    is_range = np.zeros((n_locs, max_steps, len(NUMERIC_FEATURES)), dtype=bool)
    wind_dir = np.full((n_locs, max_steps), MISSING_CODE, dtype=np.int16)
    weather = np.full((n_locs, max_steps), MISSING_CODE, dtype=np.int16)
    times_i64 = np.full((n_locs, max_steps), _NAT_INT64, dtype=np.int64)

    for i, loc in enumerate(parsed_locs):
        k = len(loc["times"])
        if k == 0:
            continue
        values[i, :k] = np.asarray(loc["values"], dtype=np.float32)
        is_range[i, :k] = np.asarray(loc["is_range"], dtype=bool)
        wind_dir[i, :k] = np.asarray(loc["wind_dir"], dtype=np.int16)
        weather[i, :k] = np.asarray(loc["weather"], dtype=np.int16)
        times_i64[i, :k] = np.asarray(loc["times"], dtype=np.int64)

    arrays = {
        "values": values,
        "is_range": is_range,
        "wind_dir": wind_dir,
        "weather": weather,
        "times": times_i64.view("datetime64[s]"),
        "n_steps": n_steps_arr,
        "location": np.asarray([loc["location"] for loc in parsed_locs], dtype=str),
        "municipality": np.asarray([loc["municipality"] for loc in parsed_locs], dtype=str),
        "province": np.asarray([loc["province"] for loc in parsed_locs], dtype=str),
        "init_time": init_arr,
    }

    rel_npz = f"{split}/{month}/{run_name}.npz"
    run_info = {
        "split": split,
        "month": month,
        "run": run_name,
        "path": rel_npz,
        "init_time": init_iso,
        "n_locations": n_locs,
        "max_steps": max_steps,
    }
    return run_info, arrays, extra_wind_labels, dict(issues.counts), issues.examples


def _is_inside_dir(candidate: Path, parent: Path) -> bool:
    c_res = candidate.resolve()
    p_res = parent.resolve()
    return c_res == p_res or p_res in c_res.parents


def _is_owned_cache_dir(path: Path) -> bool:
    if (path / BUILDING_MARKER).is_file():
        return True
    manifest_path = path / MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    try:
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, dict) and data.get("format") == CACHE_FORMAT
    except (OSError, json.JSONDecodeError):
        return False


def _remove_owned_dir(path: Path, label: str) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise RuntimeError(f"{label} path exists and is not a directory: {path}")
    if any(path.iterdir()) and not _is_owned_cache_dir(path):
        raise RuntimeError(
            f"Refusing to remove {path}: directory is non-empty and not owned by weather cache."
        )
    shutil.rmtree(path)


def _remap_wind_codes(
    wind_arr: np.ndarray,
    extra_labels: list[str],
    global_vocab: list[str],
    global_map: dict[str, int],
) -> None:
    """Register any non-standard wind labels globally and remap local codes ``>= 16``."""
    base = len(COMPASS_16)
    mapping: dict[int, int] = {}
    for offset, label in enumerate(extra_labels):
        local_code = base + offset
        g_code = global_map.get(label)
        if g_code is None:
            g_code = len(global_vocab)
            global_vocab.append(label)
            global_map[label] = g_code
        mapping[local_code] = g_code

    if all(k == v for k, v in mapping.items()):
        return

    updated = wind_arr.copy()
    for local_code, g_code in mapping.items():
        updated[wind_arr == local_code] = g_code
    wind_arr[:] = updated


def build_cache(
    dataset_dir: Path,
    cache_dir: Path,
    workers: int,
    limit: int | None,
) -> int:
    dataset_res = dataset_dir.resolve()
    cache_res = cache_dir.resolve()

    if not dataset_res.is_dir():
        print(f"Error: dataset directory does not exist: {dataset_res}", file=sys.stderr)
        return 1

    if _is_inside_dir(cache_res, dataset_res):
        print(
            f"Error: cache directory ({cache_res}) must not be inside dataset ({dataset_res}).",
            file=sys.stderr,
        )
        return 1

    if cache_res.exists():
        if not cache_res.is_dir():
            print(f"Error: cache path exists and is not a directory: {cache_res}", file=sys.stderr)
            return 1
        if any(cache_res.iterdir()) and not _is_owned_cache_dir(cache_res):
            print(
                f"Error: refusing to overwrite non-empty unmanaged directory: {cache_res}",
                file=sys.stderr,
            )
            return 1

    staging_dir = cache_res.parent / f"{cache_res.name}.building"
    backup_dir = cache_res.parent / f"{cache_res.name}.old"

    try:
        _remove_owned_dir(staging_dir, "Staging")
        _remove_owned_dir(backup_dir, "Backup")
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    runs = _discover_runs(dataset_res)
    if limit is not None:
        if limit < 1:
            print("Error: --limit must be >= 1.", file=sys.stderr)
            return 1
        runs = runs[:limit]

    if not runs:
        print(f"Error: no outlook folders found under {dataset_res}.", file=sys.stderr)
        return 1

    staging_dir.mkdir(parents=True, exist_ok=True)
    marker_path = staging_dir / BUILDING_MARKER
    marker_path.write_text(f"pid={os.getpid()}\n", encoding="utf-8")

    tasks = [(s, m, r, str(p)) for s, m, r, p in runs]
    global_issues = IssueTracker()
    wind_vocab = list(COMPASS_16)
    wind_map = dict(_COMPASS_TO_CODE)
    manifest_runs: list[dict] = []
    total_locations = 0
    total_steps = 0
    disk_checked = False
    total_runs = len(tasks)

    try:
        if workers > 1 and total_runs > 1:
            pool_ctx = ProcessPoolExecutor(max_workers=min(workers, total_runs))
        else:
            pool_ctx = None

        if pool_ctx is not None:
            with pool_ctx as pool:
                results_iter = pool.map(_process_run, tasks)
                for idx, result in enumerate(results_iter, start=1):
                    disk_checked, loc_n, step_n = _handle_run_result(
                        idx, total_runs, result, staging_dir,
                        wind_vocab, wind_map, global_issues, manifest_runs, disk_checked,
                    )
                    total_locations += loc_n
                    total_steps += step_n
        else:
            for idx, task in enumerate(tasks, start=1):
                result = _process_run(task)
                disk_checked, loc_n, step_n = _handle_run_result(
                    idx, total_runs, result, staging_dir,
                    wind_vocab, wind_map, global_issues, manifest_runs, disk_checked,
                )
                total_locations += loc_n
                total_steps += step_n

        report = {
            "counts": dict(sorted(global_issues.counts.items())),
            "examples": dict(sorted(global_issues.examples.items())),
        }
        report_path = staging_dir / REPORT_NAME
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        manifest = {
            "format": CACHE_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": str(dataset_res).replace("\\", "/"),
            "source_format": "csv",
            "partial": limit is not None,
            "numeric_features": list(NUMERIC_FEATURES),
            "pair_axis": ["min", "max"],
            "dtype": "float32",
            "units": UNITS_DESC,
            "vocab": {
                "wind_dir": wind_vocab,
                "weather": list(WEATHER_CLASSES),
            },
            "runs": manifest_runs,
            "totals": {
                "runs": len(manifest_runs),
                "locations": total_locations,
                "steps": total_steps,
            },
            "issue_counts": dict(sorted(global_issues.counts.items())),
        }
        manifest_path = staging_dir / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        marker_path.unlink(missing_ok=True)
    except BaseException:
        if staging_dir.exists() and _is_owned_cache_dir(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    # Atomic directory swap; if cache_res is locked on Windows, leave staging_dir intact.
    if cache_res.exists():
        try:
            cache_res.rename(backup_dir)
        except OSError as exc:
            print(
                f"Error: could not replace {cache_res} ({exc}). "
                "Close any programs reading from the cache and swap manually.\n"
                f"Existing cache is untouched; completed build is preserved at: {staging_dir}",
                file=sys.stderr,
            )
            return 1
        try:
            staging_dir.rename(cache_res)
        except OSError as exc:
            try:
                backup_dir.rename(cache_res)
            except OSError:
                pass
            print(
                f"Error: could not move new cache into {cache_res} ({exc}). "
                f"Completed build is preserved at: {staging_dir}",
                file=sys.stderr,
            )
            return 1
        if backup_dir.exists() and _is_owned_cache_dir(backup_dir):
            shutil.rmtree(backup_dir, ignore_errors=True)
    else:
        staging_dir.rename(cache_res)

    print(
        f"Done: {len(manifest_runs)} runs, {total_locations} locations, "
        f"{total_steps} steps -> {cache_res}"
    )
    return 0


def _handle_run_result(
    idx: int,
    total_runs: int,
    result: tuple[dict | None, dict[str, np.ndarray] | None, list[str], dict[str, int], dict[str, list[str]]],
    staging_dir: Path,
    wind_vocab: list[str],
    wind_map: dict[str, int],
    global_issues: IssueTracker,
    manifest_runs: list[dict],
    disk_checked: bool,
) -> tuple[bool, int, int]:
    run_info, arrays, extra_wind, counts, examples = result
    global_issues.merge(counts, examples)

    if run_info is None or arrays is None:
        print(f"[{idx}/{total_runs}] skipped empty run")
        return disk_checked, 0, 0

    if extra_wind:
        _remap_wind_codes(arrays["wind_dir"], extra_wind, wind_vocab, wind_map)

    out_path = staging_dir / run_info["path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)

    if not disk_checked:
        first_size = out_path.stat().st_size
        est_bytes = int(first_size * total_runs * 1.15)
        free_bytes = shutil.disk_usage(staging_dir).free
        if free_bytes < est_bytes:
            raise RuntimeError(
                f"Insufficient disk space in {staging_dir}: need ~{est_bytes / 1e9:.2f} GB, "
                f"have {free_bytes / 1e9:.2f} GB free."
            )
        disk_checked = True

    manifest_runs.append(run_info)
    loc_n = int(run_info["n_locations"])
    step_n = int(arrays["n_steps"].sum())
    print(f"[{idx}/{total_runs}] {run_info['split']}/{run_info['month']}/{run_info['run']} "
          f"({loc_n} locs, {run_info['max_steps']} max steps)")
    return disk_checked, loc_n, step_n


def main() -> int:
    default_workers = min(4, os.cpu_count() or 1)
    parser = argparse.ArgumentParser(
        description="Build the NumPy weather dataset cache from extensionless MUN_* CSVs."
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Source dataset directory")
    parser.add_argument("--cache", type=Path, default=Path("cache"), help="Target cache directory")
    parser.add_argument("--workers", type=int, default=default_workers, help="Worker process count")
    parser.add_argument("--limit", type=int, default=None, help="Build only the first N outlook folders")
    args = parser.parse_args()
    return build_cache(args.dataset, args.cache, max(1, args.workers), args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
