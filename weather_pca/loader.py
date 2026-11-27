import collections
import fnmatch
import logging
from pathlib import Path
import warnings

import numpy as np

from weather_cache import MAX, MIN, WeatherCache, valid_mask
from weather_pca import config, features

logger = logging.getLogger(__name__)


def _load_split(
    wc: WeatherCache,
    split_name: str,
    runs: list[dict],
    include_rainfall: bool,
    limit_files: int | None,
    counters: collections.Counter,
    src_indices: dict[str, int],
    wx_lut: np.ndarray,
    wx_valid_lut: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Stream runs from WeatherCache, validate values, and return (X_with_bias, y, dates)."""
    max_rows = sum(
        (
            min(int(r["n_locations"]), limit_files)
            if limit_files is not None
            else int(r["n_locations"])
        )
        * int(r["max_steps"])
        for r in runs
    )
    if max_rows <= 0:
        raise ValueError(f"No forecast steps found in {split_name} split")

    n_features = len(config.BASE_FEATURES) + (1 if include_rainfall else 0)
    X_buf = np.empty((max_rows, 1 + n_features), dtype=np.float64)
    y_buf = np.empty(max_rows, dtype="<U5")

    idx_pressure = src_indices["pressure"]
    idx_temp = src_indices["temp"]
    idx_hi = src_indices["heat_index"]
    idx_rh = src_indices["relhum"]
    idx_ws = src_indices["wind_speed"]
    idx_rain = src_indices.get("rainfall")

    active_src_indices = [idx_pressure, idx_temp, idx_hi, idx_rh, idx_ws]
    if idx_rain is not None:
        active_src_indices.append(idx_rain)
    range_src_indices = (idx_temp, idx_hi, idx_rh, idx_ws)

    lows, highs = features.feature_bounds_arrays(include_rainfall=include_rainfall)
    vocab_len = len(wx_valid_lut)
    total_runs = len(runs)
    write_pos = 0
    dates_seen: list[str] = []
    dates_set: set[str] = set()

    for run_idx, run in enumerate(runs, start=1):
        arrays = wc.load_run(run)
        locs = arrays["location"]
        loc_indices = [
            i
            for i, loc in enumerate(locs)
            if str(loc).startswith(config.DATA_FILE_PREFIX)
        ]
        if limit_files is not None:
            loc_indices = loc_indices[:limit_files]

        if loc_indices:
            loc_idx_arr = np.asarray(loc_indices, dtype=np.intp)
            vm = valid_mask(arrays)[loc_idx_arr]
            vals = arrays["values"][loc_idx_arr][vm]
            wx = arrays["weather"][loc_idx_arr][vm]

            if vals.shape[0] > 0:
                wx_ok = (wx >= 0) & (wx < vocab_len)
                if np.any(wx_ok):
                    wx_ok[wx_ok] = wx_valid_lut[wx[wx_ok]]
                counters["row_unknown_label"] += int(wx.shape[0] - np.sum(wx_ok))

                if np.any(wx_ok):
                    vals = vals[wx_ok]
                    wx = wx[wx_ok]

                    finite_ok = np.all(
                        np.isfinite(vals[:, active_src_indices, :]), axis=(1, 2)
                    )
                    counters["row_bad_value"] += int(
                        vals.shape[0] - np.sum(finite_ok)
                    )

                    if np.any(finite_ok):
                        vals = vals[finite_ok]
                        wx = wx[finite_ok]

                        n_rows_run = vals.shape[0]
                        feat_mat = np.empty((n_rows_run, n_features), dtype=np.float64)

                        p_lo = vals[:, idx_pressure, MIN].astype(np.float64)
                        p_hi = vals[:, idx_pressure, MAX].astype(np.float64)
                        feat_mat[:, 0] = 0.5 * (p_lo + p_hi)

                        col = 1
                        for src_idx in range_src_indices:
                            lo = vals[:, src_idx, MIN].astype(np.float64)
                            hi = vals[:, src_idx, MAX].astype(np.float64)
                            swapped = lo > hi
                            if np.any(swapped):
                                counters["range_swaps"] += int(np.sum(swapped))
                                lo_fixed = np.where(swapped, hi, lo)
                                hi_fixed = np.where(swapped, lo, hi)
                                lo, hi = lo_fixed, hi_fixed
                            feat_mat[:, col] = lo
                            feat_mat[:, col + 1] = hi
                            col += 2

                        if idx_rain is not None:
                            r_lo = vals[:, idx_rain, MIN].astype(np.float64)
                            r_hi = vals[:, idx_rain, MAX].astype(np.float64)
                            feat_mat[:, col] = 0.5 * (r_lo + r_hi)

                        bounds_ok = np.all(
                            np.isfinite(feat_mat)
                            & (feat_mat >= lows)
                            & (feat_mat <= highs),
                            axis=1,
                        )
                        counters["row_out_of_bounds"] += int(
                            n_rows_run - np.sum(bounds_ok)
                        )
                        k = int(np.sum(bounds_ok))
                        if k > 0:
                            X_buf[write_pos : write_pos + k, 0] = 1.0
                            X_buf[write_pos : write_pos + k, 1:] = feat_mat[bounds_ok]
                            y_buf[write_pos : write_pos + k] = wx_lut[wx[bounds_ok]]
                            write_pos += k
                            month = str(run["month"])
                            if month not in dates_set:
                                dates_set.add(month)
                                dates_seen.append(month)

        if run_idx % 25 == 0 or run_idx == total_runs:
            logger.info(
                "Loaded %s %d/%d runs (%.1f%%) | rows=%d",
                split_name,
                run_idx,
                total_runs,
                100.0 * run_idx / total_runs,
                write_pos,
            )

    if write_pos == 0:
        raise ValueError(f"No valid data rows found in {split_name} split")

    X = X_buf[:write_pos]
    y = y_buf[:write_pos]
    if write_pos < int(0.8 * max_rows):
        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(y)
    return X, y, dates_seen


def load_from_cache(
    cache_dir: Path,
    outlook_glob: str = config.OUTLOOK_DIR_GLOB,
    include_rainfall: bool = False,
    limit_files: int | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    collections.Counter,
    list[str],
    list[str],
    dict,
]:
    """Load and validate train/test arrays directly from a built WeatherCache."""
    if limit_files is not None and limit_files < 1:
        raise ValueError("--limit-files must be >= 1")

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        wc = WeatherCache(cache_dir)

    if wc.partial:
        logger.warning(
            "Cache at %s is a partial build (--limit was used); continuing.",
            wc.root,
        )

    train_runs = sorted(
        (
            r
            for r in wc.runs(split="training")
            if fnmatch.fnmatchcase(r["run"], outlook_glob)
        ),
        key=lambda r: (r["month"], r["run"]),
    )
    test_runs = sorted(
        (
            r
            for r in wc.runs(split="testing")
            if fnmatch.fnmatchcase(r["run"], outlook_glob)
        ),
        key=lambda r: (r["month"], r["run"]),
    )
    if not train_runs:
        raise ValueError(
            f"No training runs matching {outlook_glob!r} found in cache {wc.root}"
        )
    if not test_runs:
        raise ValueError(
            f"No testing runs matching {outlook_glob!r} found in cache {wc.root}"
        )

    src_indices: dict[str, int] = {
        "pressure": wc.feature_index("pressure"),
        "temp": wc.feature_index("temp"),
        "heat_index": wc.feature_index("heat_index"),
        "relhum": wc.feature_index("relhum"),
        "wind_speed": wc.feature_index("wind_speed"),
    }
    if include_rainfall:
        src_indices["rainfall"] = wc.feature_index("rainfall")

    vocab_weather = wc.vocab.get("weather", ())
    wx_lut = np.asarray(
        [c if c in config.CLASSES else "" for c in vocab_weather], dtype="<U5"
    )
    wx_valid_lut = np.asarray(
        [c in config.CLASSES for c in vocab_weather], dtype=bool
    )

    counters: collections.Counter = collections.Counter()
    X_train, y_train, train_dates = _load_split(
        wc=wc,
        split_name="training",
        runs=train_runs,
        include_rainfall=include_rainfall,
        limit_files=limit_files,
        counters=counters,
        src_indices=src_indices,
        wx_lut=wx_lut,
        wx_valid_lut=wx_valid_lut,
    )
    X_test, y_test, test_dates = _load_split(
        wc=wc,
        split_name="testing",
        runs=test_runs,
        include_rainfall=include_rainfall,
        limit_files=limit_files,
        counters=counters,
        src_indices=src_indices,
        wx_lut=wx_lut,
        wx_valid_lut=wx_valid_lut,
    )

    if limit_files is None:
        issue_counts = wc.manifest.get("issue_counts", {})
        counters["file_bad_header"] += int(issue_counts.get("file_unreadable", 0)) + sum(
            int(v) for k, v in issue_counts.items() if k.startswith("missing_column:")
        )
        counters["row_bad_ncols"] += int(issue_counts.get("short_row", 0)) + int(
            issue_counts.get("long_row", 0)
        )
        counters["range_swaps"] += sum(
            int(v) for k, v in issue_counts.items() if k.startswith("range_swapped:")
        )

    cache_info = {
        "dir": str(wc.root),
        "created": wc.manifest.get("created"),
        "schema_version": wc.manifest.get("schema_version"),
        "partial": wc.partial,
        "source": wc.manifest.get("source"),
    }
    return X_train, y_train, X_test, y_test, counters, train_dates, test_dates, cache_info
