import collections
import csv
import fnmatch
import logging
from pathlib import Path
import numpy as np

from weather_pca import config, features

logger = logging.getLogger(__name__)


def _list_mun_files(directory: Path, limit_files: int | None) -> list[Path]:
    if not directory.is_dir():
        return []
    files = sorted(
        f
        for f in directory.iterdir()
        if f.is_file()
        and f.name.startswith(config.DATA_FILE_PREFIX)
        and not f.name.endswith(config.IGNORED_SUFFIXES)
    )
    if limit_files is not None:
        files = files[:limit_files]
    return files


def _discover_split(
    split_root: Path,
    outlook_glob: str,
    limit_files: int | None,
) -> tuple[list[Path], list[str]]:
    """Collect MUN_* files from <split_root>/<date>/<outlook_glob>/ in deterministic order."""
    split_root = Path(split_root)
    if not split_root.is_dir():
        raise ValueError(f"Split directory does not exist: {split_root}")

    all_files: list[Path] = []
    date_names: list[str] = []
    for date_dir in sorted(split_root.iterdir()):
        if not date_dir.is_dir():
            logger.debug("Ignoring non-directory under %s: %s", split_root, date_dir.name)
            continue

        date_files: list[Path] = []
        for entry in sorted(date_dir.iterdir()):
            if not entry.is_dir() or not fnmatch.fnmatchcase(entry.name, outlook_glob):
                logger.debug("Ignoring non-outlook entry under %s: %s", date_dir, entry.name)
                continue
            date_files.extend(_list_mun_files(entry, limit_files))

        if date_files:
            date_names.append(date_dir.name)
            all_files.extend(date_files)

    if not all_files:
        raise ValueError(
            f"No {config.DATA_FILE_PREFIX}* data files found under {split_root}/<date>/{outlook_glob}"
        )

    return all_files, date_names


def discover_files(
    train_root: Path,
    test_root: Path,
    outlook_glob: str = config.OUTLOOK_DIR_GLOB,
    limit_files: int | None = None,
) -> tuple[list[Path], list[Path], list[str], list[str]]:
    """Discover extensionless MUN_* CSV files under <split_root>/<date>/<outlook_glob>/."""
    train_files, train_dates = _discover_split(train_root, outlook_glob, limit_files)
    test_files, test_dates = _discover_split(test_root, outlook_glob, limit_files)
    return train_files, test_files, train_dates, test_dates


def parse_file(
    path: Path,
    include_rainfall: bool,
    counters: collections.Counter,
) -> tuple[list[list[float]], list[str]]:
    """Parse one MUN_* CSV file, skipping invalid headers or malformed rows."""
    rows: list[list[float]] = []
    labels: list[str] = []

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or tuple(cell.strip() for cell in header) != config.EXPECTED_HEADER:
            counters["file_bad_header"] += 1
            logger.warning("Skipping file with invalid header: %s", path)
            return ([], [])

        for raw_row in reader:
            if len(raw_row) != 9:
                counters["row_bad_ncols"] += 1
                continue
            row = [cell.strip() for cell in raw_row]
            label = features.map_label(row[8])
            if label is None:
                counters["row_unknown_label"] += 1
                continue
            try:
                vec = features.build_row(row, include_rainfall)
            except ValueError:
                counters["row_bad_value"] += 1
                continue
            rows.append(vec)
            labels.append(label)

    return (rows, labels)


def load_split(
    files: list[Path],
    include_rainfall: bool,
    counters: collections.Counter,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate all parsed rows and labels from a list of CSV paths."""
    swaps_before = features.SWAP_COUNTER
    all_rows: list[list[float]] = []
    all_labels: list[str] = []

    total_files = len(files)
    for idx, path in enumerate(files, 1):
        rows, labels = parse_file(path, include_rainfall, counters)
        all_rows.extend(rows)
        all_labels.extend(labels)
        if idx % 1000 == 0 or idx == total_files:
            logger.info(
                "Parsed %d/%d files (%.1f%%) | rows=%d",
                idx,
                total_files,
                100.0 * idx / total_files,
                len(all_rows),
            )

    counters["range_swaps"] += features.SWAP_COUNTER - swaps_before

    if not all_rows:
        raise ValueError("No valid data rows found in split")

    X = np.asarray(all_rows, dtype=np.float64)
    y = np.asarray(all_labels, dtype="<U5")
    return X, y
