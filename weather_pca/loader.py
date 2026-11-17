import collections
import csv
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
        and f.name.startswith("MUN_")
        and not f.name.endswith(".json")
    )
    if limit_files is not None:
        files = files[:limit_files]
    return files


def discover_files(
    train_root: Path,
    run_glob: str,
    test_dir: Path,
    limit_files: int | None = None,
) -> tuple[list[Path], list[Path]]:
    """Discover extensionless MUN_* CSV files for train run directories and test directory."""
    train_root = Path(train_root)
    test_dir = Path(test_dir)

    train_dirs = sorted(p for p in train_root.glob(run_glob) if p.is_dir())
    train_files: list[Path] = []
    for run_dir in train_dirs:
        train_files.extend(_list_mun_files(run_dir, limit_files))

    if not train_files:
        raise ValueError(f"No data files found in {train_root}")

    test_files = _list_mun_files(test_dir, limit_files)
    if not test_files:
        raise ValueError(f"No data files found in {test_dir}")

    return train_files, test_files


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

    for path in files:
        rows, labels = parse_file(path, include_rainfall, counters)
        all_rows.extend(rows)
        all_labels.extend(labels)

    counters["range_swaps"] += features.SWAP_COUNTER - swaps_before

    if not all_rows:
        raise ValueError("No valid data rows found in split")

    X = np.asarray(all_rows, dtype=np.float64)
    y = np.asarray(all_labels, dtype="<U5")
    return X, y
