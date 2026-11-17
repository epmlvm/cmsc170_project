import collections
import hashlib
import json
from pathlib import Path
import numpy as np

from weather_pca import config


def compute_fingerprint(
    train_files: list[Path],
    test_files: list[Path],
    include_rainfall: bool,
) -> str:
    """Compute SHA-256 fingerprint over file relative paths, sizes, mtimes, and rainfall flag."""
    lines = [
        f"{p.relative_to(config.PROJECT_ROOT)}|{p.stat().st_size}|{p.stat().st_mtime_ns}"
        for p in train_files + test_files
    ]
    payload = "\n".join(lines) + f"|rain={include_rainfall}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_path_for(
    cache_dir: Path,
    train_files: list[Path],
    test_files: list[Path],
    include_rainfall: bool,
) -> Path:
    """Return the .npz cache file path for the given dataset fingerprint."""
    digest = compute_fingerprint(train_files, test_files, include_rainfall)
    return Path(cache_dir) / f"dataset_{digest[:16]}.npz"


def load_cached(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, collections.Counter]:
    """Load raw unstandardized splits and skip counters from .npz cache."""
    with np.load(path, allow_pickle=False) as data:
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        counters = collections.Counter(json.loads(str(data["counters_json"])))
    return X_train, y_train, X_test, y_test, counters


def save_cached(
    path: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    counters: collections.Counter,
) -> None:
    """Save raw unstandardized splits and skip counters to .npz cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        counters_json=np.array(json.dumps(dict(counters))),
    )
