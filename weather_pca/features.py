import re
from typing import Sequence
import numpy as np

from weather_pca import config

RANGE_RE = re.compile(r"^(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)$")
SCALAR_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
LABEL_RE = re.compile(r"(Clear|Cloud|Rain)\d*[DN]")

SWAP_COUNTER = 0


def parse_value(cell: str) -> tuple[float, float]:
    """Parse a scalar float or 'min-max' range string into a (min, max) float pair."""
    global SWAP_COUNTER
    if SCALAR_RE.fullmatch(cell):
        v = float(cell)
        return (v, v)
    m = RANGE_RE.fullmatch(cell)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if a > b:
            a, b = b, a
            SWAP_COUNTER += 1
        return (a, b)
    raise ValueError(f"Invalid numeric or range value: {cell!r}")


def _parse_scalar(cell: str) -> float:
    if not SCALAR_RE.fullmatch(cell):
        raise ValueError(f"Invalid scalar value: {cell!r}")
    return float(cell)


def build_row(row: Sequence[str], include_rainfall: bool) -> list[float]:
    """Extract numeric weather features from a stripped 9-column CSV row."""
    pressure = _parse_scalar(row[7])
    rainfall = _parse_scalar(row[1]) if include_rainfall else None
    t_min, t_max = parse_value(row[2])
    hi_min, hi_max = parse_value(row[3])
    rh_min, rh_max = parse_value(row[4])
    ws_min, ws_max = parse_value(row[5])

    features = [
        pressure,
        t_min,
        t_max,
        hi_min,
        hi_max,
        rh_min,
        rh_max,
        ws_min,
        ws_max,
    ]
    if rainfall is not None:
        features.append(rainfall)
    return features


def map_label(cell: str) -> str | None:
    """Map raw Weather code (e.g. 'Cloud2D') to base class ('Clear', 'Cloud', 'Rain')."""
    m = LABEL_RE.fullmatch(cell)
    return m.group(1) if m else None


def fit_standardizer(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute column-wise mean and population std on training data; zero std becomes 1.0."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0).copy()
    std[std == 0] = 1.0
    return mean, std


def apply_standardizer(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Standardize features using pre-fitted mean and std."""
    return (X - mean) / std


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Prepend a constant bias column of 1.0 at column index 0."""
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    return np.hstack([ones, X])


def filter_out_of_bounds(
    X: np.ndarray,
    y: np.ndarray,
    include_rainfall: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop rows containing non-finite or physically out-of-bounds feature values."""
    cols = list(config.BASE_FEATURES)
    if include_rainfall:
        cols.append(config.RAINFALL_FEATURE)
    lows = np.array([config.FEATURE_BOUNDS[c][0] for c in cols], dtype=np.float64)
    highs = np.array([config.FEATURE_BOUNDS[c][1] for c in cols], dtype=np.float64)

    valid_mask = np.all(np.isfinite(X) & (X >= lows) & (X <= highs), axis=1)
    n_dropped = int(X.shape[0] - np.sum(valid_mask))
    if not np.any(valid_mask):
        raise ValueError("All rows were filtered out as out of bounds")
    return X[valid_mask], y[valid_mask], n_dropped
