import numpy as np

from weather_pca import config


def feature_bounds_arrays(
    include_rainfall: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lows, highs) float64 arrays aligned with active feature columns."""
    cols = list(config.BASE_FEATURES)
    if include_rainfall:
        cols.append(config.RAINFALL_FEATURE)
    lows = np.array([config.FEATURE_BOUNDS[c][0] for c in cols], dtype=np.float64)
    highs = np.array([config.FEATURE_BOUNDS[c][1] for c in cols], dtype=np.float64)
    return lows, highs


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


def standardize_inplace(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> None:
    """Standardize feature columns in-place using pre-fitted mean and std."""
    X -= mean
    X /= std


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
    lows, highs = feature_bounds_arrays(include_rainfall=include_rainfall)
    valid_mask = np.all(np.isfinite(X) & (X >= lows) & (X <= highs), axis=1)
    n_dropped = int(X.shape[0] - np.sum(valid_mask))
    if not np.any(valid_mask):
        raise ValueError("All rows were filtered out as out of bounds")
    return X[valid_mask], y[valid_mask], n_dropped
