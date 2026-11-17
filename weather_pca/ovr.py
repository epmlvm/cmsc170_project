import logging
import numpy as np

from weather_pca import config, perceptron

logger = logging.getLogger(__name__)


def train_ovr(
    X: np.ndarray,
    labels: np.ndarray,
    lr: float,
    bias_init: float,
    max_epochs: int,
    seed: int,
) -> dict[str, dict | None]:
    """Train one binary perceptron per class in CLASSES order using a single RNG."""
    rng = np.random.default_rng(seed)
    models: dict[str, dict | None] = {}

    for cls in config.CLASSES:
        pos_mask = labels == cls
        if not np.any(pos_mask):
            logger.warning("Class %s has no training samples; skipping.", cls)
            models[cls] = None
            continue

        d = np.where(pos_mask, 1, -1)
        result = perceptron.train_binary(X, d, lr, bias_init, max_epochs, rng)
        if not result["converged"]:
            logger.info(
                "%s: Maximum Epoch Reached (%d). Non-linearly separable. Returning pocket weights.",
                cls,
                max_epochs,
            )
        models[cls] = result

    return models


def scores(X: np.ndarray, models: dict[str, dict | None]) -> np.ndarray:
    """Compute raw dot-product decision scores of shape (N, len(CLASSES))."""
    n_samples = X.shape[0]
    n_classes = len(config.CLASSES)
    out = np.empty((n_samples, n_classes), dtype=np.float64)

    for idx, cls in enumerate(config.CLASSES):
        model = models.get(cls)
        if model is None:
            out[:, idx] = -np.inf
        else:
            out[:, idx] = X @ model["weights"]

    return out


def predict(X: np.ndarray, models: dict[str, dict | None]) -> np.ndarray:
    """Predict class labels via argmax over raw OvR scores (deterministic tie-break)."""
    raw_scores = scores(X, models)
    best_idx = np.argmax(raw_scores, axis=1)
    classes_arr = np.asarray(config.CLASSES, dtype="<U5")
    return classes_arr[best_idx]
