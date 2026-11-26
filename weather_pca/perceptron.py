import logging
import numpy as np

logger = logging.getLogger(__name__)


def sign(v: float) -> int:
    """Bipolar step activation matching main.py (returns 0 when v == 0)."""
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def count_errors(X: np.ndarray, d: np.ndarray, w: np.ndarray) -> int:
    """Count samples where sign(X @ w) != d (v == 0 gives y == 0, counted as error)."""
    y = np.sign(X @ w)
    return int(np.sum(y != d))


def train_binary(
    X: np.ndarray,
    d: np.ndarray,
    lr: float,
    bias_init: float,
    max_epochs: int,
    rng: np.random.Generator,
) -> dict:
    """Train a binary perceptron with online updates, shuffling, and Pocket weights."""
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=np.float64)
    w[0] = bias_init

    pocket_w = w.copy()
    pocket_err = count_errors(X, d, w)
    converged = False
    stopped_epoch = max_epochs

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(n_samples)
        errors = 0
        for i in order:
            xi = X[i]
            y = sign(float(xi @ w))
            di = int(d[i])
            if y != di:
                errors += 1
                w += lr * (di - y) * xi
        if errors == 0:
            logger.info(
                "  Epoch %d/%d: converged (0 update errors)",
                epoch,
                max_epochs,
            )
            converged = True
            stopped_epoch = epoch
            break
        e = count_errors(X, d, w)
        if e < pocket_err:
            pocket_err = e
            pocket_w = w.copy()
        logger.info(
            "  Epoch %d/%d: update_errors=%d, eval_errors=%d, best_pocket_errors=%d",
            epoch,
            max_epochs,
            errors,
            e,
            pocket_err,
        )

    final_w = w if converged else pocket_w
    return {
        "weights": final_w,
        "converged": converged,
        "stopped_epoch": stopped_epoch,
        "train_errors": count_errors(X, d, final_w),
    }
