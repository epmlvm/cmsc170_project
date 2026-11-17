import numpy as np

from weather_pca import config


def binary_metrics(t: np.ndarray, t_hat: np.ndarray) -> dict[str, float | None]:
    """Compute Option A unitless SSE, MSE, MAPE (%), and R^2 on +/-1 outputs."""
    t_arr = np.asarray(t, dtype=np.float64)
    t_hat_arr = np.asarray(t_hat, dtype=np.float64)
    n = len(t_arr)

    diff = t_arr - t_hat_arr
    sse = float(np.sum(diff * diff))
    mse = float(sse / n)
    mape = float((100.0 / n) * np.sum(np.abs(diff) / np.abs(t_arr)))

    tss = float(np.sum((t_arr - np.mean(t_arr)) ** 2))
    r2 = None if tss == 0.0 else float(1.0 - (sse / tss))
    return {"SSE": sse, "MSE": mse, "MAPE": mape, "R2": r2}


def confusion_matrix(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Build (3, 3) confusion matrix with rows=actual, cols=predicted in CLASSES order."""
    n_cls = len(config.CLASSES)
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for i, actual_cls in enumerate(config.CLASSES):
        a_mask = actual == actual_cls
        for j, pred_cls in enumerate(config.CLASSES):
            cm[i, j] = int(np.sum(a_mask & (predicted == pred_cls)))
    return cm


def accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute overall classification accuracy."""
    return float(np.mean(actual == predicted))


def majority_baseline(
    train_labels: np.ndarray, test_labels: np.ndarray
) -> tuple[str, float]:
    """Find most frequent training class (ties -> earlier in CLASSES) and test accuracy."""
    counts = [int(np.sum(train_labels == cls)) for cls in config.CLASSES]
    best_idx = int(np.argmax(counts))
    cls = config.CLASSES[best_idx]
    acc = float(np.mean(test_labels == cls))
    return cls, acc


def evaluate_split_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    models: dict[str, dict | None],
) -> dict[str, dict[str, float | None] | None]:
    """Compute binary_metrics per trained class and unweighted Mean across active classes."""
    results: dict[str, dict[str, float | None] | None] = {}
    active: list[dict[str, float | None]] = []

    for cls in config.CLASSES:
        model = models.get(cls)
        if model is None:
            results[cls] = None
            continue
        t = np.where(labels == cls, 1.0, -1.0)
        t_hat = np.sign(X @ model["weights"])
        m = binary_metrics(t, t_hat)
        results[cls] = m
        active.append(m)

    if active:
        r2_vals = [m["R2"] for m in active if m["R2"] is not None]
        results["Mean"] = {
            "SSE": float(np.mean([m["SSE"] for m in active])),
            "MSE": float(np.mean([m["MSE"] for m in active])),
            "MAPE": float(np.mean([m["MAPE"] for m in active])),
            "R2": float(np.mean(r2_vals)) if r2_vals else None,
        }
    else:
        results["Mean"] = None

    return results
