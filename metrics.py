"""
All functions accept 1-D or 2-D label arrays; 2-D arrays are flattened, so
a forecast's 121 target hours all count equally.
"""
from collections import Counter

import numpy as np

from dbscan import UNKNOWN


def _flat(a):
    return np.asarray(a, dtype=object).ravel()


def accuracy(y_true, y_pred):
    """Share of labels predicted exactly. UNKNOWN never matches, so it counts as wrong."""
    t, p = _flat(y_true), _flat(y_pred)
    return float(np.mean(t == p)) if len(t) else float("nan")


def per_class(y_true, y_pred):
    """Precision, recall, F1 and support for every class that appears in y_true or y_pred.

    UNKNOWN is not a class: it is only ever a wrong prediction, which lowers recall.
    """
    t, p = _flat(y_true), _flat(y_pred)
    classes = sorted((set(t) | set(p)) - {UNKNOWN})
    true_count, pred_count = Counter(t), Counter(p)
    hit = Counter(t[t == p])
    out = {}
    for c in classes:
        tp = hit[c]
        prec = tp / pred_count[c] if pred_count[c] else 0.0
        rec = tp / true_count[c] if true_count[c] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "support": true_count[c]}
    return out


def macro_f1(y_true, y_pred):
    """Mean F1 over classes present in y_true, so rare labels weigh as much as common ones."""
    scores = [v["f1"] for v in per_class(y_true, y_pred).values() if v["support"] > 0]
    return float(np.mean(scores)) if scores else float("nan")


def confusion_matrix(y_true, y_pred):
    """Rows = true class, columns = predicted class, with a final UNKNOWN column."""
    t, p = _flat(y_true), _flat(y_pred)
    classes = sorted((set(t) | set(p)) - {UNKNOWN}) + [UNKNOWN]
    index = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for a, b in zip(t, p):
        cm[index[a], index[b]] += 1
    return classes, cm


def skill(score, reference):
    """Improvement over a reference, scaled so 1 = perfect and 0 = no better than reference."""
    if reference >= 1.0:
        return 0.0
    return (score - reference) / (1.0 - reference)


def strip_daynight(labels):
    """'Cloud1D' -> 'Cloud1'. Isolates sky type and intensity from the day/night letter."""
    out = np.array(labels, dtype=object)
    flat = out.ravel()
    for i, lab in enumerate(flat):
        if lab != UNKNOWN and lab and lab[-1] in "DN":
            flat[i] = lab[:-1]
    return out


def per_column(Y_true, Y_pred):
    """Accuracy and macro-F1 for each target column (each forecast lead hour)."""
    Y_true, Y_pred = np.asarray(Y_true, dtype=object), np.asarray(Y_pred, dtype=object)
    acc = np.array([accuracy(Y_true[:, j], Y_pred[:, j]) for j in range(Y_true.shape[1])])
    f1 = np.array([macro_f1(Y_true[:, j], Y_pred[:, j]) for j in range(Y_true.shape[1])])
    return acc, f1


def summary(Y_true, Y_pred):
    """All headline numbers for one method, as a JSON-friendly dict."""
    unknown_rate = float(np.mean(_flat(Y_pred) == UNKNOWN))
    return {
        "accuracy": accuracy(Y_true, Y_pred),
        "macro_f1": macro_f1(Y_true, Y_pred),
        "sky_type_accuracy": accuracy(strip_daynight(Y_true), strip_daynight(Y_pred)),
        "sky_type_macro_f1": macro_f1(strip_daynight(Y_true), strip_daynight(Y_pred)),
        "unknown_rate": unknown_rate,
        "per_class": per_class(Y_true, Y_pred),
    }


def group_scores(y_true, y_pred, groups):
    """Accuracy and macro-F1 separately for each group value (e.g. day of the week)."""
    t, p, g = _flat(y_true), _flat(y_pred), np.asarray(groups).ravel()
    out = {}
    for value in sorted(set(g.tolist())):
        m = g == value
        out[value] = {"n": int(m.sum()), "accuracy": accuracy(t[m], p[m]),
                      "macro_f1": macro_f1(t[m], p[m])}
    return out
