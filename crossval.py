"""Leave-one-day-out evaluation for a single 7-day station file.

Outer loop: hide one day, train on the others (minus a gap of hours next to
the hidden day), predict the hidden day. Every row is predicted exactly once.

Inner loop: eps and min_pts are chosen by repeating leave-one-day-out on the
training days only, so the hidden day never influences the settings.

Scaling: each outer fold standardizes with its own training rows only; the
inner loop reuses that scaling (its rows are a subset of the outer training
rows, so nothing from the hidden day leaks in).
"""
import numpy as np

from baselines import HourClimatology, MajorityBaseline
from dbscan import DBSCAN, k_distance, UNKNOWN
from features import Standardizer
from metrics import macro_f1


def day_numbers(times):
    """Day 1 = the first 24 hours from the first row, day 2 = the next 24, ..."""
    first = min(times)
    return np.array([int((t - first).total_seconds() // 86400) + 1 for t in times])


def day_folds(days, times, gap_hours=3):
    """(day, train_mask, test_mask) per day.

    Training rows closer than gap_hours to the hidden day's first or last row
    are dropped: neighbouring hours look almost the same and would make the
    test too easy.
    """
    hours = np.array([t.timestamp() / 3600.0 for t in times])
    days = np.asarray(days)
    folds = []
    for d in sorted(set(days.tolist())):
        test = days == d
        lo, hi = hours[test].min(), hours[test].max()
        train = (~test) & ((hours < lo - gap_hours) | (hours > hi + gap_hours))
        folds.append((d, train, test))
    return folds


def eps_candidates(X, min_pts, n_eps):
    """eps values spread over the k-distance distribution (5th to 95th percentile)."""
    kd = k_distance(X, min_pts)
    return np.unique(np.quantile(kd, np.linspace(0.05, 0.95, n_eps)))


def tune(X, Y, days, times, min_pts_list, n_eps, gap_hours):
    """Pick (eps, min_pts) by leave-one-day-out macro-F1 on these rows only."""
    inner = day_folds(days, times, gap_hours)
    grid = []
    for mp in min_pts_list:
        if mp > len(X) // 2:                      # too few rows for this min_pts
            continue
        for eps in eps_candidates(X, mp, n_eps):
            pred = np.full(Y.shape, UNKNOWN, dtype=object)
            for _, tr, te in inner:
                if tr.sum() >= mp:
                    model = DBSCAN(eps, mp).fit(X[tr], Y[tr])
                    pred[te], _ = model.predict(X[te])
            grid.append({"min_pts": int(mp), "eps": float(eps),
                         "macro_f1": macro_f1(Y, pred),
                         "unknown_rate": float(np.mean(pred == UNKNOWN))})
    if not grid:
        raise ValueError("no min_pts value fits %d training rows" % len(X))
    # best macro-F1; ties -> fewer unknowns -> smaller eps
    return max(grid, key=lambda g: (g["macro_f1"], -g["unknown_rate"], -g["eps"])), grid


def leave_one_day_out(X, Y, hours, times, min_pts_list=(3, 4, 5, 6, 8), n_eps=12, gap_hours=3):
    """Predict every row with models that never saw its day. Returns predictions and fold details."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=object)
    days = day_numbers(times)
    preds = {name: np.full(Y.shape, UNKNOWN, dtype=object)
             for name in ("DBSCAN", "Climatology", "Majority")}
    folds = []
    for d, tr, te in day_folds(days, times, gap_hours):
        Xs = Standardizer().fit(X[tr]).transform(X)
        tr_times = [times[i] for i in np.flatnonzero(tr)]
        best, _ = tune(Xs[tr], Y[tr], days[tr], tr_times, min_pts_list, n_eps, gap_hours)
        model = DBSCAN(best["eps"], best["min_pts"]).fit(Xs[tr], Y[tr])
        preds["DBSCAN"][te], _ = model.predict(Xs[te])
        preds["Climatology"][te] = HourClimatology().fit(hours[tr], Y[tr]).predict(hours[te])
        preds["Majority"][te] = MajorityBaseline().fit(hours[tr], Y[tr]).predict(hours[te])
        folds.append({"day": int(d), "test_rows": int(te.sum()), "train_rows": int(tr.sum()),
                      "tuned_on_days": sorted(set(days[tr].tolist())),
                      "eps": best["eps"], "min_pts": best["min_pts"],
                      "inner_macro_f1": best["macro_f1"], "n_clusters": model.n_clusters_,
                      "train_noise_rate": float(np.mean(model.labels_ == -1))})
    return {"days": days, "predictions": preds, "folds": folds}
