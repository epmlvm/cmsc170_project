"""Run the full DBSCAN weather experiment and save results and figures.

    python run_experiment.py --data dataset/training --mode forecast
    python run_experiment.py --data dataset/training --mode classify
    python run_experiment.py --data dataset/training --mode towns     # test on Los Banos, train on other provinces
    python run_experiment.py --data dataset/training --mode week      # Los Banos only: leave one day out

"""
import argparse
import json
import os
import platform

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # write image files, no window needed
import matplotlib.pyplot as plt

import data_loader as dl
import features as ft
import metrics as mt
import split as sp
import crossval as cv
import report
from baselines import HourClimatology, MajorityBaseline, persistence
from dbscan import DBSCAN, k_distance, UNKNOWN


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", default="dataset/training", help="folder holding the run folders")
    p.add_argument("--station", default=dl.STATION_FILE)
    p.add_argument("--mode", choices=["forecast", "classify", "towns", "week"], default="forecast")
    p.add_argument("--out", default="results")
    p.add_argument("--lead-min", type=int, default=168, help="first lead hour (7 days)")
    p.add_argument("--lead-max", type=int, default=288, help="last lead hour (12 days)")
    p.add_argument("--history", type=int, default=24, help="hours summarised before the start")
    p.add_argument("--stride", type=int, default=1, help="use every n-th start hour")
    p.add_argument("--min-pts", type=int, nargs="+", default=[3, 5, 8, 12, 20])
    p.add_argument("--n-eps", type=int, default=12, help="eps values per min_pts")
    p.add_argument("--split", type=float, nargs=3, default=[0.70, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"), help="share of timeline hours per block")
    p.add_argument("--val-share", type=float, default=0.15,
                   help="towns mode: share of towns (whole provinces) used for validation")
    p.add_argument("--seed", type=int, default=0, help="towns mode: seed for province pick and sampling")
    p.add_argument("--max-train", type=int, default=15000,
                   help="towns mode: cap on training rows (fixed random sample) to keep DBSCAN fast")
    p.add_argument("--gap", type=int, default=3,
                   help="week mode: hours next to the hidden day also left out of training")
    p.add_argument("--max-val", type=int, default=5000, help="towns mode: cap on validation rows")
    args = p.parse_args()
    if abs(sum(args.split) - 1.0) > 1e-9 or min(args.split) <= 0:
        p.error("--split needs three positive shares that add up to 1, e.g. 0.5 0.25 0.25")
    return args


# ---------------------------------------------------------------- data
def forecast_sets(args, timeline, blocks):
    ex = ft.build_forecast_examples(timeline, args.lead_min, args.lead_max,
                                    args.history, args.stride)
    block_of = sp.assign(ex["start_idx"], blocks, before=args.history, after=args.lead_max)
    sets = {}
    for b, name in enumerate(sp.NAMES):
        keep = block_of == b
        sets[name] = {k: ex[k][keep] for k in ("X", "Y", "target_hours", "start_idx")}
    info = {"examples_built": int(len(ex["start_idx"])), "skipped_missing_hours": ex["skipped"],
            "dropped_crossing_boundary": int(np.sum(block_of == -1))}
    return sets, ex["leads"], info


def classify_sets(runs, timeline, blocks):
    t0 = timeline.times[0]
    parts = {name: [] for name in sp.NAMES}
    dropped = 0
    for _, path in runs:
        rows = dl.read_station_file(path)
        first = int((rows[0]["time"] - t0).total_seconds() // 3600)
        last = int((rows[-1]["time"] - t0).total_seconds() // 3600)
        b = sp.assign([first], blocks, before=0, after=last - first)[0]
        if b < 0:
            dropped += 1
        else:
            parts[sp.NAMES[b]].extend(rows)
    sets = {name: ft.build_classify_examples(rows) for name, rows in parts.items()}
    return sets, np.array([0]), {"runs_dropped_crossing_boundary": dropped}


def _sample(ex, cap, seed):
    """Fixed-seed random subset of at most cap rows, kept in original order."""
    n = len(ex["X"])
    if n <= cap:
        return ex
    keep = np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))
    return {"X": ex["X"][keep], "Y": ex["Y"][keep], "target_hours": ex["target_hours"][keep],
            "times": [ex["times"][i] for i in keep]}


def towns_sets(args):
    """Test = the station file (Los Banos); its province is excluded; other provinces train/validate."""
    folders = sorted(os.path.join(args.data, d) for d in os.listdir(args.data)
                     if os.path.isfile(os.path.join(args.data, d, args.station)))
    if not folders:
        raise SystemExit("no run folders with %s under %s" % (args.station, args.data))
    names, skipped = set(), set()
    for folder in folders:
        for f in dl.find_station_files(folder):
            try:
                dl.parse_station_name(f)
                names.add(os.path.basename(f))
            except ValueError:
                skipped.add(os.path.basename(f))
    if skipped:
        print("skipping files without a TYPE_Town_Province name:", ", ".join(sorted(skipped)[:10]))
    roles = sp.town_roles(sorted(names), args.station, args.val_share, args.seed)
    unreadable = []
    rows = {"train": [], "validation": [], "test": []}
    for folder in folders:
        for path in dl.find_station_files(folder):
            role = roles.get(os.path.basename(path), "excluded")
            if role == "excluded":
                continue
            try:
                file_rows = dl.read_station_file(path)
            except (ValueError, IndexError) as err:      # malformed file: report and skip it
                if role == "test":
                    raise
                unreadable.append("%s (%s)" % (os.path.basename(path), err))
                continue
            rows[role].extend(file_rows)
    if unreadable:
        print("skipped %d unreadable files, e.g. %s" % (len(unreadable), unreadable[0]))
    sets = {name: ft.build_classify_examples(r) for name, r in rows.items()}
    full = {name: len(s["X"]) for name, s in sets.items()}
    sets["train"] = _sample(sets["train"], args.max_train, args.seed)
    sets["validation"] = _sample(sets["validation"], args.max_val, args.seed + 1)
    first = min(sets["test"]["times"])
    sets["test"]["day"] = np.array([int((t - first).total_seconds() // 86400) + 1
                                    for t in sets["test"]["times"]])
    count = lambda role: sum(1 for r in roles.values() if r == role)
    info = {"run_folders": len(folders), "unreadable_files": unreadable,
            "towns": {role: count(role) for role in ["train", "validation", "test", "excluded"]},
            "rows_before_sampling": full,
            "excluded_province": dl.parse_station_name(args.station)[1],
            "validation_provinces": sorted({dl.parse_station_name(f)[1]
                                            for f, r in roles.items() if r == "validation"}),
            "roles": roles}
    return sets, np.array([0]), info


# ---------------------------------------------------------------- tuning
def tune(Xtr, Ytr, Xva, Yva, min_pts_list, n_eps):
    """Score every (eps, min_pts) pair on validation; eps values come from k-distance quantiles."""
    quantiles = np.linspace(0.05, 0.95, n_eps)
    kdists, grid = {}, []
    for mp in min_pts_list:
        kd = k_distance(Xtr, mp)
        kdists[mp] = kd
        for q, eps in zip(quantiles, np.quantile(kd, quantiles)):
            m = DBSCAN(eps, mp).fit(Xtr, Ytr)
            pred, _ = m.predict(Xva)
            grid.append({"min_pts": mp, "quantile": float(q), "eps": float(eps),
                         "val_macro_f1": mt.macro_f1(Yva, pred),
                         "val_accuracy": mt.accuracy(Yva, pred),
                         "val_unknown_rate": float(np.mean(pred == UNKNOWN)),
                         "n_clusters": m.n_clusters_,
                         "train_noise_rate": float(np.mean(m.labels_ == -1))})
            print("  min_pts=%2d  eps=%.3f  clusters=%4d  val macro-F1=%.3f  unknown=%.2f"
                  % (mp, eps, m.n_clusters_, grid[-1]["val_macro_f1"], grid[-1]["val_unknown_rate"]))
    # best macro-F1; ties -> fewer unknowns -> smaller eps (deterministic)
    best = max(grid, key=lambda g: (g["val_macro_f1"], -g["val_unknown_rate"], -g["eps"]))
    return best, grid, kdists, quantiles


# ---------------------------------------------------------------- figures
def plot_kdistance(kdists, best, path):
    plt.figure(figsize=(7, 4))
    for mp, kd in kdists.items():
        plt.plot(np.sort(kd), label="min_pts=%d" % mp)
    plt.axhline(best["eps"], color="k", ls="--", lw=1, label="chosen eps")
    plt.xlabel("training points, sorted"); plt.ylabel("k-distance (standardized units)")
    plt.title("k-distance plot (k = min_pts)"); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def plot_grid(grid, min_pts_list, quantiles, path):
    Z = np.array([[g["val_macro_f1"] for g in grid if g["min_pts"] == mp] for mp in min_pts_list])
    plt.figure(figsize=(1 + 0.7 * len(quantiles), 1.5 + 0.5 * len(min_pts_list)))
    plt.imshow(Z, aspect="auto", cmap="viridis")
    plt.colorbar(label="validation macro-F1")
    plt.xticks(range(len(quantiles)), ["%d" % round(q * 100) for q in quantiles])
    plt.yticks(range(len(min_pts_list)), min_pts_list)
    plt.xlabel("eps = this percentile of k-distances"); plt.ylabel("min_pts")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            plt.text(j, i, "%.2f" % Z[i, j], ha="center", va="center", fontsize=7, color="w")
    plt.title("Validation grid"); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_leads(leads, per_lead, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    days = leads / 24.0
    for name, (acc, f1) in per_lead.items():
        axes[0].plot(days, acc, label=name)
        axes[1].plot(days, f1, label=name)
    for ax, title in zip(axes, ["Accuracy", "Macro-F1"]):
        ax.set_xlabel("lead (days)"); ax.set_title(title + " by lead (test)"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_confusion(Y_true, Y_pred, path):
    classes, cm = mt.confusion_matrix(Y_true, Y_pred)
    rows = [i for i, c in enumerate(classes) if c != UNKNOWN]      # true labels are never unknown
    cm = cm[rows]
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    plt.figure(figsize=(1.5 + 0.6 * len(classes), 1.5 + 0.5 * len(rows)))
    plt.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="share of true class")
    plt.xticks(range(len(classes)), classes, rotation=60, ha="right")
    plt.yticks(range(len(rows)), [classes[i] for i in rows])
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title("DBSCAN test confusion (row-normalized)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_clusters(model, path, title=None):
    order = np.argsort(-model.sizes_)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(order)), model.sizes_[order], color="tab:blue")
    ax.set_xlabel("cluster (largest first)"); ax.set_ylabel("members", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(range(len(order)), model.purity_[order], "o", color="tab:red", ms=3)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("purity (majority-label share)", color="tab:red")
    ax.set_title(title or "Clusters after final fit (noise points: %d)" % int(np.sum(model.labels_ == -1)))
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_days(by_day, path):
    names = list(by_day)
    days = sorted(by_day[names[0]])
    width = 0.8 / len(names)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k, name in enumerate(names):
        x = np.arange(len(days)) + k * width
        axes[0].bar(x, [by_day[name][d]["accuracy"] for d in days], width, label=name)
        axes[1].bar(x, [by_day[name][d]["macro_f1"] for d in days], width, label=name)
    n = [by_day[names[0]][d]["n"] for d in days]
    for ax, title in zip(axes, ["Accuracy", "Macro-F1"]):
        ax.set_xticks(np.arange(len(days)) + 0.4 - width / 2)
        ax.set_xticklabels(["day %d\n(n=%d)" % (d, c) for d, c in zip(days, n)])
        ax.set_ylim(0, 1.18); ax.set_title(title + " by day (Los Banos test)")
        ax.legend(loc="upper center", ncol=len(names), fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def run_week(args):
    """Los Banos only: every row is predicted by a model that never saw its day."""
    runs = dl.find_runs(args.data, args.station)
    if not runs:
        raise SystemExit("no run folders with %s under %s" % (args.station, args.data))
    if len(runs) > 1:
        print("week mode uses one run; using the first: %s" % runs[0][0])
    rows = dl.read_station_file(runs[0][1])
    ex = ft.build_classify_examples(rows)
    print("rows: %d   from %s to %s" % (len(ex["X"]), min(ex["times"]), max(ex["times"])))
    print("leave-one-day-out with nested tuning ...")
    out = cv.leave_one_day_out(ex["X"], ex["Y"], ex["target_hours"], ex["times"],
                               args.min_pts, args.n_eps, args.gap)
    Y, preds, days = ex["Y"], out["predictions"], out["days"]

    results = {name: mt.summary(Y, P) for name, P in preds.items()}
    clim_f1 = results["Climatology"]["macro_f1"]
    by_day = {name: mt.group_scores(Y, P, days) for name, P in preds.items()}
    for name in results:
        results[name]["skill_vs_climatology"] = mt.skill(results[name]["macro_f1"], clim_f1)
        results[name]["by_day"] = {str(d): v for d, v in by_day[name].items()}

    # descriptive picture only: one fit on all rows with the most often chosen settings
    choice = max({(f["eps"], f["min_pts"]) for f in out["folds"]},
                 key=lambda c: sum(1 for f in out["folds"] if (f["eps"], f["min_pts"]) == c))
    Xs = ft.Standardizer().fit(ex["X"]).transform(ex["X"])
    model = DBSCAN(choice[0], choice[1]).fit(Xs, Y)

    plot_days(by_day, os.path.join(args.out, "scores_by_day.png"))
    plot_confusion(Y, preds["DBSCAN"], os.path.join(args.out, "confusion_dbscan.png"))
    plot_kdistance({m: k_distance(Xs, m) for m in args.min_pts if m < len(Xs)},
                   {"eps": choice[0]}, os.path.join(args.out, "kdistance.png"))
    if model.n_clusters_:
        plot_clusters(model, os.path.join(args.out, "clusters.png"),
                      "Descriptive fit on all rows, eps=%.2f min_pts=%d (not used for scoring)" % choice)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"results": results, "folds": out["folds"],
                   "rows": [{"time": str(t), "true": y[0], "dbscan": p[0]}
                            for t, y, p in zip(ex["times"], Y, preds["DBSCAN"])]}, f, indent=2)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"args": vars(args), "features": ft.CLASSIFY_FEATURES,
                   "python": platform.python_version(), "numpy": np.__version__,
                   "matplotlib": matplotlib.__version__}, f, indent=2)

    # plain-text outputs, same comma-separated style as the dataset
    stem = os.path.splitext(args.station)[0].replace("MUN_", "").replace("CTY_", "")
    pred_path = os.path.join(args.out, "predictions_%s.txt" % stem)
    report.write_predictions(runs[0][1], pred_path,
                             {name: dict(zip(ex["times"], P[:, 0])) for name, P in preds.items()})

    lines = ["DBSCAN weather prediction - %s, week mode (leave one day out)" % args.station,
             "data: %d rows, %s to %s" % (len(Y), min(ex["times"]), max(ex["times"])),
             "gap: %d hours   min_pts tried: %s   eps values per min_pts: %d"
             % (args.gap, args.min_pts, args.n_eps),
             "", "FOLDS (one per hidden day)"]
    for fd in out["folds"]:
        lines.append("  day %d: test %2d rows, train %2d rows, eps=%.3f min_pts=%d, clusters=%d"
                     % (fd["day"], fd["test_rows"], fd["train_rows"], fd["eps"], fd["min_pts"],
                        fd["n_clusters"]))
    lines += ["", "RESULTS over all %d rows" % len(Y),
              "%-12s %9s %9s %10s %9s %9s" % ("method", "accuracy", "macro-F1", "skill/clim",
                                              "sky acc", "unknown")]
    for name, r in results.items():
        lines.append("%-12s %9.3f %9.3f %10.3f %9.3f %9.3f" % (name, r["accuracy"], r["macro_f1"],
                     r["skill_vs_climatology"], r["sky_type_accuracy"], r["unknown_rate"]))
    lines += ["", "BY DAY (accuracy)", "%-12s" % "method" + "".join(
        "  day %s (n=%d)" % (d, v["n"]) for d, v in by_day["DBSCAN"].items())]
    for name in results:
        lines.append("%-12s" % name + "".join("  %13.2f" % v["accuracy"] for v in by_day[name].values()))
    lines += ["", "DBSCAN PER CLASS",
              "%-10s %9s %9s %9s %8s" % ("label", "precision", "recall", "F1", "support")]
    for lab, v in results["DBSCAN"]["per_class"].items():
        lines.append("%-10s %9.3f %9.3f %9.3f %8d" % (lab, v["precision"], v["recall"], v["f1"],
                                                      v["support"]))
    lines += ["", "row-by-row predictions: %s" % os.path.basename(pred_path)]
    text = "\n".join(lines)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print("saved to", os.path.abspath(args.out))


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.mode == "week":
        run_week(args)
        return

    if args.mode == "towns":
        sets, leads, split_info = towns_sets(args)
        counts = {name: int(len(s["X"])) for name, s in sets.items()}
        split_info["examples_per_block"] = counts
        coverage = "run folders: %d   towns: %s   excluded province: %s" % (
            split_info["run_folders"], split_info["towns"], split_info["excluded_province"])
        print(coverage)
        print("validation provinces:", ", ".join(split_info["validation_provinces"]))
        with open(os.path.join(args.out, "split.json"), "w") as f:
            json.dump(split_info, f, indent=2)
        split_info = {k: v for k, v in split_info.items() if k != "roles"}   # keep the printout short
    else:
        runs = dl.find_runs(args.data, args.station)
        if not runs:
            raise SystemExit("no run folders with %s under %s" % (args.station, args.data))
        timeline = dl.stitch(runs)
        coverage = dl.describe(runs, timeline)
        print(coverage)
        blocks = sp.make_blocks(len(timeline.times), tuple(args.split))
        if args.mode == "forecast":
            sets, leads, split_info = forecast_sets(args, timeline, blocks)
        else:
            sets, leads, split_info = classify_sets(runs, timeline, blocks)
        counts = {name: int(len(s["X"])) for name, s in sets.items()}
        split_info["examples_per_block"] = counts
        sp.save(os.path.join(args.out, "split.json"), blocks, timeline.times, split_info)
    print("examples per block:", counts, split_info)
    if args.mode == "towns" and counts["train"] < 2:
        raise SystemExit("no training towns found. Towns mode trains on other municipalities: copy "
                         "their files (e.g. MUN_Town_Province.txt) into the run folder next to %s. "
                         "Towns in the same province as the test town are left out on purpose."
                         % args.station)
    for name, n in counts.items():
        if n < 2:
            raise SystemExit("not enough %s examples (%d). The data span is too short for "
                             "this mode / horizon; see split.json." % (name, n))

    # standardize with training statistics only
    scaler = ft.Standardizer().fit(sets["train"]["X"])
    Xtr, Xva, Xte = (scaler.transform(sets[n]["X"]) for n in sp.NAMES)
    Ytr, Yva, Yte = (sets[n]["Y"] for n in sp.NAMES)

    print("tuning on validation ...")
    best, grid, kdists, quantiles = tune(Xtr, Ytr, Xva, Yva, args.min_pts, args.n_eps)
    print("best:", best)

    # final model on train + validation, scored once on test
    Xfit = np.vstack([Xtr, Xva]); Yfit = np.vstack([Ytr, Yva])
    Hfit = np.vstack([sets["train"]["target_hours"], sets["validation"]["target_hours"]])
    model = DBSCAN(best["eps"], best["min_pts"]).fit(Xfit, Yfit)
    pred_db, _ = model.predict(Xte)

    preds = {"DBSCAN": pred_db,
             "Climatology": HourClimatology().fit(Hfit, Yfit).predict(sets["test"]["target_hours"])}
    if args.mode == "forecast":
        preds["Persistence"] = persistence(timeline.labels, sets["test"]["start_idx"], leads)
    else:
        preds["Majority"] = MajorityBaseline().fit(Hfit, Yfit).predict(sets["test"]["target_hours"])

    results = {name: mt.summary(Yte, P) for name, P in preds.items()}
    clim_f1 = results["Climatology"]["macro_f1"]
    for name in results:
        results[name]["skill_vs_climatology"] = mt.skill(results[name]["macro_f1"], clim_f1)
    results["DBSCAN"].update({"n_clusters": model.n_clusters_,
                              "train_noise_rate": float(np.mean(model.labels_ == -1)),
                              "mean_cluster_purity": float(np.mean(model.purity_)) if model.n_clusters_ else None})

    by_day = {}
    if args.mode == "towns":
        by_day = {name: mt.group_scores(Yte, P, sets["test"]["day"]) for name, P in preds.items()}
        for name in preds:
            results[name]["by_day"] = {str(d): v for d, v in by_day[name].items()}

    per_lead = {}
    if args.mode == "forecast":
        per_lead = {name: mt.per_column(Yte, P) for name, P in preds.items()}
        for name, (acc, f1) in per_lead.items():
            results[name]["per_lead"] = {"lead_hours": leads.tolist(),
                                         "accuracy": acc.tolist(), "macro_f1": f1.tolist()}

    # figures
    plot_kdistance(kdists, best, os.path.join(args.out, "kdistance.png"))
    plot_grid(grid, args.min_pts, quantiles, os.path.join(args.out, "validation_grid.png"))
    plot_confusion(Yte, pred_db, os.path.join(args.out, "confusion_dbscan.png"))
    if model.n_clusters_:
        plot_clusters(model, os.path.join(args.out, "clusters.png"))
    if by_day:
        plot_days(by_day, os.path.join(args.out, "scores_by_day.png"))
    if per_lead:
        plot_leads(leads, per_lead, os.path.join(args.out, "scores_by_lead.png"))

    config = {"args": vars(args), "best": best,
              "features": ft.FORECAST_FEATURES if args.mode == "forecast" else ft.CLASSIFY_FEATURES,
              "towns": split_info.get("towns"),
              "python": platform.python_version(), "numpy": np.__version__,
              "matplotlib": matplotlib.__version__}
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"coverage": coverage, "results": results, "grid": grid}, f, indent=2)

    print("\nTEST RESULTS (%s mode)" % args.mode)
    print("%-12s %9s %9s %10s %9s %9s" % ("method", "accuracy", "macro-F1", "skill/clim",
                                         "sky acc", "unknown"))
    for name, r in results.items():
        print("%-12s %9.3f %9.3f %10.3f %9.3f %9.3f" % (name, r["accuracy"], r["macro_f1"],
              r["skill_vs_climatology"], r["sky_type_accuracy"], r["unknown_rate"]))
    print("DBSCAN: %d clusters, train noise %.1f%%, eps=%.3f, min_pts=%d"
          % (model.n_clusters_, 100 * results["DBSCAN"]["train_noise_rate"],
             best["eps"], best["min_pts"]))
    if by_day:
        print("\nDBSCAN by day:  " + "  ".join("day %s: %.2f (n=%d)" % (d, v["accuracy"], v["n"])
                                               for d, v in by_day["DBSCAN"].items()))
    print("saved to", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
