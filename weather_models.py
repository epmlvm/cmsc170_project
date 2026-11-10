
import argparse
import csv
import datetime
import glob
import itertools
import math
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


# ===================== Settings =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # folder containing this script
TARGET = "2m Temp(C)"
WEATHER_COL = "Weather"
FORECAST_DAYS = 7
TEST_FRACTION = 0.2      # last 20% of days are the test period
MIN_ROWS = 30            # minimum usable days per municipality
HMM_WINDOW = 3           # observed days used for filtering (enumeration cost 3^3 = 27)
HMM_STATES = 3           # hidden states: the 3 most common Weather labels

COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW",
           "W", "WNW", "NW", "NNW"]
COMPASS_DEG = {c: i * 22.5 for i, c in enumerate(COMPASS)}
EXTRA_NUMERIC = ["Rainfall(mm)", "Relative Humidity(%)", "Wind Speed(m/s)",
                 "Pressure(hPa)", "Heat Index(C)"]
FORECAST_FIELDS = ["Date", TARGET, "Rainfall(mm)", "Relative Humidity(%)", "Wind Speed(m/s)",
                   "Wind Direction", "Pressure(hPa)", "Heat Index(C)", "Weather", "Weather prob"]


# ===================== Loading =====================
def to_float(text):
    """Convert text to float, or return None if it is not a number."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def read_csv_file(path):
    """Read one CSV file and return a list of clean daily records."""
    records = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            row = {k.strip(): (v or "").strip() for k, v in raw.items() if k is not None}

            temp = to_float(row.get(TARGET, ""))
            humid = to_float(row.get("Relative Humidity(%)", ""))
            wind = to_float(row.get("Wind Speed(m/s)", ""))
            pres = to_float(row.get("Pressure(hPa)", ""))
            rain = to_float(row.get("Rainfall(mm)", ""))
            heat = to_float(row.get("Heat Index(C)", ""))
            direction = COMPASS_DEG.get(row.get("Wind Direction", ""))

            # Skip the day if a required value is missing
            if None in (temp, humid, wind, pres, direction):
                continue
            if rain is None:
                rain = 0.0                             # no rain recorded = 0 mm

            records.append({
                "date": row.get("Date", ""),
                "y": temp,
                "w": row.get(WEATHER_COL, ""),         # Weather label, may be empty
                "deg": direction,
                "extra": {"Rainfall(mm)": rain, "Relative Humidity(%)": humid,
                          "Wind Speed(m/s)": wind, "Pressure(hPa)": pres,
                          "Heat Index(C)": heat},
            })
    return records


def find_csv_files(paths):
    """
    Find dataset files. Searches folders recursively for MUN_*.csv, so the
    subfolder layout does not matter. Output files are never picked up
    because they do not start with MUN_.
    """
    files = []
    for p in paths:
        if os.path.isdir(p):
            found = glob.glob(os.path.join(p, "**", "MUN_*.csv"), recursive=True)
            files.extend(sorted(found))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"Warning: not found, skipping: {p}")
    return sorted(set(files))


def load_all(paths):
    """Load the dataset files. Returns {municipality: [records sorted by date]}."""
    files = find_csv_files(paths)
    if not files:
        sys.exit("No MUN_*.csv files found. Put the datasets in the same folder as the script.")

    by_muni = {}
    for path in files:
        # "MUN_Manila.csv" -> "Manila"
        name = os.path.splitext(os.path.basename(path))[0].replace("MUN_", "").replace("_", " ")
        recs = read_csv_file(path)
        by_muni.setdefault(name, []).extend(recs)
        print(f"Loaded {os.path.basename(path)}: {len(recs)} usable rows")

    for name in by_muni:
        by_muni[name].sort(key=lambda r: r["date"])   # ISO dates sort correctly as text
    return by_muni


# ===================== Helpers =====================
def gauss_pdf(x, mu, sd):
    """Gaussian density of temperature x under a state with mean mu and std sd."""
    z = (x - mu) / sd
    return math.exp(-0.5 * z * z) / (sd * math.sqrt(2 * math.pi))


def circ_mean_deg(weights, degs):
    """Weighted circular mean of angles in degrees (handles the 359 to 1 wrap-around)."""
    s = sum(w * math.sin(math.radians(d)) for w, d in zip(weights, degs))
    c = sum(w * math.cos(math.radians(d)) for w, d in zip(weights, degs))
    return math.degrees(math.atan2(s, c)) % 360


def deg_to_compass(deg):
    return COMPASS[int(((deg + 11.25) % 360) // 22.5)]


def next_dates(last_date, H):
    """Future dates after last_date (expects ISO format YYYY-MM-DD)."""
    try:
        d0 = datetime.date.fromisoformat(last_date[:10])
        return [(d0 + datetime.timedelta(days=k)).isoformat() for k in range(1, H + 1)]
    except ValueError:
        return [f"day +{k}" for k in range(1, H + 1)]


# ===================== HMM =====================
def hmm_fit(train_recs):
    """Estimate the HMM from labeled training days. Returns None if labels are unusable."""
    counts = Counter(r["w"] for r in train_recs if r["w"])
    top = [s for s, _ in counts.most_common(HMM_STATES)]
    if len(top) < 2:
        return None
    N = len(top)
    idx = {s: i for i, s in enumerate(top)}
    total = sum(counts[s] for s in top)

    # Emissions: temperature mean and standard deviation per state
    groups = [[] for _ in range(N)]
    for r in train_recs:
        if r["w"] in idx:
            groups[idx[r["w"]]].append(r["y"])
    if any(len(g) < 2 for g in groups):
        return None
    means, stds = [], []
    for g in groups:
        m = sum(g) / len(g)
        var = sum((x - m) ** 2 for x in g) / len(g)
        means.append(m)
        stds.append(max(math.sqrt(var), 0.5))        # floor avoids a zero-width Gaussian

    # Other variables: per-state average, and per-state circular mean of wind direction
    extra, dirs = [], []
    for s in range(N):
        members = [r for r in train_recs if r["w"] == top[s]]
        stats = {}
        for var in EXTRA_NUMERIC:
            vals = [r["extra"][var] for r in members if r["extra"][var] is not None]
            stats[var] = sum(vals) / len(vals) if vals else float("nan")
        extra.append(stats)
        dirs.append(circ_mean_deg([1.0] * len(members), [r["deg"] for r in members]))

    # Transitions: count consecutive labeled days, Laplace smoothing (add 1 to each cell)
    C = [[1.0] * N for _ in range(N)]
    prev = None
    for r in train_recs:
        cur = idx.get(r["w"])
        if cur is not None and prev is not None:
            C[prev][cur] += 1
        prev = cur
    trans = [[c / sum(row) for c in row] for row in C]

    prior = [counts[s] / total for s in top]         # label frequencies
    return {"top": top, "mean": means, "std": stds, "trans": trans,
            "prior": prior, "extra": extra, "dir": dirs}


def hmm_filter(evidence, model):
    """
    Exact inference by enumeration: P(X_W | e_{1:W}).
    Sums the joint P(x_{1:W}, e_{1:W}) over every hidden sequence x_{1:W},
    grouped by the last state, then normalizes.
    """
    mean, std, T, pi = model["mean"], model["std"], model["trans"], model["prior"]
    N, W = len(mean), len(evidence)
    filt = [0.0] * N
    for hidden in itertools.product(range(N), repeat=W):          # all N^W sequences
        p = pi[hidden[0]] * gauss_pdf(evidence[0], mean[hidden[0]], std[hidden[0]])
        for t in range(1, W):
            p *= T[hidden[t - 1]][hidden[t]] * gauss_pdf(evidence[t], mean[hidden[t]], std[hidden[t]])
        filt[hidden[-1]] += p
    total = sum(filt)
    return [f / total for f in filt] if total > 0 else [1.0 / N] * N


def hmm_predict_horizon(evidence, model, H):
    """
    State distributions P(X_{W+k} | e_{1:W}) for k = 1..H.
    Future temperatures are unobserved, so their likelihood terms sum to 1 and
    drop out. Each step propagates the distribution through the transition matrix.
    """
    dist = hmm_filter(evidence, model)
    T = model["trans"]
    N = len(dist)
    out = []
    for _ in range(H):
        dist = [sum(dist[i] * T[i][j] for i in range(N)) for j in range(N)]
        out.append(dist)
    return out


def hmm_summarize(dist, model):
    """Expected value of each weather variable under one day's state distribution."""
    N = len(dist)
    out = {TARGET: sum(dist[s] * model["mean"][s] for s in range(N))}
    for var in EXTRA_NUMERIC:
        out[var] = sum(dist[s] * model["extra"][s][var] for s in range(N))
    out["Wind Direction"] = deg_to_compass(circ_mean_deg(dist, model["dir"]))
    best = max(range(N), key=lambda s: dist[s])
    out["Weather"] = model["top"][best]
    out["Weather prob"] = dist[best]
    return out


def forecast_future(recs, model, H):
    """Forecast the next H days after the last observed day."""
    evidence = [r["y"] for r in recs[-HMM_WINDOW:]]
    dists = hmm_predict_horizon(evidence, model, H)
    rows = []
    for date, dist in zip(next_dates(recs[-1]["date"], H), dists):
        row = hmm_summarize(dist, model)
        row["Date"] = date
        rows.append(row)
    return rows


# ===================== Metrics =====================
def metrics(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    n = len(y)
    sse = float(np.sum((y - yhat) ** 2))                      # SSE = sum (y - yhat)^2
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "R2": 1 - sse / sst,                                  # 1 - SSE / sum (y - ybar)^2
        "MSE": sse / n,                                       # SSE / n
        "MAPE": 100.0 / n * float(np.sum(np.abs(y - yhat) / np.abs(y))),
        "SSE": sse,
    }


# ===================== Walk-forward evaluation =====================
def horizon_eval(recs, split, model, H):
    """
    From every test-period origin, forecast the next H days and score them against
    the observed temperatures. Each origin uses only data before it.
    """
    y = [r["y"] for r in recs]
    clim = sum(y[:split]) / split                             # training mean
    t_true, t_hmm, t_pers, t_clim = [], [], [], []
    day_sq = [0.0] * H
    day_n = [0] * H

    for o in range(split, len(y) - H + 1):
        dists = hmm_predict_horizon(y[o - HMM_WINDOW:o], model, H)
        for h in range(H):
            actual = y[o + h]
            pred = hmm_summarize(dists[h], model)[TARGET]
            t_true.append(actual)
            t_hmm.append(pred)
            t_pers.append(y[o - 1])
            t_clim.append(clim)
            day_sq[h] += (actual - pred) ** 2
            day_n[h] += 1

    results = {
        "HMM (enumeration)": metrics(t_true, t_hmm),
        "Persistence (last temp)": metrics(t_true, t_pers),
        "Climatology (train mean)": metrics(t_true, t_clim),
    }
    per_day = [day_sq[h] / day_n[h] if day_n[h] else float("nan") for h in range(H)]
    return results, per_day, len(t_true)


def run_municipality(recs, H):
    if len(recs) < MIN_ROWS:
        raise ValueError(f"Not enough usable rows ({len(recs)}). Need at least {MIN_ROWS}.")
    split = int(len(recs) * (1 - TEST_FRACTION))
    if len(recs) - split < H + HMM_WINDOW:
        raise ValueError("Test period is too short for the forecast horizon.")

    model = hmm_fit(recs[:split])
    if model is None:
        raise ValueError("Weather labels are missing or too few per state in the training data.")

    future = forecast_future(recs, model, H)
    results, per_day, n_pairs = horizon_eval(recs, split, model, H)
    return model, future, results, per_day, n_pairs, recs[split]["date"], recs[-1]["date"]


# ===================== Output =====================
def print_states(model):
    for s, name in enumerate(model["top"]):
        print(f"  {name:<12s} prior {model['prior'][s]:.2f}   "
              f"temp mean {model['mean'][s]:.1f} C (sd {model['std'][s]:.1f})")


def print_future(future):
    header = (f"{'Date':<11s} {'Temp C':>7s} {'Rain mm':>8s} {'RH %':>6s} {'Wind m/s':>9s} "
              f"{'Dir':>4s} {'Press hPa':>10s} {'HI C':>6s} {'Weather':<10s} {'Prob':>5s}")
    print(header)
    print("-" * len(header))
    for f in future:
        print(f"{f['Date']:<11s} {f[TARGET]:7.1f} {f['Rainfall(mm)']:8.2f} "
              f"{f['Relative Humidity(%)']:6.1f} {f['Wind Speed(m/s)']:9.2f} "
              f"{f['Wind Direction']:>4s} {f['Pressure(hPa)']:10.1f} "
              f"{f['Heat Index(C)']:6.1f} {f['Weather']:<10s} {f['Weather prob']:5.2f}")


def print_eval(results, per_day):
    header = f"{'Model':28s} {'Test R2':>9s} {'Test MSE':>10s} {'Test MAPE(%)':>13s}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:28s} {m['R2']:9.4f} {m['MSE']:10.4f} {m['MAPE']:13.2f}")
    print("\nHMM temperature MSE by forecast day: " +
          ", ".join(f"day {h + 1}: {v:.2f}" for h, v in enumerate(per_day)))


def save_rows(rows, fields, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def plot_forecast(recs, future, muni, out_dir):
    hist = recs[-30:]
    labels = [r["date"] for r in hist] + [f["Date"] for f in future]
    hx = list(range(len(hist)))
    fx = list(range(len(hist), len(hist) + len(future)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hx, [r["y"] for r in hist], marker="o", label="observed")
    ax.plot(fx, [f[TARGET] for f in future], marker="o", linestyle="--", label="HMM forecast")
    ax.set_xticks(hx + fx)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("2m temperature (C)")
    ax.set_title(f"{len(future)}-day temperature forecast: {muni}")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, f"forecast_{muni.replace(' ', '_')}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def choose_municipality(munis):
    print("\nAvailable municipalities:")
    for i, m in enumerate(munis, 1):
        print(f"  {i:3d}. {m}")
    while True:
        choice = input("\nEnter a number or name: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(munis):
            return munis[int(choice) - 1]
        for m in munis:
            if choice.lower() == m.lower():
                return m
        print("Not recognized, try again.")


def main():
    parser = argparse.ArgumentParser(description="7-day weather forecast for one municipality.")
    parser.add_argument("inputs", nargs="*",
                        help="folder or CSV file(s) (optional, default: this script's folder)")
    parser.add_argument("--muni", help="municipality name (skips the menu)")
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "results"), help="output folder")
    args = parser.parse_args()

    paths = args.inputs if args.inputs else [BASE_DIR]

    by_muni = load_all(paths)
    munis = sorted(by_muni)
    print(f"\nFound {len(munis)} municipality(ies).")

    if args.muni:
        matches = [m for m in munis if m.lower() == args.muni.lower()]
        if not matches:
            sys.exit(f"'{args.muni}' not found. Options: {', '.join(munis)}")
        muni = matches[0]
    else:
        muni = choose_municipality(munis)

    recs = by_muni[muni]
    print(f"\nForecasting {FORECAST_DAYS} days for {muni} from {len(recs)} days of data...")
    try:
        model, future, results, per_day, n_pairs, t_start, t_end = run_municipality(recs, FORECAST_DAYS)
    except ValueError as e:
        sys.exit(str(e))

    print("\nHMM states learned from the training period:")
    print_states(model)

    print(f"\n=== {FORECAST_DAYS}-day forecast for {muni} ===")
    print_future(future)

    print(f"\n=== Evaluation on test period {t_start} to {t_end} ({n_pairs} forecasts) ===")
    print_eval(results, per_day)

    os.makedirs(args.out, exist_ok=True)
    tag = muni.replace(" ", "_")
    fc_path = os.path.join(args.out, f"forecast_{tag}.csv")
    ev_path = os.path.join(args.out, f"evaluation_{tag}.csv")
    save_rows(future, FORECAST_FIELDS, fc_path)
    save_rows([{"Model": k, **v} for k, v in results.items()],
              ["Model", "R2", "MSE", "MAPE", "SSE"], ev_path)
    png_path = plot_forecast(recs, future, muni, args.out)
    print(f"\nSaved forecast to {fc_path}")
    print(f"Saved evaluation to {ev_path}")
    print(f"Saved plot to {png_path}")


if __name__ == "__main__":
    main()