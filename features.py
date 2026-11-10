"""Turn parsed weather data into numeric feature vectors and label targets.

"""
import math

import numpy as np

from data_loader import COL

# Shared by both modes: the current hour's conditions.
_CURRENT = ["rain", "temp", "rh", "wind_speed", "wind_sin", "wind_cos", "pressure"]
_HOUR = ["hour_sin", "hour_cos"]

CLASSIFY_FEATURES = _CURRENT + _HOUR
FORECAST_FEATURES = _CURRENT + ["rain_prev_sum", "temp_prev_mean", "rh_prev_mean",
                                "pressure_change"] + _HOUR


def encode_angle(degrees):
    """Circular value -> (sin, cos) so 350 degrees sits next to 10 degrees."""
    r = math.radians(degrees)
    return math.sin(r), math.cos(r)


def _hour_pair(hour):
    return encode_angle(hour * 15.0)            # 24 h -> 360 degrees


def _current(rain, temp, rh, ws, wdir, pressure, hour):
    ws_sin, ws_cos = encode_angle(wdir)
    h_sin, h_cos = _hour_pair(hour)
    return [rain, temp, rh, ws, ws_sin, ws_cos, pressure], [h_sin, h_cos]


class Standardizer:
    """z = (x - mean) / std, with mean and std taken from the data passed to fit only."""

    def fit(self, X):
        self.mean_ = X.mean(0)
        std = X.std(0)
        self.std_ = np.where(std > 0, std, 1.0)    # constant column: leave scale alone
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_


def build_forecast_examples(timeline, lead_min=168, lead_max=288, history=24, stride=1):
    """One example per start hour t on the stitched timeline.

    Features : hour t's conditions, summaries of hours t-history .. t-1, hour of day.
    Targets  : labels at t+lead_min .. t+lead_max (inclusive), one column per hour.
    Any example that needs a missing hour is skipped and counted.
    """
    v, labels, times = timeline.values, timeline.labels, timeline.times
    n = len(times)
    leads = np.arange(lead_min, lead_max + 1)
    starts, X, Y, hours = [], [], [], []
    skipped = 0
    for t in range(history, n - lead_max, stride):
        window = v[t - history:t + 1]
        target = [labels[t + L] for L in leads]
        if np.isnan(window).any() or any(lab is None for lab in target):
            skipped += 1
            continue
        prev = v[t - history:t]
        cur, hour = _current(*v[t], times[t].hour)
        extra = [prev[:, COL["rain"]].sum(),
                 prev[:, COL["temp"]].mean(),
                 prev[:, COL["rh"]].mean(),
                 v[t, COL["pressure"]] - v[t - history, COL["pressure"]]]
        starts.append(t)
        X.append(cur + extra + hour)
        Y.append(target)
        hours.append([times[t + L].hour for L in leads])
    return {
        "start_idx": np.array(starts, dtype=int),
        "X": np.array(X, dtype=float).reshape(-1, len(FORECAST_FEATURES)),
        "Y": np.array(Y, dtype=object).reshape(-1, len(leads)),
        "target_hours": np.array(hours, dtype=int).reshape(-1, len(leads)),
        "leads": leads,
        "skipped": skipped,
    }


def build_classify_examples(rows):
    """One example per file row: that row's conditions -> that row's label."""
    X, Y, hours, times = [], [], [], []
    for r in rows:
        cur, hour = _current(r["rain"], r["temp"], r["rh"], r["wind_speed"],
                             r["wind_dir"], r["pressure"], r["time"].hour)
        if any(math.isnan(x) for x in cur):
            continue
        X.append(cur + hour)
        Y.append([r["label"]])
        hours.append([r["time"].hour])
        times.append(r["time"])
    return {
        "times": times,
        "X": np.array(X, dtype=float).reshape(-1, len(CLASSIFY_FEATURES)),
        "Y": np.array(Y, dtype=object).reshape(-1, 1),
        "target_hours": np.array(hours, dtype=int).reshape(-1, 1),
    }
