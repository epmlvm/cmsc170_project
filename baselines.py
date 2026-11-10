"""Simple reference methods, scored with exactly the same code as DBSCAN.

"""
from collections import Counter

import numpy as np

from dbscan import UNKNOWN


def _most_common(counter):
    return min(counter, key=lambda lab: (-counter[lab], lab))   # ties: alphabetical


class MajorityBaseline:
    """Always predict the most common training label."""

    def fit(self, hours, Y):
        self.label_ = _most_common(Counter(np.asarray(Y, dtype=object).ravel()))
        return self

    def predict(self, hours):
        return np.full(np.shape(hours), self.label_, dtype=object)


class HourClimatology:
    """Predict the most common training label for the target's hour of day."""

    def fit(self, hours, Y):
        hours, Y = np.asarray(hours).ravel(), np.asarray(Y, dtype=object).ravel()
        self.default_ = _most_common(Counter(Y))
        self.by_hour_ = {}
        for h in np.unique(hours):
            self.by_hour_[int(h)] = _most_common(Counter(Y[hours == h]))
        return self

    def predict(self, hours):
        hours = np.asarray(hours)
        out = np.empty(hours.shape, dtype=object)
        for idx, h in np.ndenumerate(hours):
            out[idx] = self.by_hour_.get(int(h), self.default_)
        return out


def persistence(labels, start_idx, leads):
    """Label seen at the same hour of day, at or before the start time.

    For target hour t+L, look back whole days until reaching an hour <= t that
    has a label. Uses only information available at the start time.
    """
    out = np.full((len(start_idx), len(leads)), UNKNOWN, dtype=object)
    for i, t in enumerate(start_idx):
        for j, L in enumerate(leads):
            src = t + L - 24 * (-(-L // 24))       # -(-L // 24) = ceil(L / 24)
            while src >= 0 and labels[src] is None:
                src -= 24
            if src >= 0:
                out[i, j] = labels[src]
    return out
