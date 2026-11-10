"""DBSCAN written from scratch with NumPy, plus cluster voting for prediction.

Definitions used (Ester et al., 1996):
  * neighbours of p  = all points within Euclidean distance eps of p, p included
  * core point       = a point with at least min_pts neighbours
  * a cluster grows from a core point through the neighbours of core points;
    non-core points reached this way are border points; the rest are noise (-1).
"""
from collections import Counter, deque

import numpy as np

UNKNOWN = "unknown"


def _sq_dists(A, B):
    """Squared Euclidean distances between every row of A and every row of B."""
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)          # tiny negatives from rounding -> 0


def k_distance(X, k, chunk=1024):
    """Distance from each point to its k-th nearest point, counting itself as the 1st.

    With k = min_pts, a point is a core point exactly when k_distance <= eps,
    which is why the sorted curve of these values helps choose eps.
    """
    X = np.asarray(X, dtype=float)
    out = np.empty(len(X))
    for i0 in range(0, len(X), chunk):
        d = _sq_dists(X[i0:i0 + chunk], X)
        out[i0:i0 + chunk] = np.partition(d, k - 1, axis=1)[:, k - 1]
    return np.sqrt(out)


class DBSCAN:
    def __init__(self, eps, min_pts, chunk=1024):
        self.eps = float(eps)
        self.min_pts = int(min_pts)
        self.chunk = chunk           # rows per distance block, only affects memory use

    # ---------- clustering (no labels used) ----------
    def _neighbours(self, X):
        eps2 = self.eps ** 2
        neigh = []
        for i0 in range(0, len(X), self.chunk):
            close = _sq_dists(X[i0:i0 + self.chunk], X) <= eps2
            neigh.extend(np.flatnonzero(row) for row in close)
        return neigh

    def fit(self, X, Y=None):
        """Cluster X. If Y (n_samples x n_targets labels) is given, also store cluster votes."""
        X = np.asarray(X, dtype=float)
        neigh = self._neighbours(X)
        self.is_core_ = np.array([len(nb) >= self.min_pts for nb in neigh], dtype=bool)
        labels = np.full(len(X), -1, dtype=int)
        cluster = 0
        for i in range(len(X)):
            if not self.is_core_[i] or labels[i] != -1:
                continue
            labels[i] = cluster
            queue = deque([i])
            while queue:                      # breadth-first expansion
                p = queue.popleft()
                for q in neigh[p]:
                    if labels[q] == -1:       # unclaimed: join this cluster
                        labels[q] = cluster
                        if self.is_core_[q]:  # only core points spread the cluster
                            queue.append(q)
            cluster += 1
        self.labels_ = labels
        self.n_clusters_ = cluster
        self.core_X_ = X[self.is_core_]
        self.core_cluster_ = labels[self.is_core_]
        if Y is not None:
            self._vote(np.asarray(Y, dtype=object))
        return self

    # ---------- voting (labels used only after clustering) ----------
    def _vote(self, Y):
        if Y.ndim == 1:
            Y = Y[:, None]
        k = Y.shape[1]
        overall = Counter(Y.ravel())
        self.votes_ = np.empty((self.n_clusters_, k), dtype=object)
        self.purity_ = np.zeros(self.n_clusters_)
        self.sizes_ = np.zeros(self.n_clusters_, dtype=int)
        for c in range(self.n_clusters_):
            members = Y[self.labels_ == c]
            self.sizes_[c] = len(members)
            shares = []
            for j in range(k):
                counts = Counter(members[:, j])
                # most votes, then more common overall, then alphabetical
                best = min(counts, key=lambda lab: (-counts[lab], -overall[lab], lab))
                self.votes_[c, j] = best
                shares.append(counts[best] / len(members))
            self.purity_[c] = float(np.mean(shares))

    def predict(self, X):
        """Label rows of X with the votes of the nearest core point's cluster.

        Returns (predictions, cluster_ids). A row whose nearest core point is
        farther than eps gets UNKNOWN for every target and cluster id -1.
        """
        X = np.asarray(X, dtype=float)
        k = self.votes_.shape[1]
        pred = np.full((len(X), k), UNKNOWN, dtype=object)
        assigned = np.full(len(X), -1, dtype=int)
        if len(self.core_X_) == 0:
            return pred, assigned
        eps2 = self.eps ** 2
        for i0 in range(0, len(X), self.chunk):
            d = _sq_dists(X[i0:i0 + self.chunk], self.core_X_)
            nearest = d.argmin(1)
            ok = d[np.arange(len(d)), nearest] <= eps2
            rows = np.arange(i0, i0 + len(d))[ok]
            assigned[rows] = self.core_cluster_[nearest[ok]]
            pred[rows] = self.votes_[assigned[rows]]
        return pred, assigned
