"""Time-ordered train / validation / test split with no shared hours.

"""
import json

import numpy as np

NAMES = ["train", "validation", "test"]


def make_blocks(n_hours, fractions=(0.70, 0.15, 0.15)):
    """Cut hour indices 0..n_hours-1 into consecutive [start, end) blocks."""
    cut1 = int(n_hours * fractions[0])
    cut2 = int(n_hours * (fractions[0] + fractions[1]))
    return [(0, cut1), (cut1, cut2), (cut2, n_hours)]


def assign(idx, blocks, before=0, after=0):
    """Block number for each example, or -1 if the hours it touches cross a boundary.

    An example at index i touches hours i-before .. i+after (inclusive). Dropping
    examples that cross a boundary is what keeps test hours out of training.
    """
    idx = np.asarray(idx)
    out = np.full(len(idx), -1, dtype=int)
    for b, (lo, hi) in enumerate(blocks):
        inside = (idx - before >= lo) & (idx + after < hi)
        out[inside] = b
    return out


def save(path, blocks, times, extra=None):
    """Write block boundaries as timestamps so other methods can reuse the exact split."""
    data = {name: {"first_hour": str(times[lo]), "last_hour": str(times[hi - 1])}
            for name, (lo, hi) in zip(NAMES, blocks)}
    if extra:
        data.update(extra)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def town_roles(file_names, test_file, val_share=0.15, seed=0):
    """Assign each station file to 'train', 'validation', 'test' or 'excluded'.

    * the test file is the only test town
    * every other town in the test town's province is excluded, so no near
      neighbour with almost the same weather ends up in training
    * whole provinces go to validation (picked with a fixed seed) until they
      hold at least val_share of the remaining towns; the rest is training
    """
    import random
    from data_loader import parse_station_name

    if test_file not in file_names:
        raise ValueError("test file %s not found" % test_file)
    test_prov = parse_station_name(test_file)[1]
    roles, by_prov = {}, {}
    for f in file_names:
        prov = parse_station_name(f)[1]
        if f == test_file:
            roles[f] = "test"
        elif prov == test_prov:
            roles[f] = "excluded"
        else:
            by_prov.setdefault(prov, []).append(f)
    provinces = sorted(by_prov)
    random.Random(seed).shuffle(provinces)
    target = val_share * sum(len(v) for v in by_prov.values())
    n_val = 0
    for prov in provinces:
        role = "validation" if n_val < target else "train"
        for f in by_prov[prov]:
            roles[f] = role
        if role == "validation":
            n_val += len(by_prov[prov])
    return roles
