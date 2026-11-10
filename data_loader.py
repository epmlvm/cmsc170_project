"""Read the Los Banos forecast files and stitch them into one hourly timeline.

Allowed libraries only: csv, os, re, datetime, math (standard library) and NumPy.
"""
import csv
import os
import re
from datetime import datetime, timedelta

import numpy as np

STATION_FILE = "MUN_LosBanos_Laguna.txt"

# Numeric columns kept, in this order. Heat index is dropped on purpose:
# it is computed from temperature and humidity, so keeping it would
# count those two twice in DBSCAN's distance.
COLUMNS = ["rain", "temp", "rh", "wind_speed", "wind_dir", "pressure"]
COL = {name: i for i, name in enumerate(COLUMNS)}

# 16-point compass rose -> degrees clockwise from north.
_POINTS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
WIND_DEGREES = {p: i * 22.5 for i, p in enumerate(_POINTS)}

_RUN_RE = re.compile(r"_(\d{10})_")


def parse_value(text):
    """Number from a cell. A range like '24.9-30.0' becomes its midpoint."""
    text = text.strip()
    dash = text.find("-", 1)          # start at 1 so a leading minus sign is not a range
    if dash > 0:
        low, high = float(text[:dash]), float(text[dash + 1:])
        return (low + high) / 2.0
    return float(text)


def parse_wind(text):
    """Compass direction (with stray spaces) -> degrees; unknown -> NaN."""
    return WIND_DEGREES.get(text.strip().upper(), float("nan"))


def parse_run_time(folder_name):
    """'weather-outlook_d01_2015111208_pst' -> datetime(2015, 11, 12, 8)."""
    match = _RUN_RE.search(folder_name)
    if match is None:
        raise ValueError("no run time in folder name: %r" % folder_name)
    return datetime.strptime(match.group(1), "%Y%m%d%H")


def read_station_file(path):
    """Return one dict per row with parsed values and the gap to the previous row."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)                                   # header
        for cells in reader:
            if not cells or not cells[0].strip():
                continue
            time = datetime.strptime(cells[0].strip(), "%Y-%m-%d %H:%M:%S")
            step = None
            if rows:
                step = int((time - rows[-1]["time"]).total_seconds() // 3600)
            rows.append({
                "time": time,
                "rain": parse_value(cells[1]),
                "temp": parse_value(cells[2]),
                # cells[3] is heat index: skipped
                "rh": parse_value(cells[4]),
                "wind_speed": parse_value(cells[5]),
                "wind_dir": parse_wind(cells[6]),
                "pressure": parse_value(cells[7]),
                "label": cells[8].strip(),
                "step_hours": step,
            })
    return rows


def find_runs(root, station=STATION_FILE):
    """List (run_time, file_path) for every run folder holding the station file, oldest first."""
    runs = []
    for name in os.listdir(root):
        path = os.path.join(root, name, station)
        if os.path.isfile(path):
            runs.append((parse_run_time(name), path))
    runs.sort()
    return runs


class Timeline:
    """Continuous hourly series. values[i] holds COLUMNS at times[i]; NaN / None = missing."""

    def __init__(self, times, values, labels):
        self.times = times
        self.values = values
        self.labels = labels


def stitch(runs):
    """Merge runs into one hourly timeline.

    Runs are processed oldest first and later runs overwrite earlier ones,
    so every hour keeps the value from the most recently issued run that
    covers it (the shortest forecast lead).
    """
    latest = {}
    for _, path in runs:
        for row in read_station_file(path):
            latest[row["time"]] = row
    if not latest:
        raise ValueError("no rows found")
    start, end = min(latest), max(latest)
    n = int((end - start).total_seconds() // 3600) + 1
    times = [start + timedelta(hours=i) for i in range(n)]
    values = np.full((n, len(COLUMNS)), np.nan)
    labels = [None] * n
    for i, t in enumerate(times):
        row = latest.get(t)
        if row is not None:
            values[i] = [row[c] for c in COLUMNS]
            labels[i] = row["label"]
    return Timeline(times, values, labels)


def describe(runs, timeline):
    """Short text report on coverage, printed by run_experiment.py."""
    lines = ["runs found: %d" % len(runs)]
    if runs:
        lines.append("first run: %s   last run: %s" % (runs[0][0], runs[-1][0]))
        gaps = [(b[0] - a[0]) for a, b in zip(runs, runs[1:])]
        odd = [(runs[i][0], g) for i, g in enumerate(gaps) if g != timedelta(days=1)]
        lines.append("runs not 24 h after the previous one: %d" % len(odd))
        for t, g in odd[:10]:
            lines.append("   after %s: gap %s" % (t, g))
    missing = sum(1 for lab in timeline.labels if lab is None)
    lines.append("timeline: %s to %s, %d hours, %d missing"
                 % (timeline.times[0], timeline.times[-1], len(timeline.times), missing))
    return "\n".join(lines)


def parse_station_name(file_name):
    """'MUN_LosBanos_Laguna.txt' -> ('LosBanos', 'Laguna'): town, then province (last part)."""
    stem = os.path.splitext(os.path.basename(file_name))[0]
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError("unexpected station file name: %r" % file_name)
    return "_".join(parts[1:-1]), parts[-1]


def find_station_files(run_folder):
    """All station .txt files in one run folder, sorted by name."""
    return sorted(os.path.join(run_folder, n) for n in os.listdir(run_folder)
                  if n.lower().endswith(".txt"))
