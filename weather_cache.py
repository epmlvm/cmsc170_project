"""Read-only access to the NumPy weather cache built by ``build_cache.py``.

This module never builds or modifies the cache. Build it yourself with::

    python build_cache.py

Layout (one uncompressed .npz per outlook folder, holding every location)::

    cache/
      manifest.json                       schema, vocabularies, run index, totals
      build_report.json                   data-quality counts with example files
      <split>/<month>/<outlook_folder>.npz

Arrays in each .npz  (L = locations, T = time steps, padded to the longest location):

    values        float32 (L, T, 6, 2)   numeric features as (min, max); NaN = missing/padding
    is_range      bool    (L, T, 6)      True where the source cell was a "lo-hi" range
    wind_dir      int16   (L, T)         index into vocab["wind_dir"]; -1 = missing/padding
    weather       int16   (L, T)         index into vocab["weather"];  -1 = missing/padding
    times         datetime64[s] (L, T)   forecast valid time (PST); NaT = missing/padding
    n_steps       int32   (L,)           real (unpadded) number of steps per location
    location      str     (L,)           filename, e.g. "MUN_Aborlan_Palawan"
    municipality  str     (L,)
    province      str     (L,)
    init_time     datetime64[s] ()       forecast initialization time (NaT if unknown)

Scalars are stored as min == max. Feature order is ``NUMERIC_FEATURES``.

Example::

    from weather_cache import WeatherCache, MIN, MAX
    cache = WeatherCache()
    for run, arrays in cache.iter_runs(split="training"):
        temp = arrays["values"][:, :, cache.feature_index("temp")]   # (L, T, 2)
        temp_mid = temp.mean(axis=-1)                                 # midpoint
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np

SCHEMA_VERSION = 1
CACHE_FORMAT = "cmsc170-weather-npz"

SPLITS = ("training", "testing")
NUMERIC_FEATURES = ("rainfall", "temp", "heat_index", "relhum", "wind_speed", "pressure")
CATEGORICAL_FEATURES = ("wind_dir", "weather")

CSV_COLUMN_MAP = {
    "Date": "date",
    "Rainfall(mm)": "rainfall",
    "2m Temp(C)": "temp",
    "Heat Index(C)": "heat_index",
    "Relative Humidity(%)": "relhum",
    "Wind Speed(m/s)": "wind_speed",
    "Wind Direction": "wind_dir",
    "Pressure(hPa)": "pressure",
    "Weather": "weather",
}

MIN, MAX = 0, 1
MISSING_CODE = -1

# 16-point compass in clockwise order, so wind_dir code i (< 16) means i * 22.5 degrees.
COMPASS_16 = (
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
)
WEATHER_CLASSES = ("Clear", "Cloud", "Rain")

UNITS_DESC = (
    "date:PST, rainfall:mm, temp:C, heat_index:C, relhum:%, "
    "wind_speed:m/s, wind_dir:None, pressure:hPa, weather:None"
)

MANIFEST_NAME = "manifest.json"
REPORT_NAME = "build_report.json"
BUILDING_MARKER = ".weather_cache_building"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"

_WEATHER_SPLIT_RE = re.compile(r"^([A-Za-z]+?)(\d*)([DN])$")
_WEATHER_CLASS_RE = re.compile(r"^(clear|cloudy?|rainy?)(?:\d+[DN]?|[DN])?$", re.IGNORECASE)
_WEATHER_ALIASES = {
    "clear": "Clear",
    "cloud": "Cloud",
    "cloudy": "Cloud",
    "rain": "Rain",
    "rainy": "Rain",
}


class CacheError(RuntimeError):
    """The cache is missing, incomplete, or was built with a different schema."""


class WeatherCache:
    """Read-only index over a built cache. Loading a run reads one .npz file."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR):
        self.root = Path(cache_dir)
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise CacheError(
                f"No cache found at {self.root} (missing {MANIFEST_NAME}). "
                "Build it with: python build_cache.py"
            )
        try:
            with open(manifest_path, encoding="utf-8") as f:
                self.manifest = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise CacheError(f"Failed to read {manifest_path}: {exc}") from exc

        if self.manifest.get("format") != CACHE_FORMAT:
            raise CacheError(f"{manifest_path} is not a weather cache manifest.")
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise CacheError(
                f"Cache schema v{self.manifest.get('schema_version')} does not match "
                f"reader v{SCHEMA_VERSION}. Rebuild with: python build_cache.py"
            )

        self.numeric_features = tuple(self.manifest["numeric_features"])
        self.vocab = {k: tuple(v) for k, v in self.manifest["vocab"].items()}
        self.partial = bool(self.manifest.get("partial", False))
        if self.partial:
            warnings.warn(
                "Cache was built with --limit (partial build).",
                UserWarning,
                stacklevel=2,
            )

    def runs(self, split: str | None = None, month: str | None = None) -> list[dict]:
        """Return manifest run entries, optionally filtered by split and month."""
        return [
            r for r in self.manifest["runs"]
            if (split is None or r["split"] == split)
            and (month is None or r["month"] == month)
        ]

    def splits(self) -> list[str]:
        """Return sorted split names present in the manifest."""
        return sorted({r["split"] for r in self.manifest["runs"]})

    def load_run(self, run: dict | str) -> dict[str, np.ndarray]:
        """Load every array of one outlook folder. ``run`` is a manifest entry or relative path."""
        rel = run["path"] if isinstance(run, dict) else run
        path = self.root / rel
        if not path.is_file():
            raise CacheError(f"Cache file missing: {path}. Rebuild with: python build_cache.py")
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def iter_runs(
        self,
        split: str | None = None,
        month: str | None = None,
    ) -> Iterator[tuple[dict, dict[str, np.ndarray]]]:
        """Yield ``(manifest_entry, arrays)`` one run at a time to keep memory flat."""
        for run in self.runs(split, month):
            yield run, self.load_run(run)

    def feature_index(self, name: str) -> int:
        """Return axis-2 index in ``values`` for a numeric feature name."""
        try:
            return self.numeric_features.index(name)
        except ValueError:
            raise KeyError(
                f"{name!r} is not a numeric feature; choose from {self.numeric_features}"
            ) from None

    def decode(self, field: str, codes: np.ndarray) -> np.ndarray:
        """Map integer codes back to labels; ``MISSING_CODE`` (-1) becomes ``''``."""
        table = np.array(list(self.vocab[field]) + [""])
        return table[np.asarray(codes)]


def valid_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Return ``(L, T)`` bool mask that is True for real (non-padding) steps."""
    _, n_t = arrays["times"].shape
    return np.arange(n_t)[None, :] < arrays["n_steps"][:, None]


def split_weather(label: str) -> tuple[str, int | None, str | None]:
    """Split a raw weather label: ``'Cloud2D'`` -> ``('Cloud', 2, 'D')``."""
    m = _WEATHER_SPLIT_RE.match((label or "").strip())
    if not m:
        return label, None, None
    return m.group(1), (int(m.group(2)) if m.group(2) else None), m.group(3)


def normalize_weather_class(label: str) -> str | None:
    """Map raw weather label to ``'Clear'``, ``'Cloud'``, ``'Rain'``, or ``None``."""
    cleaned = (label or "").strip()
    if not cleaned:
        return None
    m = _WEATHER_CLASS_RE.match(cleaned)
    if not m:
        return None
    return _WEATHER_ALIASES.get(m.group(1).lower())
