from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
TRAIN_ROOT = DATASET_ROOT / "training"
TEST_ROOT = DATASET_ROOT / "testing"
OUTLOOK_DIR_GLOB = "weather-outlook_d01_*_pst"
DATA_FILE_PREFIX = "MUN_"
IGNORED_SUFFIXES = (".json",)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / "cache"

LEARNING_RATE = 0.1
BIAS_INIT = 0.5
MAX_EPOCHS = 30
SEED = 170
INCLUDE_RAINFALL = False

CLASSES = ("Clear", "Cloud", "Rain")  # fixed order everywhere (tie-break, matrix rows/cols)

EXPECTED_HEADER = (
    "Date",
    "Rainfall(mm)",
    "2m Temp(C)",
    "Heat Index(C)",
    "Relative Humidity(%)",
    "Wind Speed(m/s)",
    "Wind Direction",
    "Pressure(hPa)",
    "Weather",
)

BASE_FEATURES = (
    "Pressure",
    "Temp_min",
    "Temp_max",
    "HeatIndex_min",
    "HeatIndex_max",
    "RH_min",
    "RH_max",
    "WindSpeed_min",
    "WindSpeed_max",
)
RAINFALL_FEATURE = "Rainfall"  # appended last only if INCLUDE_RAINFALL is True

FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "Pressure": (800.0, 1100.0),
    "Temp_min": (0.0, 60.0),
    "Temp_max": (0.0, 60.0),
    "HeatIndex_min": (-20.0, 80.0),
    "HeatIndex_max": (-20.0, 80.0),
    "RH_min": (0.0, 200.0),
    "RH_max": (0.0, 200.0),
    "WindSpeed_min": (0.0, 100.0),
    "WindSpeed_max": (0.0, 100.0),
    "Rainfall": (0.0, 2000.0),  # active only when --include-rainfall is set
}
