"""Default project paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
BENCHMARK_DIR = REPORTS_DIR / "benchmarking"
LOGS_DIR = PROJECT_ROOT / "logs"

DEFAULT_FREQ = "15min"
DEFAULT_HORIZON = 96 * 7
DEFAULT_RANDOM_SEED = 15926

