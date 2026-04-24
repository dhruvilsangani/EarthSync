# EarthSync IEX DAM Forecasting

Notebook-first forecasting workspace for IEX Day-Ahead Market buy/sell bid volume prediction at 15-minute granularity.

## Project structure

- `notebooks/01_eda`: data extraction and exploratory analysis.
- `notebooks/02_features`: feature engineering (e.g. interim feature tables aligned with `src/forecasting/features`).
- `notebooks/03_models`: tuning and experiments — `classical_ts` (StatsForecast / skforecast-style workflows), `boosting` (LightGBM), `lstm` (TensorFlow).
- `notebooks/04_benchmarking`: model comparison (`benchmark_models.ipynb`).
- `notebooks/05_submission`: submission-oriented forecast notebooks (e.g. time-series results for purchase vs sell bids under `ts-forecast/`).
- `src/forecasting`: reusable Python modules for data loading, features, models, evaluation, and plotting.
- `scripts`: CLI runners for train, backtest, forecast, and submission packaging (see below).
- `data/raw`: primary IEX DAM extract (default `iex-dam-0201-0421.csv`).
- `data/interim`: derived tables such as engineered features (`iex_dam_features.csv`).
- `data/predictions`: per-model or per-run forecast CSVs (including dated exports used for benchmarking or submission).
- `data/weather`: supplementary weather series (e.g. `weather_master.csv`) for feature work.
- `reports/results-all.ipynb`: consolidated results notebook.
- `reports/figures`, `reports/benchmarking`: optional outputs for plots and metrics (e.g. backtest CSVs from `run_backtest.py`).
- `logs/ai_prompts.md`: prompt interaction log scaffold.

## Setup

1. Create and activate a Python 3.10+ environment (the repo is exercised on 3.12 as well).
2. Install dependencies: `pip install -r requirements.txt`
3. Place the market CSV in `data/raw/` (default path: `data/raw/iex-dam-0201-0421.csv`).

To use `src.forecasting` from notebooks or REPL, add the project root to `PYTHONPATH` or run notebooks with the project root as the working directory so `import forecasting` resolves via `src/`.

## Runner scripts

CLI scripts use the **recursive LightGBM** forecaster in `src/forecasting`. Classical time-series, boosting tuning, and LSTM workflows live in `notebooks/03_models`.

All scripts accept `--target purchase_bid` or `--target sell_bid`.

- Train (writes `data/processed/model_lgbm.pkl` by default):

  `python scripts/run_train.py --target purchase_bid`

- Backtest (writes metrics CSV under `reports/benchmarking/` by default):

  `python scripts/run_backtest.py --target purchase_bid`

- Forecast (writes CSV under `data/predictions/` with a date-stamped filename):

  `python scripts/run_forecast.py --target purchase_bid`

- Build combined submission CSV from separate purchase/sell prediction files:

  `python scripts/build_submission.py --purchase-file <purchase_csv> --sell-file <sell_csv>`

## Notebook workflow

Use notebooks for experimentation and interpretation. Move repeated logic into `src/forecasting` and import it so cells stay short and reproducible. Optional: pair notebooks with [Jupytext](https://jupytext.readthedocs.io/) (listed in `requirements.txt`) if you want text-backed versions.

## Deliverables mapping

- EDA: `notebooks/01_eda`.
- Features: `notebooks/02_features` and `src/forecasting/features`.
- Model code and tuning: `notebooks/03_models` and `src/forecasting`.
- Forecast CSVs: `data/predictions`.
- Benchmarking and visuals: `notebooks/04_benchmarking`, `reports/results-all.ipynb`, and `reports/` subfolders as generated.
- AI prompts log: `logs/ai_prompts.md`.
