# EarthSync IEX DAM Forecasting

Notebook-first forecasting workspace for IEX Day-Ahead Market buy/sell bid volume prediction at 15-minute granularity.

## Project Structure

- `notebooks/01_eda`: data extraction and exploratory analysis notebooks.
- `notebooks/03_models`: experiment notebooks for classical time-series, LightGBM, and LSTM.
- `notebooks/04_benchmarking`: result comparison and analysis notebooks.
- `src/forecasting`: reusable Python modules for data loading, features, models, evaluation, and plotting.
- `scripts`: lightweight runners for train, backtest, forecast, and submission packaging.
- `data/raw`, `data/interim`, `data/processed`, `data/predictions`: data lifecycle folders.
- `reports/figures`, `reports/benchmarking`: generated charts and metric outputs.
- `logs/ai_prompts.md`: mandatory prompt interaction log scaffold.

## Setup

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Keep your data in `data/raw/` (default runner path expects `data/raw/iex-dam-0201-0421.csv`).

## Runner Scripts

All scripts support `--target purchase_bid` or `--target sell_bid`.

- Train:
  - `python scripts/run_train.py --target purchase_bid`
- Backtest:
  - `python scripts/run_backtest.py --target purchase_bid`
- Forecast:
  - `python scripts/run_forecast.py --target purchase_bid`
- Build combined submission:
  - `python scripts/build_submission.py --purchase-file <purchase_csv> --sell-file <sell_csv>`

## Notebook Workflow

Use notebooks for experimentation and interpretation. Move repeated logic into `src/forecasting` and import it into notebooks to keep cells concise and reproducible.

## Deliverables Mapping

- EDA report: notebooks under `notebooks/01_eda`.
- Model code + tuning: notebooks and `src/forecasting`.
- Forecast CSV outputs: `data/predictions`.
- Benchmark report and visuals: `notebooks/04_benchmarking` and `reports/`.
- AI prompts log: `logs/ai_prompts.md`.

