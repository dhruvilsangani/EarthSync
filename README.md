# IEX DAM Forecasting Pipeline

This repository provides a structured, reproducible workflow for forecasting IEX DAM buy and sell bid volumes at 15-minute granularity.

## Goals

- Build leakage-safe features and validation folds.
- Benchmark multiple model families.
- Generate final forecast for `2026-04-15` to `2026-04-21` (7 days, 672 blocks).

## Project Structure

- `configs/`: Runtime and model configuration.
- `src/`: Reusable pipeline modules (features, validation, models, evaluation).
- `scripts/`: CLI entry points for feature build, backtesting, and final forecast.
- `artifacts/`: Generated models and diagnostics.
- `outputs/`: Submission-ready forecast outputs.

## Quick Start

1. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Build processed dataset from raw DAM file:

   ```bash
   python scripts/run_feature_build.py --config configs/base.yaml
   ```

   By default this reads `iex-dam-0201-0421.csv` and writes:
   - `data/processed/iex_dam_modeling.csv`
   - `data/processed/feature_build_report.json`
   - `data/external/weather_nci.csv` (API fetched once, then reused from cache)

3. Run backtests:

   ```bash
   python scripts/run_backtests.py --config configs/base.yaml
   ```

4. Generate final forecasts (`15 Apr` to `21 Apr`):

   ```bash
   python scripts/run_final_forecast.py --config configs/base.yaml
   ```

## Notes

- The pipeline runs for both targets: `purchase_bid` and `sell_bid`.
- Features are built in a modular raw-to-feature flow (`time`, `market`, `weather`, `lag`, `rolling`) configured via `configs/base.yaml`.
- Core notebook-style features are included: `is_evening`, `prev_day_purchase_bid`, `prev_day_uncleared_vol`.
- Weather-derived features are included with API+cache: `NCI`, `cooling_load_factor`, `rolling_3d_clf`.
- It tries `lightgbm`, `xgboost`, and `lstm` if installed, and gracefully falls back when unavailable.
- All timestamps are treated at 15-minute frequency.
