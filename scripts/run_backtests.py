from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forecasting.config import load_config
from src.forecasting.data import ensure_required_columns, load_dataset
from src.forecasting.pipeline import run_backtest_for_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model backtests for IEX forecasting.")
    parser.add_argument("--config", default="configs/base.yaml", help="YAML config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    df = load_dataset(config["data"]["input_csv"], config["data"]["timestamp_col"])

    ensure_required_columns(
        df,
        required_cols=[
            config["data"]["timestamp_col"],
            "purchase_bid",
            "sell_bid",
            *config["features"]["shared"],
        ],
    )

    holdout_start = pd.Timestamp(config["forecast"]["holdout_start"])
    holdout_end = pd.Timestamp(config["forecast"]["holdout_end"])
    holdout_rows = df[(df[config["data"]["timestamp_col"]] >= holdout_start) & (df[config["data"]["timestamp_col"]] <= holdout_end)]
    if len(holdout_rows) < int(config["forecast"]["horizon_steps"]):
        raise ValueError(
            f"Holdout window has {len(holdout_rows)} rows, expected at least {config['forecast']['horizon_steps']}."
        )

    all_results = []
    for target in config["targets"]:
        result = run_backtest_for_target(df, config, target)
        all_results.append(result)
        print(f"\nTop models for {target}:")
        print(result.head(3).to_string(index=False))

    metrics_df = pd.concat(all_results, ignore_index=True)
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts_dir / "benchmark_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved benchmark metrics to {metrics_path}")


if __name__ == "__main__":
    main()
