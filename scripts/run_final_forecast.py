from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forecasting.config import load_config
from src.forecasting.data import load_dataset
from src.forecasting.pipeline import run_backtest_for_target, train_final_and_forecast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final 7-day forecast output.")
    parser.add_argument("--config", default="configs/base.yaml", help="YAML config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    df = load_dataset(config["data"]["input_csv"], config["data"]["timestamp_col"])
    holdout_start = pd.Timestamp(config["forecast"]["holdout_start"])
    holdout_end = pd.Timestamp(config["forecast"]["holdout_end"])
    holdout_rows = df[(df[config["data"]["timestamp_col"]] >= holdout_start) & (df[config["data"]["timestamp_col"]] <= holdout_end)]
    if len(holdout_rows) < int(config["forecast"]["horizon_steps"]):
        raise ValueError(
            f"Holdout window has {len(holdout_rows)} rows, expected at least {config['forecast']['horizon_steps']}."
        )

    forecast_frames = []
    for target in config["targets"]:
        backtest = run_backtest_for_target(df, config, target)
        best_model = backtest.iloc[0]["model"]
        print(f"Selected best model for {target}: {best_model}")
        forecast_frames.append(train_final_and_forecast(df, config, target, best_model_name=best_model))

    final_df = pd.concat(forecast_frames, ignore_index=True)
    out_dir = Path(config["paths"]["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "forecasts_2026-04-15_2026-04-21.csv"
    final_df.to_csv(out_path, index=False)
    print(f"Saved forecast to {out_path}")


if __name__ == "__main__":
    main()
