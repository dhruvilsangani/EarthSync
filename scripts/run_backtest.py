#!/usr/bin/env python3
"""Run backtesting for a selected target."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from forecasting.config.defaults import RAW_DATA_DIR
from forecasting.data.loaders import load_market_data
from forecasting.evaluation.backtesting import run_backtest
from forecasting.models.lgbm_models import build_recursive_lgbm
from forecasting.models.pipeline import select_target_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["purchase_bid", "sell_bid"])
    parser.add_argument("--steps", type=int, default=96 * 7)
    parser.add_argument("--data-path", default=str(RAW_DATA_DIR / "iex-dam-0201-0421.csv"))
    parser.add_argument("--output", default="reports/benchmarking/backtest_metrics.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_market_data(args.data_path)
    target_series = select_target_series(data, args.target)
    forecaster = build_recursive_lgbm()
    initial_train_size = len(data) - args.steps
    metrics, predictions = run_backtest(
        forecaster=forecaster,
        y=target_series,
        steps=args.steps,
        initial_train_size=initial_train_size,
    )

    metrics["target"] = args.target
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    print(f"Saved metrics to {output_path}")
    print(predictions.head())


if __name__ == "__main__":
    main()

