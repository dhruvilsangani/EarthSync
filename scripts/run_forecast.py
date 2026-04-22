#!/usr/bin/env python3
"""Generate 7-day forecast for target."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from forecasting.config.defaults import DEFAULT_HORIZON, PREDICTIONS_DIR, RAW_DATA_DIR
from forecasting.data.loaders import load_market_data
from forecasting.models.lgbm_models import build_recursive_lgbm
from forecasting.models.pipeline import select_target_series
from forecasting.utils.io import save_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["purchase_bid", "sell_bid"])
    parser.add_argument("--steps", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--data-path", default=str(RAW_DATA_DIR / "iex-dam-0201-0421.csv"))
    parser.add_argument("--output-dir", default=str(PREDICTIONS_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_market_data(args.data_path)
    target_series = select_target_series(data, args.target)
    model = build_recursive_lgbm()
    model.fit(y=target_series)

    predictions = model.predict(steps=args.steps)

    start_str = predictions.index[0].strftime("%m%d")
    end_str = predictions.index[-1].strftime("%m%d")
    output_path = Path(args.output_dir) / f"{args.target}_forecast_{start_str}_{end_str}.csv"
    save_predictions(predictions, output_path, value_col=args.target)
    print(f"Saved forecast to {output_path}")


if __name__ == "__main__":
    main()

