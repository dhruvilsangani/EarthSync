#!/usr/bin/env python3
"""Train selected model family for a target."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from forecasting.config.defaults import DEFAULT_RANDOM_SEED, RAW_DATA_DIR
from forecasting.data.loaders import load_market_data
from forecasting.models.lgbm_models import build_recursive_lgbm
from forecasting.models.pipeline import select_target_series
from forecasting.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["purchase_bid", "sell_bid"])
    parser.add_argument("--model", default="lgbm", choices=["lgbm"])
    parser.add_argument("--data-path", default=str(RAW_DATA_DIR / "iex-dam-0201-0421.csv"))
    parser.add_argument("--output", default="data/processed/model_lgbm.pkl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(DEFAULT_RANDOM_SEED)

    data = load_market_data(args.data_path)
    target_series = select_target_series(data, args.target)
    model = build_recursive_lgbm()
    model.fit(y=target_series)

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file_obj:
        pickle.dump(model, file_obj)
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()

