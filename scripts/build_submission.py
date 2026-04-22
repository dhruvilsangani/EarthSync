#!/usr/bin/env python3
"""Build assignment submission forecast CSV from target files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--purchase-file", required=True)
    parser.add_argument("--sell-file", required=True)
    parser.add_argument("--output", default="data/predictions/submission_1week.csv")
    return parser.parse_args()


def _load(path: str, target_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    first_col = frame.columns[0]
    value_col = frame.columns[1]
    frame = frame.rename(columns={first_col: "period_start", value_col: target_col})
    return frame[["period_start", target_col]]


def main() -> None:
    args = parse_args()
    purchase = _load(args.purchase_file, "purchase_bid_pred")
    sell = _load(args.sell_file, "sell_bid_pred")
    out = purchase.merge(sell, on="period_start", how="inner")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"Saved submission file to {output}")


if __name__ == "__main__":
    main()

