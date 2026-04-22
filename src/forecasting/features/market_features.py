from __future__ import annotations

import numpy as np
import pandas as pd


def add_market_features(df: pd.DataFrame, include_prev_day_features: bool = True) -> pd.DataFrame:
    out = df.copy()
    out["purchase_minus_sell"] = out["purchase_bid"] - out["sell_bid"]
    out["purchase_to_sell_ratio"] = out["purchase_bid"] / np.clip(out["sell_bid"], 1e-6, None)
    out["uncleared_volume"] = out["purchase_bid"] - out["mcv"]
    out["cleared_ratio"] = out["mcv"] / np.clip(out["purchase_bid"], 1e-6, None)
    if include_prev_day_features:
        out["prev_day_purchase_bid"] = out["purchase_bid"].shift(96)
        out["prev_day_uncleared_vol"] = out["uncleared_volume"].shift(96)
    return out
