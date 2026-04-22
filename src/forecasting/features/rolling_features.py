from __future__ import annotations

import pandas as pd


def add_rolling_features(df: pd.DataFrame, targets: list[str], windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    for target in targets:
        shifted = out[target].shift(1)
        for window in windows:
            out[f"{target}_roll_mean_{window}"] = shifted.rolling(window).mean()
            out[f"{target}_roll_std_{window}"] = shifted.rolling(window).std()
            out[f"{target}_roll_min_{window}"] = shifted.rolling(window).min()
            out[f"{target}_roll_max_{window}"] = shifted.rolling(window).max()
    return out
