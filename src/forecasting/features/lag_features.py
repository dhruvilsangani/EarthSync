from __future__ import annotations

import pandas as pd


def add_lag_features(df: pd.DataFrame, targets: list[str], lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for target in targets:
        for lag in lags:
            out[f"{target}_lag_{lag}"] = out[target].shift(lag)
    return out
