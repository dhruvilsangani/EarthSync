from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(
    df: pd.DataFrame,
    timestamp_col: str,
    is_evening_start_block: int = 73,
    is_evening_end_block: int = 88,
) -> pd.DataFrame:
    out = df.copy()
    ts = out[timestamp_col]
    out["period_enum"] = ts.dt.hour * 4 + ts.dt.minute // 15 + 1
    out["weekday_enum"] = ts.dt.weekday + 1
    out["is_weekend"] = out["weekday_enum"].isin([6, 7]).astype(int)
    out["is_evening"] = out["period_enum"].between(is_evening_start_block, is_evening_end_block).astype(int)

    # Cyclical encodings preserve periodic continuity (e.g., 96->1).
    out["period_sin"] = np.sin(2 * np.pi * out["period_enum"] / 96.0)
    out["period_cos"] = np.cos(2 * np.pi * out["period_enum"] / 96.0)
    out["weekday_sin"] = np.sin(2 * np.pi * out["weekday_enum"] / 7.0)
    out["weekday_cos"] = np.cos(2 * np.pi * out["weekday_enum"] / 7.0)
    return out
