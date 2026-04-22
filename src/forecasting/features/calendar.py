"""Calendar and cyclical feature utilities."""

import numpy as np
import pandas as pd


def build_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    features = pd.DataFrame(index=index)
    day_of_week = index.dayofweek + 1
    hour = index.hour
    features["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    features["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return features


def future_time_index(last_timestamp: pd.Timestamp, steps: int, freq: str = "15min") -> pd.DatetimeIndex:
    return pd.date_range(start=last_timestamp + pd.Timedelta(minutes=15), periods=steps, freq=freq)

