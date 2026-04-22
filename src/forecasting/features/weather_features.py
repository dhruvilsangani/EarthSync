from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import requests


def get_nci_data(start_date: str, end_date: str, locations: Dict[str, dict]) -> pd.DataFrame:
    all_weather = pd.DataFrame()
    for name, loc in locations.items():
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max",
            "timezone": "Asia/Kolkata",
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        temp_df = pd.DataFrame(
            {
                "date": pd.to_datetime(payload["daily"]["time"]),
                f"temp_{name}": payload["daily"]["temperature_2m_max"],
            }
        )
        all_weather = temp_df if all_weather.empty else all_weather.merge(temp_df, on="date", how="inner")

    all_weather["NCI"] = 0.0
    for name, loc in locations.items():
        all_weather["NCI"] += all_weather[f"temp_{name}"] * float(loc["weight"])
    all_weather["date_only"] = all_weather["date"].dt.date
    return all_weather


def fetch_or_load_weather_cache(
    cache_path: str | Path,
    start_date: str,
    end_date: str,
    locations: Dict[str, dict],
    refresh: bool = False,
) -> tuple[pd.DataFrame, str]:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not refresh:
        cached = pd.read_csv(path)
        cached["date"] = pd.to_datetime(cached["date"])
        cached["date_only"] = pd.to_datetime(cached["date_only"]).dt.date
        return cached, "cache_hit"

    weather_df = get_nci_data(start_date=start_date, end_date=end_date, locations=locations)
    weather_df.to_csv(path, index=False)
    return weather_df, "api_fetch"


def merge_weather_features(df: pd.DataFrame, weather_df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    out = df.copy()
    out["date_only"] = pd.to_datetime(out[timestamp_col]).dt.date
    merged = out.merge(
        weather_df[["date_only", "NCI", "temp_delhi", "temp_mumbai", "temp_akola"]],
        on="date_only",
        how="left",
    )
    merged["cooling_load_factor"] = (merged["NCI"] - 30).clip(lower=0) ** 2
    daily_cooling = (
        merged.groupby("date_only", as_index=False)["cooling_load_factor"]
        .mean()
        .sort_values("date_only")
    )
    daily_cooling["rolling_3d_clf"] = daily_cooling["cooling_load_factor"].rolling(window=3, min_periods=1).mean()
    merged = merged.merge(daily_cooling[["date_only", "rolling_3d_clf"]], on="date_only", how="left")
    return merged
