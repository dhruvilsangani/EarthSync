from __future__ import annotations

import pandas as pd

from .lag_features import add_lag_features
from .market_features import add_market_features
from .rolling_features import add_rolling_features
from .time_features import add_time_features
from .weather_features import fetch_or_load_weather_cache, merge_weather_features


def build_feature_frame(
    raw_df: pd.DataFrame,
    timestamp_col: str,
    targets: list[str],
    include_time: bool,
    include_market: bool,
    include_prev_day_features: bool,
    include_weather: bool,
    weather_cache_csv: str,
    weather_refresh_cache: bool,
    weather_locations: dict,
    is_evening_blocks: list[int],
    lags: list[int],
    rolling_windows: list[int],
    dropna_after_feature_build: bool = True,
) -> tuple[pd.DataFrame, dict]:
    df = raw_df.copy()
    diagnostics: dict = {"weather_cache_source": None}

    if include_time:
        df = add_time_features(
            df,
            timestamp_col=timestamp_col,
            is_evening_start_block=int(is_evening_blocks[0]),
            is_evening_end_block=int(is_evening_blocks[1]),
        )
    if include_market:
        df = add_market_features(df, include_prev_day_features=include_prev_day_features)
    if include_weather:
        start_date = pd.to_datetime(df[timestamp_col]).min().date().isoformat()
        end_date = pd.to_datetime(df[timestamp_col]).max().date().isoformat()
        weather_df, weather_source = fetch_or_load_weather_cache(
            cache_path=weather_cache_csv,
            start_date=start_date,
            end_date=end_date,
            locations=weather_locations,
            refresh=weather_refresh_cache,
        )
        diagnostics["weather_cache_source"] = weather_source
        diagnostics["weather_start_date"] = str(weather_df["date"].min())
        diagnostics["weather_end_date"] = str(weather_df["date"].max())
        df = merge_weather_features(df, weather_df, timestamp_col=timestamp_col)
        diagnostics["missing_weather_rows_after_merge"] = int(df["NCI"].isna().sum())
    if lags:
        df = add_lag_features(df, targets=targets, lags=lags)
    if rolling_windows:
        df = add_rolling_features(df, targets=targets, windows=rolling_windows)

    diagnostics["rows_before_dropna"] = int(len(df))
    if dropna_after_feature_build:
        df = df.dropna().reset_index(drop=True)
    diagnostics["rows_after_dropna"] = int(len(df))
    diagnostics["null_summary_by_column"] = {c: int(v) for c, v in df.isna().sum().to_dict().items() if int(v) > 0}
    return df, diagnostics
