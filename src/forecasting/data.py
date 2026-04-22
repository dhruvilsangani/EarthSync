from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

RAW_REQUIRED_COLUMNS = [
    "period_start",
    "period",
    "purchase_bid",
    "sell_bid",
    "mcv",
    "mcp",
    "final_scheduled_volume",
]


def load_dataset(csv_path: str | Path, timestamp_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    return df


def ensure_required_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_and_validate_raw_dataset(
    csv_path: str | Path,
    timestamp_col: str,
    expected_frequency: str,
) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    ensure_required_columns(df, RAW_REQUIRED_COLUMNS)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).drop_duplicates(subset=[timestamp_col], keep="last").reset_index(drop=True)
    gap_report = validate_time_cadence(df, timestamp_col, expected_frequency)

    metadata = {
        "rows_after_cleanup": int(len(df)),
        "duplicates_removed": int(len(pd.read_csv(csv_path)) - len(df)),
        "gap_count": int(gap_report["gap_count"]),
        "gap_examples": gap_report["gap_examples"],
        "start_timestamp": str(df[timestamp_col].min()),
        "end_timestamp": str(df[timestamp_col].max()),
    }
    return df, metadata


def validate_time_cadence(df: pd.DataFrame, timestamp_col: str, expected_frequency: str) -> dict:
    expected_delta = pd.to_timedelta(expected_frequency)
    deltas = df[timestamp_col].diff().dropna()
    bad = deltas[deltas != expected_delta]
    gap_examples = []
    if not bad.empty:
        idxs = bad.index[:5]
        for idx in idxs:
            gap_examples.append(
                {
                    "previous": str(df.loc[idx - 1, timestamp_col]),
                    "current": str(df.loc[idx, timestamp_col]),
                    "delta": str(df.loc[idx, timestamp_col] - df.loc[idx - 1, timestamp_col]),
                }
            )
    return {"gap_count": int(len(bad)), "gap_examples": gap_examples}


def select_model_frame(
    df: pd.DataFrame,
    target_col: str,
    timestamp_col: str,
    shared_feature_cols: list[str],
) -> pd.DataFrame:
    keep_cols = [timestamp_col, target_col, *shared_feature_cols]
    keep_cols += [c for c in df.columns if c.startswith(f"{target_col}_lag_")]
    keep_cols += [c for c in df.columns if c.startswith(f"{target_col}_roll_")]
    model_df = df[keep_cols].dropna().reset_index(drop=True)
    return model_df
