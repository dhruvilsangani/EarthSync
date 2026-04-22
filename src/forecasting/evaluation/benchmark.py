"""Benchmark aggregation utilities."""

import pandas as pd


def benchmark_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    sort_cols = [col for col in ("target", "model", "mae") if col in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols).reset_index(drop=True)
    return frame

