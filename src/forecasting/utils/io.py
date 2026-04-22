"""I/O helpers for shared scripts and notebooks."""

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_predictions(
    predictions: pd.Series | pd.DataFrame,
    output_path: Path,
    timestamp_col: str = "period_start",
    value_col: str = "prediction",
) -> Path:
    """Save a forecast series/dataframe with standard naming."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(predictions, pd.Series):
        frame = predictions.to_frame(name=value_col).reset_index()
    else:
        frame = predictions.reset_index()

    if frame.columns[0] != timestamp_col:
        frame = frame.rename(columns={frame.columns[0]: timestamp_col})

    frame.to_csv(output_path, index=False)
    return output_path


def save_json(payload: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_json(input_path: Path) -> dict:
    return json.loads(input_path.read_text(encoding="utf-8"))

