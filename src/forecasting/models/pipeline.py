"""Shared target-driven model pipeline helpers."""

import pandas as pd

from forecasting.features.validators import validate_target


def select_target_series(data: pd.DataFrame, target: str) -> pd.Series:
    validate_target(target)
    return data[target]

