"""Validation helpers for consistent checks."""

from forecasting.data.schema import TARGET_COLUMNS


def validate_target(target: str) -> None:
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Invalid target '{target}'. Choose one of: {TARGET_COLUMNS}.")

