"""Reusable backtesting wrappers."""

import pandas as pd
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster


def run_backtest(
    forecaster,
    y: pd.Series,
    steps: int,
    initial_train_size: int,
    metric: str = "mean_absolute_error",
    exog: pd.DataFrame | None = None,
):
    cv = TimeSeriesFold(steps=steps, initial_train_size=initial_train_size, refit=False)
    metric_result, predictions = backtesting_forecaster(
        forecaster=forecaster,
        y=y,
        exog=exog,
        cv=cv,
        metric=metric,
        verbose=False,
    )
    return metric_result, predictions

