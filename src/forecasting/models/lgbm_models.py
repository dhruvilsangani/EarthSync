"""LightGBM forecaster factories."""

from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive

from forecasting.config.defaults import DEFAULT_RANDOM_SEED
from forecasting.features.rolling import default_rolling_mean


def build_recursive_lgbm(lags: int = 96 * 3, random_state: int = DEFAULT_RANDOM_SEED):
    return ForecasterRecursive(
        estimator=LGBMRegressor(random_state=random_state, verbose=-1),
        lags=lags,
        window_features=default_rolling_mean(),
    )

