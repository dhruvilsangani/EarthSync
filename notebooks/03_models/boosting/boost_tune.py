# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (es_env)
#     language: python
#     name: es_env
# ---

# %%
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.append(str(PROJECT_ROOT / 'src'))

import pandas as pd
from lightgbm import LGBMRegressor
from skforecast.model_selection import TimeSeriesFold, bayesian_search_forecaster
from skforecast.recursive import ForecasterRecursive

from forecasting.data.loaders import load_market_data
from forecasting.features.calendar import build_time_features
from forecasting.features.rolling import default_rolling_mean
from forecasting.utils.io import save_json, load_json

# %%
TARGET_COL = 'purchase_bid'  # switch to 'sell_bid' when needed
HORIZON = 96 * 7
PARAMS_PATH = Path('data/processed/params/lgbm_params.json')

# %%
data_indexed = load_market_data('data/raw/iex-dam-0201-0421.csv')
exog = build_time_features(data_indexed.index)

# %%
window_features = default_rolling_mean(window_size=96 * 3)
forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(random_state=15926, verbose=-1),
    lags=96,
    window_features=window_features,
)


def search_space(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 200, 900, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'lags': trial.suggest_categorical('lags', [96, 192, 288, (1, 2, 3, 95, 96, 97)]),
    }


cv = TimeSeriesFold(steps=HORIZON, initial_train_size=len(data_indexed) - HORIZON * 2, refit=False)
results, _ = bayesian_search_forecaster(
    forecaster=forecaster,
    y=data_indexed[TARGET_COL],
    exog=exog,
    cv=cv,
    metric='mean_absolute_error',
    search_space=search_space,
    n_trials=12,
    return_best=True,
)

best = results.iloc[0]
best_payload = {
    'target': TARGET_COL,
    'model': 'lgbm',
    'best_params': {**best['params'], 'random_state': 15926, 'verbose': -1},
    'best_lags': best['lags'],
    'horizon': HORIZON,
}

existing = load_json(PARAMS_PATH) if PARAMS_PATH.exists() else {}
existing[TARGET_COL] = best_payload
save_json(existing, PARAMS_PATH)
best_payload
