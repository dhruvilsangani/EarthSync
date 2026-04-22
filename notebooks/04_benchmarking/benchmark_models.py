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
from forecasting.data.loaders import load_market_data
from forecasting.evaluation.metrics import regression_metrics

# %%
TARGET_COL = 'purchase_bid'  # switch to 'sell_bid' when needed
PREDICTIONS_DIR = Path('data/predictions')

# %%
actual = load_market_data('data/raw/iex-dam-0201-0421.csv')[TARGET_COL]
rows = []
for file_path in sorted(PREDICTIONS_DIR.glob(f'{TARGET_COL}_forecast_*.csv')):
    pred = pd.read_csv(file_path)
    ts_col = pred.columns[0]
    y_col = pred.columns[1]
    pred[ts_col] = pd.to_datetime(pred[ts_col])
    merged = pd.DataFrame({'actual': actual}).reset_index().merge(
        pred[[ts_col, y_col]],
        left_on='period_start',
        right_on=ts_col,
        how='inner',
    )
    if merged.empty:
        continue
    metrics = regression_metrics(merged['actual'], merged[y_col])
    rows.append({'file': file_path.name, **metrics})

benchmark = pd.DataFrame(rows).sort_values('mae').reset_index(drop=True)
benchmark
