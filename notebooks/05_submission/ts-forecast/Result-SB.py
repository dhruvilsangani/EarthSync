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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from utilsforecast.plotting import plot_series
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import * # mae imported

from statsforecast import StatsForecast
from statsforecast.models import ARIMA

# %%
data = pd.read_csv("../../Downloads/iex-dam-0201-0421.csv")
data = data[(data['period_start'] >= '2026-02-01') & (data['period_start'] <= '2026-04-21 23:50')]
data

# %%
data = data.rename(columns={
    'period_start': 'ds',  # Your timestamp
    'sell_bid': 'y'             # The target value you want to plot (Market Clearing Price)
})
filtered_data = data[[ "ds", "y" ]].copy()
filtered_data['unique_id'] = 'series_1'
filtered_data["ds"] = pd.to_datetime(filtered_data["ds"])

# %% [markdown]
# ### SARIMA with Exogeneous variables - last 7 days

# %%
horizon = 96*7

data_with_exog = filtered_data.copy()
data_with_exog["weekday_enum"] = data_with_exog["ds"].dt.weekday + 1

test = data_with_exog.groupby("unique_id").tail(horizon)
train = data_with_exog.drop(test.index).reset_index(drop=True)

futr_exog_df = test.drop(["y"], axis=1)
futr_exog_df.head()

# %%
models = [
    ARIMA(order=(0,1,0), season_length=96, seasonal_order=(0, 1, 1), alias="SARIMA (0,1,0,0,1,1,96) exog")
]

sf = StatsForecast(models=models, freq="15min")
sf.fit(df=train)
preds = sf.predict(h=horizon, X_df=futr_exog_df)

model_col = "SARIMA (0,1,0,0,1,1,96) exog"

# Force all values below 0 to be exactly 0
preds[model_col] = preds[model_col].clip(lower=0)

test_w_preds = pd.merge(test, preds, 'left', ['ds', 'unique_id'])

# %%
metrics = evaluate(
    test_w_preds.drop(columns="weekday_enum"),
    metrics=[mae],
)
metrics

# %%
plot_series(
    df=filtered_data, 
    forecasts_df=preds,  
    max_insample_length=1000, 
    palette="viridis")

# %%
start_mmdd = preds['ds'].min().strftime('%m%d')
end_mmdd = preds['ds'].max().strftime('%m%d')

# 3. Construct the filename
filename = f"predictions_SB_{start_mmdd}_{end_mmdd}.csv"

# 4. Save to CSV
test_w_preds.drop(columns=["unique_id", "weekday_enum"]).to_csv(filename, index=False)

print(f"File saved successfully as: {filename}")

# %%
