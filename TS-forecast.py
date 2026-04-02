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

from statsmodels.tsa.seasonal import STL

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utilsforecast.plotting import plot_series
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import * # mae imported

# %%
# import the ADF test
from statsmodels.tsa.stattools import adfuller

# create a function that returns the necessary metrics to test stationarity
def test_stationarity(timeseries):
    dftest_initial = adfuller(timeseries)
    dfoutput_initial = pd.Series(dftest_initial[0:4], 
          index=['Statistical Test', 
                 'p-value', 
                 '#Lags used', 
                 'Number of observations'
                 ])
    for key, value in dftest_initial[4].items():
        dfoutput_initial['Critical value ' + key] = value
    print(dfoutput_initial)
    print('\n')


# %%
data = pd.read_csv("../../Downloads/iex_dam_feb_mar_2026.csv")

# %%
data = data.rename(columns={
    'period_start': 'ds',  # Your timestamp
    'purchase_bid': 'y'             # The target value you want to plot (Market Clearing Price)
})
filtered_data = data[[ "ds", "y" ]]
filtered_data['unique_id'] = 'series_1'
filtered_data["ds"] = pd.to_datetime(filtered_data["ds"])

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(filtered_data["y"].diff(96).dropna(), lags=200)
plot_pacf(filtered_data["y"].diff(96).dropna(), lags=200)

# %%

# %%

# %%
from statsforecast import StatsForecast
from statsforecast.models import Naive, HistoricAverage, WindowAverage, SeasonalNaive
from statsforecast.models import AutoARIMA, ARIMA

# %% [markdown]
# ### predict a day in future

# %%
horizon = 96*7

models = [
    Naive(),
    HistoricAverage(),
    WindowAverage(window_size=96),
    SeasonalNaive(season_length=96)
]

sf = StatsForecast(models=models, freq="15min")
sf.fit(df=filtered_data)
preds = sf.predict(h=horizon)

# %%
preds

# %%
plot_series(
    df=filtered_data, 
    forecasts_df=preds,  
    max_insample_length=300, 
    palette="viridis")

# %%
test = filtered_data.tail(horizon)
train = filtered_data.drop(test.index).reset_index(drop=True)

# %%
sf.fit(df=train)

preds = sf.predict(h=horizon)

eval_df = pd.merge(test, preds, 'left', ['ds', 'unique_id'])

# %%
evaluation = evaluate(
    eval_df,
    metrics=[mae],
)
evaluation.head()

# %%
evaluation = evaluation.drop(['unique_id'], axis=1).groupby('metric').mean().reset_index()
evaluation

# %%
methods = evaluation.columns[1:].tolist()  
values = evaluation.iloc[0, 1:].tolist() 

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, values)

for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.xlabel('Methods')
plt.ylabel('Mean absolute error (MAE)')
plt.tight_layout()

plt.show()

# %%
plot_series(
    df=filtered_data, 
    forecasts_df=preds,  
    max_insample_length=1000, 
    palette="viridis")

# %%

# %%

# %% [markdown]
# ## Real business

# %%
models = [
    # AutoARIMA(seasonal=False, alias="ARIMA"),
    # AutoARIMA(season_length=96, seasonal=True)
    # ARIMA(order=(2,1,3), season_length=96, seasonal_order=(3, 1, 1), alias="SARIMA (2,1,3,3,1,1,96)"),
    # ARIMA(order=(1,1,1), season_length=96, seasonal_order=(1, 1, 1), alias="SARIMA (1,1,1,1,1,1,96)")
    ARIMA(order=(1,1,1), season_length=96, seasonal_order=(1, 1, 1), alias="SARIMA (1,1,1,1,1,1,96)")
]

sf = StatsForecast(models=models, freq="15min")
sf.fit(df=train)
arima_preds = sf.predict(h=horizon)

arima_eval_df = pd.merge(arima_preds, eval_df, 'inner', ['ds', 'unique_id'])
arima_eval = evaluate(
    arima_eval_df,
    metrics=[mae],
)
arima_eval

# %%
# arima_eval = arima_eval.drop(['unique_id'], axis=1).groupby('metric').mean().reset_index()
arima_eval

# %%

# %%

# %%

# %%

# %%
plot_series(
    df=filtered_data, 
    forecasts_df=arima_eval_df[["unique_id", "ds", "SARIMA (1,1,1,1,1,1,96)"]], 
    max_insample_length=1000, 
    palette="viridis")

# %%
methods = arima_eval.columns[1:].tolist()  
values = arima_eval.iloc[0, 1:].tolist() 

sorted_data = sorted(zip(methods, values), key=lambda x: x[1], reverse=True)
methods_sorted, values_sorted = zip(*sorted_data)

plt.figure(figsize=(10, 6))
bars = plt.bar(methods_sorted, values_sorted)

for bar, value in zip(bars, values_sorted):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.xlabel('Methods')
plt.ylabel('Mean absolute error (MAE)')
plt.tight_layout()

plt.show()

# %%
SARIMA(0,0,1)(0,1,1)96 -> 2223.741567
SARIMA(4,1,1)(0,0,0)1 -> 2632.164
SARIMA(2,1,3)(3,1,1)96 -> 1167.938
SARIMA(1,1,1)(1,1,1)96 -> 1124.649

# %%
models = [
    SeasonalNaive(season_length=96),
    # AutoARIMA(seasonal=False, alias="ARIMA"),
    # AutoARIMA(season_length=96, seasonal=True)
    # ARIMA(order=(2,1,3), season_length=96, seasonal_order=(3, 1, 1), alias="SARIMA (2,1,3,3,1,1,96)"),
    ARIMA(order=(1,1,1), season_length=96, seasonal_order=(1, 1, 1), alias="SARIMA (1,1,1,1,1,1,96)")
]

sf = StatsForecast(models=models, freq="15min")
cv_df = sf.cross_validation(
    h=horizon, # 7 days
    df=filtered_data,
    n_windows=7,
    step_size=horizon,
    refit=True
)

cv_df.head()

# %%
plot_series(
    df=filtered_data, 
    forecasts_df=cv_df[96*5:96*6].drop(["y", "cutoff"], axis=1), 
    max_insample_length=900, 
    palette="viridis")

# %%
cv_eval = evaluate(
    cv_df.drop(["cutoff"], axis=1),
    metrics=[mae],
)
cv_eval = cv_eval.drop(['unique_id'], axis=1).groupby('metric').mean().reset_index()
cv_eval

# %%
cv_df#[cv_df.ds.str.contains("2026-03-23")]

# %%
# 1. Filter for March 23rd
# Since 'ds' is already a datetime object from your previous steps
march_23_df = cv_df[cv_df['ds'].dt.date == pd.to_datetime('2026-03-24').date()]

# 2. Calculate MAE for y and SARIMA
from sklearn.metrics import mean_absolute_error

# Use the exact column name from your SARIMA model
mae_23rd = mean_absolute_error(march_23_df['y'], march_23_df['SARIMA (1,1,1,1,1,1,96)'])

print(f"MAE for March 23rd: {mae_23rd:.4f}")

# Optional: View the first few rows of the filtered data
march_23_df.head()

# %%
