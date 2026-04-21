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
from utilsforecast.losses import *

from feature_engine.datetime import DatetimeFeatures
from feature_engine.creation import CyclicalFeatures

from lightgbm import LGBMRegressor

from skforecast.preprocessing import RollingFeatures
from sklearn.feature_selection import RFECV
from skforecast.feature_selection import select_features

from skforecast.direct import ForecasterDirect
from skforecast.recursive import ForecasterEquivalentDate, ForecasterRecursive
from skforecast.model_selection import TimeSeriesFold, bayesian_search_forecaster, backtesting_forecaster

import warnings
warnings.filterwarnings('once')

# %%
data = pd.read_csv("../../Downloads/iex_dam_feb_mar_2026.csv")

# %%
data_indexed = data.set_index('period_start')
data_indexed = data_indexed.asfreq('15min')

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Naive Seasonal Forecast

# %%
forecaster = ForecasterEquivalentDate(
                 offset    = pd.DateOffset(days=1),
                 n_offsets = 1
             )
forecaster.fit(y=data_indexed.loc[:, 'purchase_bid'])
forecaster

# %%
cv = TimeSeriesFold(
        steps              = 96,
        initial_train_size = len(data_indexed)-96*7,
        refit              = False
)
metric, predictions = backtesting_forecaster(
                          forecaster = forecaster,
                          y          = data_indexed['purchase_bid'],
                          cv         = cv,
                          metric     = 'mean_absolute_error'
                       )

# %%
display(metric)
predictions.head()

# %%
fig = go.Figure()
trace1 = go.Scatter(x=data_indexed.index, y=data_indexed['purchase_bid'], name="test", mode="lines")
trace2 = go.Scatter(x=predictions.index, y=predictions['pred'], name="prediction", mode="lines")
fig.add_trace(trace1)
fig.add_trace(trace2)
fig.update_layout(
    title="Real value vs predicted in test data",
    xaxis_title="Date time",
    yaxis_title="Demand",
    width=800,
    height=400,
    margin=dict(l=20, r=20, t=35, b=20),
    legend=dict(orientation="h", yanchor="top", y=1.01, xanchor="left", x=0)
)
fig.show()

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### LightGBM Regressor

# %%
cv_lgbm = TimeSeriesFold(
        steps              = 96*7,
        initial_train_size = len(data_indexed)-96*7,
        refit              = False
)

# %%
# Create forecaster
# ==============================================================================
window_features = RollingFeatures(stats=["mean"], window_sizes=96 * 3)
forecaster = ForecasterRecursive(
                 estimator       = LGBMRegressor(random_state=15926, verbose=-1),
                 lags            = 96*3,
                 window_features = window_features
             )

# Train forecaster
# ==============================================================================
forecaster.fit(y=data_indexed.loc[:, 'purchase_bid'])
forecaster

# %%
metric, predictions = backtesting_forecaster(
                          forecaster    = forecaster,
                          y             = data_indexed['purchase_bid'],
                          cv            = cv_lgbm,
                          metric        = 'mean_absolute_error',
                          verbose       = True, # Set to False to avoid printing
                          return_predictors = True,
                      )

# %%
fig = go.Figure()
trace1 = go.Scatter(x=data_indexed.index, y=data_indexed['purchase_bid'], name="test", mode="lines")
trace2 = go.Scatter(x=predictions.index, y=predictions['pred'], name="prediction", mode="lines")
fig.add_trace(trace1)
fig.add_trace(trace2)
fig.update_layout(
    title="Real value vs predicted in test data",
    xaxis_title="Date time",
    yaxis_title="Demand",
    width=800,
    height=400,
    margin=dict(l=20, r=20, t=35, b=20),
    legend=dict(orientation="h", yanchor="top", y=1.01, xanchor="left", x=0)
)
fig.show()

# %%
# forecaster.get_feature_importances().head(10)
metric

# %% [markdown]
# ### Including exogenuous variable

# %%
features_to_extract = [
    'day_of_week',
    'hour'
]
max_values = {
    "day_of_week": 7,
    "hour": 24,
}
calendar_transformer = DatetimeFeatures(
    variables           = 'index',
    features_to_extract = features_to_extract,
    drop_original       = True,
)
calendar_features = calendar_transformer.fit_transform(data_indexed)[features_to_extract]
cyclical_encoder = CyclicalFeatures(
    variables     = features_to_extract,
    max_values    = max_values,
    drop_original = False
)

exogenous_features = cyclical_encoder.fit_transform(calendar_features)
exogenous_features.head(3)

# %%
exog_features = exogenous_features.filter(regex='_sin$|_cos$').columns.tolist()

# %%
data_exog = data_indexed[['purchase_bid']].merge(
           exogenous_features[exog_features],
           left_index  = True,
           right_index = True,
           how         = 'inner'  # Use only dates for which we have all the variables
       )
data_exog = data_exog.astype('float32')
data_exog.head(5)

# %%
metric, predictions = backtesting_forecaster(
                          forecaster = forecaster,
                          y          = data_exog['purchase_bid'],
                          exog       = data_exog[exog_features],
                          cv         = cv,
                          metric     = 'mean_absolute_error'
                      )
display(metric)
predictions.head()

# %%
fig = go.Figure()
trace1 = go.Scatter(x=data_indexed.index, y=data_indexed['purchase_bid'], name="test", mode="lines")
trace2 = go.Scatter(x=predictions.index, y=predictions['pred'], name="prediction", mode="lines")
fig.add_trace(trace1)
fig.add_trace(trace2)
fig.update_layout(
    title="Real value vs predicted in test data",
    xaxis_title="Date time",
    yaxis_title="Demand",
    width=800,
    height=400,
    margin=dict(l=20, r=20, t=35, b=20),
    legend=dict(orientation="h", yanchor="top", y=1.01, xanchor="left", x=0)
)
fig.show()

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Tuning Forcaster Arguments via Bayesian Search

# %%
forecaster_bs_tuned = ForecasterRecursive(
                 estimator       = LGBMRegressor(random_state=15926, verbose=-1),
                 lags            = 96,  # This value will be replaced in the grid search
                 window_features = window_features
             )

lags_grid = [300, (1, 2, 3, 95, 96, 97, 191, 192, 193, 287, 288, 289)]

def search_space(trial):
    search_space  = {
        'n_estimators' : trial.suggest_int('n_estimators', 300, 1000, step=100),
        'max_depth'    : trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
        'reg_alpha'    : trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda'   : trial.suggest_float('reg_lambda', 0, 1),
        'lags'         : trial.suggest_categorical('lags', lags_grid)
    } 
    return search_space

cv_search = TimeSeriesFold(
                steps              = 96*7,
                initial_train_size = len(data_indexed)-96*10,
                refit              = False,
            )

# %%
results_search, frozen_trial = bayesian_search_forecaster(
                                   forecaster   = forecaster_bs_tuned,
                                   y            = data_exog.loc[:, 'purchase_bid'],
                                   exog         = data_exog.loc[:, exog_features],
                                   cv           = cv_search,
                                   metric       = 'mean_absolute_error',
                                   search_space = search_space,
                                   n_trials     = 10,  # Increase for more exhaustive search
                                   return_best  = True
                               )

# %%
# Search results
# ==============================================================================
best_params = results_search.at[0, 'params']
best_params = best_params | {'random_state': 15926, 'verbose': -1}
best_lags = results_search.at[0, 'lags']
results_search.head(3)

# %%
cv_tuned = TimeSeriesFold(
        steps              = 96*7,
        initial_train_size = len(data_indexed)-96*8,
        refit              = False
)
metric, predictions = backtesting_forecas₹ter(
                          forecaster = forecaster_bs_tuned,
                          y          = data_exog['purchase_bid'],
                          exog       = data_exog[exog_features],
                          cv         = cv_tuned,
                          metric     = 'mean_absolute_error'
                      )
display(metric)
predictions.head()

# %%
fig = go.Figure()
trace1 = go.Scatter(x=data_indexed.index, y=data_indexed['purchase_bid'], name="test", mode="lines")
trace2 = go.Scatter(x=predictions.index, y=predictions['pred'], name="prediction", mode="lines")
fig.add_trace(trace1)
fig.add_trace(trace2)
fig.update_layout(
    title="Real value vs predicted in test data",
    xaxis_title="Date time",
    yaxis_title="Demand",
    width=800,
    height=400,
    margin=dict(l=20, r=20, t=35, b=20),
    legend=dict(orientation="h", yanchor="top", y=1.01, xanchor="left", x=0)
)
fig.show()

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Feature Selection

# %%
# Create forecaster
# ==============================================================================
estimator = LGBMRegressor(
                n_estimators = 200,
                max_depth    = 3,
                random_state = 15926,
                verbose      = -1
            )

forecaster = ForecasterRecursive(
                 estimator       = estimator,
                 lags            = best_lags,
                 window_features = window_features
             )

# Recursive feature elimination with cross-validation
# ==============================================================================
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
selector = RFECV(
    estimator = estimator,
    step      = 1,
    cv        = 3,
)
lags_select, window_features_select, exog_select = select_features(
    forecaster      = forecaster,
    selector        = selector,
    y               = data_exog['purchase_bid'],
    exog            = data_exog[exog_features],
    select_only     = None,
    force_inclusion = None,
    subsample       = 0.5,  # Subsample to speed up the process
    random_state    = 123,
    verbose         = True,
)

# %%
forecaster = ForecasterRecursive(
                 estimator       = LGBMRegressor(**best_params),
                 lags            = lags_select,
                 window_features = window_features_select
             )
forecaster.fit(
    y               = data_exog['purchase_bid'],
    exog            = data_exog[exog_features],
)

# %% [markdown]
# ### Predict 7 days

# %%
import pandas as pd
import numpy as np

# 1. Create the future index (672 steps = 7 days * 96 periods)
last_date = data_exog.index[-1]
future_index = pd.date_range(
    start = last_date + pd.Timedelta(minutes=15), 
    periods = 96 * 7, 
    freq = '15min'
)

# 2. Generate the raw time components
# Matching the logic from your cell [222]
future_day_of_week = future_index.dayofweek + 1  # 1 (Mon) to 7 (Sun)
future_hour = future_index.hour

# 3. Apply Cyclical Encoding (Matching your cell [223] max_values)
# day_of_week (max=7), hour (max=24)
exog_future = pd.DataFrame(index=future_index)
exog_future['day_of_week_sin'] = np.sin(2 * np.pi * future_day_of_week / 7)
exog_future['day_of_week_cos'] = np.cos(2 * np.pi * future_day_of_week / 7)
exog_future['hour_sin'] = np.sin(2 * np.pi * future_hour / 24)
exog_future['hour_cos'] = np.cos(2 * np.pi * future_hour / 24)

# 4. Display the result
print(f"Created exog features from {exog_future.index[0]} to {exog_future.index[-1]}")
display(exog_future.head())

# %%
forecaster_bs_tuned.fit(
    y = data_exog['purchase_bid'],
    exog = data_exog[exog_features] # Uncomment if using exogenous variables
)
predictions_future = forecaster_bs_tuned.predict(steps=672, exog=exog_future)

# %%
import plotly.graph_objects as go

# 1. Prepare the Past Data (Last 7 days of training)
# 96 periods * 7 days = 672
past_7_days = data_exog['purchase_bid'].tail(96 * 7)

# 2. Prepare the Future Data
# Ensure 'predictions_future' is the output from your .predict() call
# and it has the DatetimeIndex we created in the previous step.

fig = go.Figure()

# Plot Actual Past Data
fig.add_trace(go.Scatter(
    x=past_7_days.index, 
    y=past_7_days, 
    name="Actual History (Last 7 Days)", 
    line=dict(color='#333333', width=2)
))

# Plot Future Forecast
fig.add_trace(go.Scatter(
    x=predictions_future.index, 
    y=predictions_future, 
    name="Future Forecast (Next 7 Days)", 
    line=dict(color='#FF4B4B', width=2, dash='dash')
))

# Add a vertical line to mark the "Present" moment
fig.add_vline(
    x=past_7_days.index[-1], 
    line_width=2, 
    line_dash="dot", 
    line_color="green"
)

# Formatting
fig.update_layout(
    title="<b>7-Day Purchase Bid Forecast</b><br><sup>Historical Context vs. Recursive Future Prediction</sup>",
    xaxis_title="Date Time",
    yaxis_title="Purchase Bid Volume",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.show()

# %%
df_predictions = predictions_future.to_frame(name='predictions')
df_predictions = df_predictions.reset_index().rename(columns={'index': 'period_start'})
start_str = df_predictions['period_start'].iloc[0].strftime('%m%d')
end_str = df_predictions['period_start'].iloc[-1].strftime('%m%d')
filename = f"boost_predictions_{start_str}_{end_str}.csv"

df_predictions.to_csv(filename, index=False)
print(f"File saved successfully as: {filename}")
print(f"Total rows saved: {len(df_predictions)}")
display(df_predictions.head())

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Forecaster Direct Model

# %%
import pandas as pd
import numpy as np

# 1. Create the future index (672 steps = 7 days * 96 periods)
last_date = data_exog.index[-1]
future_index = pd.date_range(
    start = last_date + pd.Timedelta(minutes=15), 
    periods = 96 * 7, 
    freq = '15min'
)

# 2. Generate the raw time components
# Matching the logic from your cell [222]
future_day_of_week = future_index.dayofweek + 1  # 1 (Mon) to 7 (Sun)
future_hour = future_index.hour

# 3. Apply Cyclical Encoding (Matching your cell [223] max_values)
# day_of_week (max=7), hour (max=24)
exog_future = pd.DataFrame(index=future_index)
exog_future['day_of_week_sin'] = np.sin(2 * np.pi * future_day_of_week / 7)
exog_future['day_of_week_cos'] = np.cos(2 * np.pi * future_day_of_week / 7)
exog_future['hour_sin'] = np.sin(2 * np.pi * future_hour / 24)
exog_future['hour_cos'] = np.cos(2 * np.pi * future_hour / 24)

# 4. Display the result
print(f"Created exog features from {exog_future.index[0]} to {exog_future.index[-1]}")
display(exog_future.head())

# %%
# Forecaster with direct method
# ==============================================================================
forecaster = ForecasterDirect(
                 estimator       = LGBMRegressor(**best_params),
                 steps           = 96*7,
                 lags            = lags_select,
                 window_features = window_features
             )

# Backtesting model
# ==============================================================================
metric, predictions = backtesting_forecaster(
                          forecaster = forecaster,
                          y               = data_exog['purchase_bid'],
                          exog            = data_exog[exog_features],
                          cv              = cv_lgbm,
                          metric          = 'mean_absolute_error'
                      )

display(metric)
predictions.head()


# %%
fig = go.Figure()
trace1 = go.Scatter(x=data_indexed.index, y=data_indexed['purchase_bid'], name="test", mode="lines")
trace2 = go.Scatter(x=predictions.index, y=predictions['pred'], name="prediction", mode="lines")
fig.add_trace(trace1)
fig.add_trace(trace2)
fig.update_layout(
    title="Real value vs predicted in test data",
    xaxis_title="Date time",
    yaxis_title="Demand",
    width=800,
    height=400,
    margin=dict(l=20, r=20, t=35, b=20),
    legend=dict(orientation="h", yanchor="top", y=1.01, xanchor="left", x=0)
)
fig.show()

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Feature explanation with SHAP

# %%
# Training matrices used by the forecaster to fit the internal estimator
# ==============================================================================
X_train, y_train = forecaster.create_train_X_y(
                       y               = data_exog['purchase_bid'],
                       exog            = data_exog[exog_features],
                   )
display(X_train.head(3))
display(y_train.head(3))

# %%
import shap
# Create SHAP explainer (for three base models)
# ==============================================================================
shap.initjs()
explainer = shap.TreeExplainer(forecaster.estimator)

# Sample 50% of the data to speed up the calculation
rng = np.random.default_rng(seed=785412)
sample = rng.choice(X_train.index, size=int(len(X_train)*0.5), replace=False)
X_train_sample = X_train.loc[sample, :]
shap_values = explainer.shap_values(X_train_sample)


# %%
shap.summary_plot(shap_values, X_train_sample, max_display=10, show=False)
fig, ax = plt.gcf(), plt.gca()
ax.set_title("SHAP Summary plot")
ax.tick_params(labelsize=8)
fig.set_size_inches(6, 4.5)

# %%
