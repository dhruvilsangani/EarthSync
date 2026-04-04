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

# %%
data = pd.read_csv("../../Downloads/iex_dam_feb_mar_2026.csv")

# %%
# conforming type
data["period_start"] = pd.to_datetime(data["period_start"])
data["period_enum"] = data["period_start"].dt.hour * 4 + data["period_start"].dt.minute // 15 + 1
data["weekday_enum"] = data["period_start"].dt.weekday + 1

# %%
data["pb_lag1d"] = data["purchase_bid"].shift(96)
data["sb_lag1d"] = data["sell_bid"].shift(96)
data["pb_lag2d"] = data["purchase_bid"].shift(192)
data["sb_lag2d"] = data["sell_bid"].shift(192)

data['target_diff'] = data['purchase_bid']

data_clean = data.dropna()

print(f"New feature count: {len(data_clean.columns)}")
data_clean.head()

# %% [markdown]
# ## LightGBM

# %%
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 1. Define Features and Target
# Using the lags you created in cell [5] plus your enums
features = [
    'period_enum', 'weekday_enum', 
    'pb_lag1d', 'pb_lag2d'
    # 'pb_lag1d_plus1', 'pb_lag1d_plus2', 'pb_lag1d_minus1', 'pb_lag1d_minus2',
    # 'sb_lag1d',
    # 'sb_lag1d_plus1', 'sb_lag1d_plus2', 'sb_lag1d_minus1', 'sb_lag1d_minus2'
]
target = 'purchase_bid'

X = data_clean[features]
y = data_clean[target]

# 2. Time-Series Split (Don't use random_split for time data!)
# We'll take the last 4 days (96 * 4) as our test set
test_size = 96*4 
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

# 3. Create LightGBM Datasets
# Explicitly telling LightGBM which columns are categorical
cat_features = ['period_enum']#, 'weekday_enum']
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, categorical_feature=cat_features)

# 4. Set Parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.005,
    'num_leaves': 64,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbosity': -1
}

# 5. Train the Model
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'valid'],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# 6. Predict and Evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)

print(f"\n--- Model Performance ---")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# 7. Quick Visualization
plt.figure(figsize=(12, 5))
plt.plot(y_test.values, label='Actual Bid', color='black', alpha=0.6)
plt.plot(y_pred, label='LGBM Forecast', color='red', linestyle='--')
plt.title('Purchase Bid Forecast: Last 4 Days')
plt.legend()
plt.show()

# %%

# %%

# %%

# %%

# %%
import pandas as pd
import seaborn as sns

# 1. Get feature importance (using 'gain' to see contribution to error reduction)
importance = model.feature_importance(importance_type='gain')
feature_names = model.feature_name()

# 2. Create a DataFrame for easy plotting
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# 3. Plot the results
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('LightGBM Feature Importance (Gain)')
plt.xlabel('Total Gain Contributed to Prediction')
plt.show()

# Optional: Print the raw values
print(feature_importance_df)
