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

# %%

# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# 1. Define your specific features
features = [
    'period_enum', 'weekday_enum', 
    'pb_lag1d', 'pb_lag2d', #'pb_lag1d_plus1', 'pb_lag1d_plus2', 'pb_lag1d_minus1', 'pb_lag1d_minus2',
    # 'sb_lag1d', 'sb_lag2d' #'sb_lag1d_plus1', 'sb_lag1d_plus2', 'sb_lag1d_minus1', 'sb_lag1d_minus2'
]
target = 'purchase_bid'

# 2. Scaling (CRITICAL for LSTMs)
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_x.fit_transform(data_clean[features])
y_scaled = scaler_y.fit_transform(data_clean[[target]])

# 3. Time-Series Split
test_size = 96 * 4
X_train_raw, X_test_raw = X_scaled[:-test_size], X_scaled[-test_size:]
y_train, y_test = y_scaled[:-test_size], y_scaled[-test_size:]

# 4. Reshape to 3D: [Samples, Time Steps, Features]
# Here we treat each row as 1 time step with 12 features
X_train = X_train_raw.reshape((X_train_raw.shape[0], 1, X_train_raw.shape[1]))
X_test = X_test_raw.reshape((X_test_raw.shape[0], 1, X_test_raw.shape[1]))

# 5. Build the LSTM Model
model = Sequential([
    # Input shape is (Time Steps, Features)
    LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dense(1) # Predicting the single purchase_bid value
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 6. Train
history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=32, 
    validation_data=(X_test, y_test),
    verbose=1
)

# 7. Predict and Inverse Scale
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

# 8. Evaluate
mae = mean_absolute_error(y_test_actual, y_pred)
print(f"LSTM MAE: {mae:.2f}")

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))

# Plotting the actual values
plt.plot(y_test_actual, label='Actual Purchase Bid', color='royalblue', linewidth=2, alpha=0.7)

# Plotting the LSTM predictions
plt.plot(y_pred, label='LSTM Forecast', color='darkorange', linewidth=2, linestyle='--')

plt.title(f'LSTM Forecast vs Actuals (Last {test_size//96} Days)', fontsize=14)
plt.xlabel('Time Steps (15-min intervals)', fontsize=12)
plt.ylabel('Purchase Bid Value', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Optional: Zoom in on the last 2 days for better clarity
# plt.xlim(192, 384) 

plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Calculate baseline MAE (already calculated as 'mae' in your LSTM cell)
baseline_mae = mae
feat_importance = []

print("Calculating permutation importance (this may take a moment)...")

for i, feat in enumerate(features):
    # Create a copy of the test set
    X_test_permuted = X_test.copy()
    
    # Shuffle only the values of the current feature across the test set
    # Since shape is (samples, 1, features), we shuffle index [:, 0, i]
    np.random.shuffle(X_test_permuted[:, 0, i])
    
    # Predict with the 'broken' feature
    y_pred_permuted_scaled = model.predict(X_test_permuted, verbose=0)
    y_pred_permuted = scaler_y.inverse_transform(y_pred_permuted_scaled)
    
    # Calculate new MAE
    permuted_mae = mean_absolute_error(y_test_actual, y_pred_permuted)
    
    # Importance is the increase in error
    importance = permuted_mae - baseline_mae
    feat_importance.append({'Feature': feat, 'Importance': importance})

# 2. Convert to DataFrame and sort
importance_df = pd.DataFrame(feat_importance).sort_values(by='Importance', ascending=False)

# 3. Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='magma')
plt.title('LSTM Feature Importance (Permutation Method)')
plt.xlabel('Increase in MAE when feature is shuffled')
plt.show()

print(importance_df)

# %%
test_days = 7

# Create a date column (without time) to group by day
data_clean['date'] = data_clean['period_start'].dt.date

# Pivot: Rows = Dates, Columns = The 96 periods
df_daily = data_clean.pivot(index='date', columns='period_enum', values='purchase_bid')
df_daily = df_daily.dropna() # Ensure we only have full 96-period days

# Scale the entire 96-column matrix
scaler = MinMaxScaler()
df_daily_scaled = scaler.fit_transform(df_daily)

train_daily_scaled = df_daily_scaled[:-test_days]
test_daily_actuals = df_daily_scaled[-test_days:]

# Create X (Day N) and y (Day N+1)
X_train_days = train_daily_scaled[:-1] 
y_train_days = train_daily_scaled[1:]

# Reshape for LSTM: [Samples, Time Steps, Features]
# Features = 96 (the whole day)
X_train_days = X_train_days.reshape((X_train_days.shape[0], 1, 96))
y_train_days = y_train_days.reshape((y_train_days.shape[0], 96))

# %%
print(f"Training on {X_train_days.shape[0]} days.")
print(f"Holding out {test_days} days for testing.")

# %%

# %%
X_days.shape, y_days.shape

# %%

# %%
from tensorflow.keras.layers import RepeatVector, TimeDistributed

model_vector = Sequential([
    # Input is the 96 values of yesterday
    LSTM(128, activation='relu', input_shape=(1, 96), return_sequences=True),
    LSTM(64, activation='relu'),
    # Output is the 96 values of tomorrow
    Dense(96) 
])

model_vector.compile(optimizer='adam', loss='mse')
model_vector.fit(X_train_days, y_train_days, epochs=100, batch_size=16, verbose=1)

# %%
# Start recursion using the day IMMEDIATELY PRECEDING your test set
current_input = train_daily_scaled[-1:].reshape((1, 1, 96))

test_predictions = []

for day in range(test_days):
    # Predict next day
    next_day_pred = model_vector.predict(current_input, verbose=0)
    test_predictions.append(next_day_pred.flatten())
    
    # Feed prediction back in for the next day's forecast
    current_input = next_day_pred.reshape((1, 1, 96))

# Convert list to array for inverse scaling
test_pred_matrix = np.array(test_predictions)
final_7day_test_forecast = scaler.inverse_transform(test_pred_matrix).flatten()

# Get actual values for the same 7 days for comparison
actual_7day_values = scaler.inverse_transform(test_daily_actuals).flatten()

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Get the timestamps for the test period (last 7 days)
test_size = 96 * 7
test_timestamps = data_clean['period_start'].iloc[-test_size:]

# 2. Calculate Evaluation Metrics
mae_7d = mean_absolute_error(actual_7day_values, final_7day_test_forecast)
rmse_7d = np.sqrt(mean_squared_error(actual_7day_values, final_7day_test_forecast))

# 3. Create the Visualization
plt.figure(figsize=(16, 7))

# Plot the Actual Ground Truth
plt.plot(test_timestamps, actual_7day_values, label='Actual Ground Truth', color='black', alpha=0.6, linewidth=1.5)

# Plot the Model's 7-Day Recursive Forecast
plt.plot(test_timestamps, final_7day_test_forecast, label='LSTM Vector Forecast (Recursive)', 
         color='crimson', linestyle='--', linewidth=2)

# Styling and Labels
plt.title(f'7-Day Out-of-Sample Evaluation\nMAE: {mae_7d:.2f} | RMSE: {rmse_7d:.2f}', fontsize=14)
plt.xlabel('Timestamp (15-min intervals)', fontsize=12)
plt.ylabel('Purchase Bid Volume', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 4. Print individual day MAEs to see error growth
for d in range(7):
    day_mae = mean_absolute_error(actual_7day_values[d*96:(d+1)*96], 
                                 final_7day_test_forecast[d*96:(d+1)*96])
    print(f"Day {d+1} MAE: {day_mae:.2f}")

# %%

# %%

# %%

# %%

# %%
# Start with the very last day in your training set (March 31st)
last_day_vector = df_daily_scaled[-1:].reshape((1, 1, 96))

all_forecasts = []
current_input = last_day_vector

for day in range(7):
    # Predict the entire next day (96 blocks) in one go
    next_day_scaled = model_vector.predict(current_input, verbose=0)
    
    # Store result
    all_forecasts.append(next_day_scaled.flatten())
    
    # Update input: Today's prediction becomes tomorrow's input
    current_input = next_day_scaled.reshape((1, 1, 96))

# Flatten and Inverse Scale
forecast_flat = np.array(all_forecasts).flatten().reshape(-1, 1)
# Create a dummy matrix to inverse scale (since scaler expects 96 columns)
dummy_matrix = np.zeros((len(all_forecasts), 96))
for i in range(len(all_forecasts)):
    dummy_matrix[i] = all_forecasts[i]

final_7day_forecast = scaler.inverse_transform(dummy_matrix).flatten()

# %%
final_7day_forecast

# %%
import matplotlib.pyplot as plt
import pandas as pd

# 1. Define the timeframe for history (last 7 days)
history_size = 96 * 7 

# 2. Extract historical data for the plot
history_y = data_clean['purchase_bid'].tail(history_size)
history_x = data_clean['period_start'].tail(history_size)

# 3. Create the forecast timeline starting from the last date in data_clean
last_timestamp = data_clean['period_start'].iloc[-1]
forecast_x = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=15), 
                           periods=len(final_7day_forecast), freq='15min')

# 4. Plotting
plt.figure(figsize=(16, 7))

# Plot historical actuals
plt.plot(history_x, history_y, label='Historical Actuals (Last 7 Days)', color='black', alpha=0.6)

# Plot the 7-day Vector LSTM Forecast
plt.plot(forecast_x, final_7day_forecast, label='LSTM Vector Forecast (7-Day Ahead)', 
         color='darkorange', linewidth=2, linestyle='-')

# Add a vertical line to mark the start of the forecast
plt.axvline(x=last_timestamp, color='red', linestyle='--', label='Forecast Start (April 1st)')

plt.title('7-Day Energy Bid Forecast: Historical Context vs. LSTM Vector Prediction', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Purchase Bid Volume', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# 1. Create the 7-day lag (essential for weekly patterns)
data_clean['pb_lag7d'] = data['purchase_bid'].shift(96 * 7)
data_clean = data_clean.dropna()

# 2. Pivot all features into daily vectors
# We create a dictionary of "daily matrices"
feature_cols = ['purchase_bid', 'pb_lag1d', 'pb_lag2d', 'pb_lag7d', 'period_enum', 'weekday_enum']
daily_matrices = {}

for col in feature_cols:
    daily_matrices[col] = data_clean.pivot(index='date', columns='period_enum', values=col).dropna()

# Ensure all matrices have the same dates (intersection)
common_dates = daily_matrices['purchase_bid'].index
for col in feature_cols:
    daily_matrices[col] = daily_matrices[col].loc[common_dates]

# 3. Scale each feature channel individually
scalers = {col: MinMaxScaler() for col in feature_cols}
scaled_matrices = {col: scalers[col].fit_transform(daily_matrices[col]) for col in feature_cols}


# %%
# Features for the model: Lag1, Lag2, Lag7, Period_Enum, Weekday_Enum
# (Note: 'purchase_bid' is our target for the NEXT day)

def create_3d_dataset(matrices, test_days=7):
    X, y = [], []
    # dates = list(matrices['purchase_bid'].index)

    num_days = matrices['purchase_bid'].shape[0]
    # We use matrices['pb_lag1d'], etc., as inputs for day D to predict matrices['purchase_bid'] for day D
    # Actually, to predict Tomorrow (D+1), we use Today's (D) features.
    
    for i in range(num_days - 1):
        # Create a 96x5 slice for the day
        day_features = np.stack([
            scaled_matrices['pb_lag1d'][i],
            scaled_matrices['pb_lag2d'][i],
            scaled_matrices['pb_lag7d'][i],
            scaled_matrices['period_enum'][i],
            scaled_matrices['weekday_enum'][i]
        ], axis=-1) # Shape: (96, 5)
        
        X.append(day_features)
        y.append(scaled_matrices['purchase_bid'][i+1]) # Target: Tomorrow's actual bids
        
    return np.array(X), np.array(y)

X_all, y_all = create_3d_dataset(scaled_matrices)

# Split for testing
X_train, X_test = X_all[:-7], X_all[-7:]
y_train, y_test = y_all[:-7], y_all[-7:]

# %%
from tensorflow.keras.layers import Flatten

model_multi = Sequential([
    # Input shape: (96 time steps, 5 features per step)
    LSTM(128, activation='relu', input_shape=(96, 5), return_sequences=True),
    LSTM(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(96) # Output: All 96 periods for the next day
])

model_multi.compile(optimizer='adam', loss='mse')
model_multi.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

# %%
# Start with the last available training day features
# For a true recursive forecast, you'd need a rolling buffer of the last 7 days of predictions
current_X = X_test[0:1] # Start with the first day of the test period
all_forecasts = []

for d in range(7):
    # 1. Predict tomorrow
    next_day_pred = model_multi.predict(current_X, verbose=0)
    all_forecasts.append(next_day_pred.flatten())
    
    if d < 6:
        # 2. Update the input for the next step
        # This is complex because Lag2 tomorrow = Lag1 today, etc.
        # For simplicity in this vector approach, we update the Lag1 channel 
        # with our prediction and shift the others if you have a buffer.
        
        new_day = current_X.copy()
        # Update Lag1 channel (index 0) with the new prediction
        new_day[0, :, 0] = next_day_pred.flatten() 
        # Update Weekday channel (index 4)
        new_day[0, :, 4] = (new_day[0, :, 4] + (1/7)) % 1.0 
        
        current_X = new_day

# Inverse scale the results
final_forecast = scalers['purchase_bid'].inverse_transform(np.array(all_forecasts)).flatten()

# %%
final_forecast

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Configuration
days_of_history_to_show = 17  # How many days of training data to show before the forecast
test_days = 7
steps_per_day = 96

# 2. Extract Timestamps
# The test period is the last (test_days * 96) rows
test_timestamps = data_clean['period_start'].iloc[-(test_days * steps_per_day):]

# The history period is the segment just before the test period
history_end_idx = len(data_clean) - (test_days * steps_per_day)
history_start_idx = history_end_idx - (days_of_history_to_show * steps_per_day)
history_timestamps = data_clean['period_start'].iloc[history_start_idx:history_end_idx]
history_values = data_clean['purchase_bid'].iloc[history_start_idx:history_end_idx]

# 3. Plotting
plt.figure(figsize=(18, 8))

# Plot Training History (Context)
plt.plot(history_timestamps, history_values, 
         label='Training Data (History)', color='gray', alpha=0.5, linewidth=1)

# Plot Actual Test Values
plt.plot(test_timestamps, actual_7day_values, 
         label='Actual Bids (Test Set)', color='black', linewidth=1.5)

# Plot Multi-Lag Forecast
# Ensure final_forecast is flattened if it isn't already
forecast_plot_values = final_forecast.flatten() if hasattr(final_forecast, 'flatten') else final_forecast
plt.plot(test_timestamps, forecast_plot_values, 
         label='Multi-Lag LSTM Forecast', color='blue', linestyle='--', linewidth=2)

# Add a vertical line to separate Training from Testing
plt.axvline(x=history_timestamps.iloc[-1], color='red', linestyle='-', alpha=0.8, label='Forecast Start')

# Formatting
plt.title(f'7-Day Multi-Lag Vector Forecast\nLags used: [1d, 2d, 7d] + Period Enum', fontsize=16)
plt.xlabel('Date & Time', fontsize=12)
plt.ylabel('Purchase Bid Volume', fontsize=12)
plt.legend(loc='upper left', frameon=True, shadow=True)
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 4. Final Accuracy Check
from sklearn.metrics import mean_absolute_error
overall_mae = mean_absolute_error(actual_7day_values, forecast_plot_values)
print(f"Overall 7-Day Forecast MAE: {overall_mae:.2f}")

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

# 1. Define feature names (order must match your X_train/X_test stacking)
feature_names = ['pb_lag1d', 'pb_lag2d', 'pb_lag7d', 'period_enum', 'weekday_enum']

# 2. Calculate Baseline MAE (the error of your current model)
# Using your existing test set and predictions
y_pred_baseline = model_multi.predict(X_test, verbose=0)
baseline_mae = mean_absolute_error(y_test, y_pred_baseline)

importances = []

print("Analyzing feature contributions...")

for i, name in enumerate(feature_names):
    # Create a copy of the test set to mess with
    X_test_permuted = X_test.copy()
    
    # Shuffle the values for this specific feature channel across all days and periods
    # X_test shape is (7, 96, 5) -> we shuffle index [:, :, i]
    # We flatten, shuffle, and reshape to break the temporal relationship
    values = X_test_permuted[:, :, i].flatten()
    np.random.shuffle(values)
    X_test_permuted[:, :, i] = values.reshape(X_test_permuted.shape[0], X_test_permuted.shape[1])
    
    # Predict with the "broken" feature
    y_pred_permuted = model_multi.predict(X_test_permuted, verbose=0)
    permuted_mae = mean_absolute_error(y_test, y_pred_permuted)
    
    # Importance = how much worse the model got
    importance = permuted_mae - baseline_mae
    importances.append(importance)

# 3. Create DataFrame and Plot
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Multi-Lag LSTM Feature Importance\n(Increase in MAE when feature is shuffled)', fontsize=14)
plt.xlabel('Importance (MAE Delta)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

print(importance_df)

# %%
