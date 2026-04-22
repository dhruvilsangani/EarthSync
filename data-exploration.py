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

# %%
data = pd.read_csv("../../Downloads/iex-dam-0201-0421.csv")

# %%
data["period_start"] = pd.to_datetime(data["period_start"])
data

# %%
data.info()

# %%
data.describe()

# %% [markdown]
# ### Data Visualization - naive plots

# %%

columns = ["purchase_bid", "sell_bid", "mcv", "mcp"]
titles  = ["Purchase Bid", "Sell Bid", "MCV", "MCP"]
colors  = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD"]

# purchase_bid, sell_bid, mcv → left y-axis (MW)
# mcp → right y-axis (₹/MWh)  — very different scale, needs its own axis

fig = go.Figure()

for col, name, color in zip(columns, titles, colors):
    fig.add_trace(go.Scatter(
        x=data["period_start"],
        y=data[col],
        mode="lines+markers",
        name=name,
        line=dict(color=color, width=2),
        marker=dict(size=5, color=color),
        yaxis="y2" if col == "mcp" else "y1",
        hovertemplate=f"<b>%{{x|%H:%M}}</b><br>{name}: %{{y:,.2f}}<extra></extra>",
    ))

fig.update_layout(
    title=dict(text="IEX Market Data", font=dict(size=18), x=0.5, xanchor="center"),
    height=520,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(t=70, b=60, l=70, r=70),
    font=dict(family="Arial, sans-serif", size=12),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="center", x=0.5,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(150,150,150,0.3)",
        borderwidth=1,
    ),
    yaxis=dict(
        title="MW",
        showgrid=True,
        gridcolor="rgba(150,150,150,0.2)",
        zeroline=False,
    ),
    yaxis2=dict(
        title="₹/MWh",
        overlaying="y",
        side="right",
        showgrid=False,
        zeroline=False,
        tickfont=dict(color="#7F77DD"),
        title_font=dict(color="#7F77DD"),
    ),
)

fig.show("notebook")

# %% [markdown]
# from the above plots, I am observing
# purchase bids 
# spike at around 7:30 am in the morning, and 6:30 pm in the evening
# A slight downtrend moving from february to march.
# sell bids
# spike at around 01:00 pm, then lower at 06:30 pm, and rise again at around 2:30 am, and lower at 07:00 pm
# MCV
# follows almost the same trend as sell bids, but with some significant noise.
# MCP
# whenever sell bid is low, MCP is high.
# capped at 10000
# interesting things
# 1. There is a massive simultaneous supply shortage from march 11 to 15 and demand spike during the same period.
#
# findings
# 1. MCV and Sell_bid are strongly correlated
# 2. MCP and ratio(purchase/sell) are linearly related (when 10,000 cap is not breached)
#
# what's happening with the difference of energy between sell_bid and MCV
#
# ideally, the energy purchase bid today at a particular period should depend on a similar period yesterday or a few days before.
# 04:00 should depend on 03:00 to 05:00 

# %% [markdown]
# ## Temporal Context

# %%

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data_temporal = data.copy()

# --- 1. Hourly Profiling (The Duck Curve) ---
def plot_hourly_distribution(df):
    plt.figure(figsize=(15, 6))
    # Boxplot shows the median, quartiles, and outliers for each 15-min block
    sns.boxplot(data=df, x='period_enum', y='purchase_bid', palette="viridis")
    plt.title("Hourly Distribution of Purchase Bids (The Duck Curve)", fontsize=14)
    plt.xlabel("Period Block (1-96)", fontsize=12)
    plt.ylabel("Purchase Bid (MW)", fontsize=12)
    plt.xticks(range(0, 97, 4), labels=range(0, 97, 4)) # Show hours on X-axis
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

# --- 2. Weekly Seasonality (Weekday vs Weekend) ---
def plot_weekly_impact(df):
    # Categorize into Weekday (1-5) and Weekend (6-7)
    df['day_type'] = df['weekday_enum'].apply(lambda x: 'Weekend' if x >= 6 else 'Weekday')
    
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df, x='day_type', y='purchase_bid', split=True, inner="quart", palette="Pastel1")
    plt.title("Purchase Bid Volume: Weekdays vs. Weekends", fontsize=14)
    plt.show()

# --- 3. Holiday Identification ---
def add_holiday_flags(df):
    # List of major Indian holidays for Feb - April 2026
    # Note: These dates affect industrial load significantly
    holidays_2026 = [
        '2026-02-18', # Maha Shivaratri
        '2026-03-03', # Holi
        '2026-03-19', # Gudi Padwa / Ugadi
        '2026-03-28', # Ram Navami
        '2026-04-03', # Good Friday
        '2026-04-10', # Ambedkar Jayanti
    ]
    
    df['is_holiday'] = df['period_start'].dt.strftime('%Y-%m-%d').isin(holidays_2026).astype(int)
    
    # Check impact
    holiday_stats = df.groupby('is_holiday')['purchase_bid'].mean()
    print("Average Bid on Holidays vs Working Days:")
    print(holiday_stats)
    return df

# Execution
data_temporal["period_enum"] = data_temporal["period_start"].dt.hour * 4 + data_temporal["period_start"].dt.minute // 15 + 1
data_temporal["weekday_enum"] = data_temporal["period_start"].dt.weekday + 1

plot_hourly_distribution(data_temporal)
plot_weekly_impact(data_temporal)
data_temporal = add_holiday_flags(data_temporal)

# %% [markdown]
# From the above plot, I am seeing outliers present during evening hours from period 75 to 96 to 4, as a result i need to have is_evening parameter here.

# %% [markdown]
# ## Environmental features

# %%
import requests
import pandas as pd

df_temp = data.copy()

def get_nci_data(start_date, end_date):
    # Defining cities and their weights for the NCI
    # Mumbai (High Commercial), Delhi (High Residential/Peak), Akola (Heat Epicenter)
    locations = {
        "delhi": {"lat": 28.61, "lon": 77.20, "weight": 0.5},
        "mumbai": {"lat": 19.07, "lon": 72.88, "weight": 0.3},
        "akola": {"lat": 20.70, "lon": 77.01, "weight": 0.2}
    }
    
    all_weather = pd.DataFrame()

    for name, loc in locations.items():
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max",
            "timezone": "Asia/Kolkata"
        }
        res = requests.get(url, params=params).json()
        
        temp_df = pd.DataFrame({
            "date": pd.to_datetime(res["daily"]["time"]),
            f"temp_{name}": res["daily"]["temperature_2m_max"]
        })
        
        if all_weather.empty:
            all_weather = temp_df
        else:
            all_weather = all_weather.merge(temp_df, on="date")

    # Calculate the National Cooling Index (Weighted Average)
    all_weather['NCI'] = 0
    for name, loc in locations.items():
        all_weather['NCI'] += all_weather[f'temp_{name}'] * loc['weight']
    
    return all_weather

# 1. Fetch the multi-city data
weather_master = get_nci_data("2026-02-01", "2026-04-21")

# 2. Merge into your main dataframe
df_temp['date_only'] = df_temp['period_start'].dt.date
weather_master['date_only'] = weather_master['date'].dt.date

df_temp = df_temp.merge(weather_master[['date_only', 'NCI', 'temp_delhi', 'temp_akola', 'temp_mumbai']], on='date_only', how='left')

# 3. Create a non-linear 'Cooling Demand' feature
# This captures the 'spike' effect when NCI crosses 35 degrees
df_temp['cooling_load_factor'] = (df_temp['NCI'] - 30).clip(lower=0)**2

print("NCI and Cooling Load Factor added to data_clean.")

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# 2. Add Purchase Bid (Primary Axis)
fig.add_trace(
    go.Scatter(
        x=df_temp['period_start'], 
        y=df_temp['purchase_bid'], 
        name="Purchase Bid (MW)",
        line=dict(color="#378ADD", width=1.5),
        opacity=0.8
    ),
    secondary_y=False,
)

# 3. Add Max Temperature (Secondary Axis)
# Note: This will appear as a 'staircase' because it is daily data
fig.add_trace(
    go.Scatter(
        x=df_temp['period_start'], 
        y=df_temp['cooling_load_factor'], 
        name="Max Temp (°C)",
        line=dict(color="#FF4B4B", width=3, dash='dot'),
    ),
    secondary_y=True,
)

# 4. Add layout details
fig.update_layout(
    title_text="<b>Impact of Heatwaves on IEX Purchase Bids (Feb - Apr 2026)</b>",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# Set y-axes titles
fig.update_yaxes(title_text="<b>Purchase Bid</b> (MW)", secondary_y=False)
fig.update_yaxes(title_text="<b>cooling_load_factore</b> (°C)", secondary_y=True)

fig.show()

# %%
# Filter for heatwave periods to see the 'true' driver
heatwave_data = df_temp[df_temp['NCI'] > 34]

# Check correlation of individual cities against purchase_bid
city_cols = [col for col in df_temp.columns if 'temp_' in col]
correlations = heatwave_data[city_cols + ['purchase_bid']].corr()['purchase_bid'].sort_values(ascending=False)

print("Correlation with Purchase Bids during Heatwaves:")
print(correlations)

# %%
# Check if yesterday's temperature in a city predicts today's bids better
for city in city_cols:
    corr_today = df_temp['purchase_bid'].corr(df_temp[city])
    corr_lagged = df_temp['purchase_bid'].corr(df_temp[city].shift(96)) # 96 blocks = 1 day
    print(f"{city} -> Today: {corr_today:.2f}, Lagged (Yesterday): {corr_lagged:.2f}")

# %%
import plotly.express as px

# Prepare data for a heatmap
temp_only = df_temp[city_cols + ['date_only']].drop_duplicates().set_index('date_only')

fig = px.imshow(temp_only.T, 
                labels=dict(x="Date", y="City", color="Temp (°C)"),
                title="Regional Temperature Heatmap vs. IEX Spikes",
                color_continuous_scale="Reds")
fig.show()

# %% [markdown]
# The features are
# 1. period_enum
# 2. weekday_enum
# 3. is_evening
# 4. rolling 3 day cooling load factor
# 5. previous day's uncleared volume
# 6. previous day's purchase bid

# %%
data

# %%
import pandas as pd
import numpy as np

# 1. Initialize the fresh dataframe
# Assuming 'df_temp' is your source with 'period_start', 'purchase_bid', 'sell_bid', and 'NCI'
features_df = data.copy()

# --- Feature 1: period_enum ---
features_df["period_enum"] = features_df["period_start"].dt.hour * 4 + data_temporal["period_start"].dt.minute // 15 + 1

# --- Feature 2: weekday_enum ---
features_df["weekday_enum"] = features_df["period_start"].dt.weekday + 1

# --- Feature 3: is_evening ---
# Defining evening peak as 6:00 PM to 10:00 PM (Blocks 73-88)
features_df['is_evening'] = features_df['period_enum'].between(73, 88).astype(int)

# --- Feature 4: rolling 3 day cooling load factor ---
# First, calculate the daily cooling load factor
df_temp['daily_clf'] = (df_temp['NCI'] - 30).clip(lower=0)**2
# Since 'daily_clf' is the same for all 96 blocks of a day, 
# a 3-day rolling window is 96 * 3 blocks
features_df['rolling_3d_clf'] = df_temp['daily_clf'].rolling(window=96*3, min_periods=1).mean()

# --- Feature 5: previous day's uncleared volume ---
# Uncleared Volume = Total Purchase Bid - Market Cleared Volume (mcv)
features_df['uncleared_vol'] = (features_df['purchase_bid'] - features_df['mcv'])
features_df['prev_day_uncleared_vol'] = features_df['uncleared_vol'].shift(96)

# --- Feature 6: previous day's purchase bid ---
features_df['prev_day_purchase_bid'] = features_df['purchase_bid'].shift(96)

# Drop the first day (96 rows) since lags will be NaN
features_df.dropna(inplace=True)

print("Fresh Feature Dataframe Created!")
features_df.head()

# %%
features_df[features_df.is_evening == 1]

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Define your features and target
X = features_df[['period_enum', 'weekday_enum', 'is_evening', 
                 'rolling_3d_clf', 'prev_day_uncleared_vol', 'prev_day_purchase_bid']]
X = features_df[['period_enum', 'weekday_enum', 'is_evening', 
                 'rolling_3d_clf']]
y = features_df['purchase_bid']

# Initialize and fit a baseline Random Forest
# We use a shallow depth to avoid overfitting on noise
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X, y)

# %%
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='importance', y='feature', data=importances, palette='magma')
plt.title("Built-in Feature Importance (Random Forest)")
plt.show()

# %%
result = permutation_importance(model, X, y, n_repeats=10, random_state=42)

perm_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': result.importances_mean
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='importance', y='feature', data=perm_importances, palette='viridis')
plt.title("Permutation Importance (The 'True' Impact)")
plt.show()

# %%
# List of required columns
required_columns = [
    'period_start', 'period', 'purchase_bid', 'sell_bid', 
    'mcv', 'mcp', 'final_scheduled_volume',
    'rolling_3d_clf', 'period_enum', 'weekday_enum'
]

# Export to CSV
features_df[required_columns].to_csv('iex_dam_features.csv', index=False)

print("File 'iex_market_features_top3.csv' has been saved to your current directory.")

# %%

# %%

# %%

# %%

# %%

# %%

# %%
# 1. Aggregate to Daily level to remove intra-day noise (Duck Curve)
daily_stats = df_temp.groupby('date_only').agg({
    'purchase_bid': 'mean',
    'NCI': 'first'  # NCI is already a daily value
}).reset_index()

# 2. Calculate a 7-day Rolling Correlation
# This shows how the relationship evolves from Feb to April
window = 7
daily_stats['rolling_corr'] = (
    daily_stats['purchase_bid']
    .rolling(window=window)
    .corr(daily_stats['NCI'])
)

# 3. Plot the result
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.plot(daily_stats['date_only'], daily_stats['rolling_corr'], 
         color='#FF5733', marker='o', linewidth=2)

# Styling
plt.title(f"{window}-Day Rolling Correlation: NCI vs. Mean Purchase Bid", fontsize=15)
plt.ylabel("Correlation Coefficient (r)", fontsize=12)
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.ylim(-1, 1) # Correlation is always between -1 and 1
plt.grid(alpha=0.2)

# Highlight the Regime Shift zones
plt.axvspan(pd.to_datetime('2026-02-01'), pd.to_datetime('2026-03-01'), 
            color='blue', alpha=0.05, label='Winter Regime (Low/No Correlation)')
plt.axvspan(pd.to_datetime('2026-04-01'), pd.to_datetime('2026-04-21'), 
            color='red', alpha=0.05, label='Summer Regime (High Correlation)')

plt.legend()
plt.show()

# %%
