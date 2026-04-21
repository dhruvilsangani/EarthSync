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

df_temp = df_temp.merge(weather_master[['date_only', 'NCI']], on='date_only', how='left')

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

# %% [markdown]
# The features are
# 1. period_enum
# 2. weekday_enum
# 3. is_evening
# 4. rolling 3 day cooling load factor
# 5. previous day's uncleared volume

# %%

# %%

# %%

# %%

# %%

# %%
columns = ["purchase_bid", "sell_bid", "mcv", "mcp"]
labels  = ["Purchase Bid", "Sell Bid", "MCV", "MCP"]
colors  = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD"]

n = len(columns)

fig, axes = plt.subplots(n, n, figsize=(14, 12))
fig.suptitle("IEX Scatter Matrix", fontsize=16, fontweight="bold", y=1.01)

for i, (col_y, label_y) in enumerate(zip(columns, labels)):
    for j, (col_x, label_x, color) in enumerate(zip(columns, labels, colors)):

        ax = axes[i, j]

        if i == j:
            # Diagonal — histogram
            ax.hist(data[col_x], bins=20, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
        else:
            x = data[col_x]
            y = data[col_y]

            # Scatter
            ax.scatter(x, y, s=15, color=color, alpha=0.6)

            # OLS trendline
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, m * x_line + b, color="red", linewidth=1.2, linestyle="--", alpha=0.7)

            # Correlation coefficient
            r = np.corrcoef(x, y)[0, 1]
            ax.annotate(f"r = {r:.2f}", xy=(0.95, 0.05), xycoords="axes fraction",
                        ha="right", fontsize=8, color="darkred",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        # Axis labels on outer edges only
        ax.set_xlabel(label_x if i == n - 1 else "", fontsize=9)
        ax.set_ylabel(label_y if j == 0 else "", fontsize=9)

        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("iex_scatter_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Adding relevant features

# %%
What i believe should be the variables over which the next day bids would depend should be
1. Previous few days purchase bids of that particular period
2. Previous few days purchase bids of period surrounding that particular period # not needed
3. If i can get the whole previous 2-3 days to forecast next days energy bids, that should be perfect. # not needed
5. week day enum
6. period_enum

# %%

# %%
data["pb_lag1d"] = data["purchase_bid"].shift(96)
data["sb_lag1d"] = data["sell_bid"].shift(96)

data["pb_lag2d"] = data["purchase_bid"].shift(192)
data["sb_lag2d"] = data["sell_bid"].shift(192)

data_clean = data.dropna()

print(f"New feature count: {len(data_clean.columns)}")
data_clean.head()

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

# %%

# %%

# %%

# %%

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Checking uncleared volume as a predictor.
# It can act as a predictor but its correlated with purchase bids.

# %%
data["ucv"] = data["purchase_bid"] - data["mcv"]
from statsmodels.tsa.stattools import grangercausalitytests
# Test if 'ucv' helps predict 'purchase_bid' up to 4 days (lags)
grangercausalitytests(data[['purchase_bid', 'ucv']], maxlag=4)

data['ucv_lag24h'] = data['ucv'].shift(96)

# 2. Calculate the "Bid Change" (Today's bid vs Yesterday's bid)
# This helps see if high UCV leads to an *increase* in bidding volume
data['pb_change'] = data['purchase_bid'] - data['pb_lag1d']

# 3. Filter for a specific period (e.g., 04:00 AM / Block 17) to reduce noise
block_17_data = data[data['period_num'] == 62].dropna()

# 4. Correlation Analysis
correlation = block_17_data['ucv_lag24h'].corr(block_17_data['pb_change'])
print(f"Correlation between Yesterday's UCV and Today's Purchase Bid (Block 17): {correlation:.4f}")

# 5. Visualization
plt.figure(figsize=(10, 6))
sns.regplot(data=block_17_data, x='ucv_lag24h', y='pb_change', 
            scatter_kws={'alpha':0.5, 'color':'#378ADD'}, 
            line_kws={'color':'red', 'ls':'--'})

plt.title("Impact of Yesterday's Uncleared Volume on Today's Purchase Bid (04:00 AM)", fontsize=14)
plt.xlabel("Uncleared Volume (UCV) Yesterday [MW]", fontsize=12)
plt.ylabel("Purchase Bid Today [MW]", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# %%

# %%
from statsmodels.tsa.stattools import adfuller

def perform_adf_test(series):
    result = adfuller(series.dropna())
    
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')

    # print("result: ", result)
    if result[1] <= 0.05:
        print("\nConclusion: Reject Null Hypothesis. Data is STATIONARY.")
    else:
        print("\nConclusion: Fail to Reject Null. Data is NON-STATIONARY (needs differencing).")

# Run it on your purchase bids
perform_adf_test(data['purchase_bid'])

# %%
# x = data["diff_bid_norm"].dropna()
x = data[data.mcp != 10000]["ratio_bid"]
mu, std = x.mean(), x.std()

fig, ax = plt.subplots(figsize=(10, 5))

# Histogram (density=True so it shares y-scale with the PDF)
ax.hist(x, bins=30, density=True, color="#378ADD", alpha=0.6,
        edgecolor="white", linewidth=0.5, label="diff_bid")

# Gaussian normal PDF
x_line = np.linspace(x.min(), x.max(), 300)
ax.plot(x_line, stats.norm.pdf(x_line, mu, std),
        color="red", linewidth=2, linestyle="--", label=f"Normal  μ={mu:,.1f}  σ={std:,.1f}")

# Stats annotation
ax.annotate(
    f"n = {len(x)}\nμ = {mu:,.1f}\nσ = {std:,.1f}\nskew = {stats.skew(x):.2f}\nkurt = {stats.kurtosis(x):.2f}",
    xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="lightgray", alpha=0.9),
)

ax.set_title("Distribution of Bid Difference (Sell − Purchase)", fontsize=14, pad=10)
ax.set_xlabel("Diff Bid (MW)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("diff_bid_histogram.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
x = data[data.mcp!=10000]["ratio_bid"].dropna()
y = data["mcp"].dropna()

# Align index in case of any NaNs
common = x.index.intersection(y.index)
x, y = x[common], y[common]

# ── OLS regression ────────────────────────────────────────────────────────────
slope, intercept, r, p_value, std_err = stats.linregress(x, y)
x_line = np.linspace(x.min(), x.max(), 300)
y_line = slope * x_line + intercept
r2 = r ** 2

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(x, y, color="#378ADD", alpha=0.6, s=25, edgecolors="white", linewidths=0.4, label="Data points")
ax.plot(x_line, y_line, color="red", linewidth=2, linestyle="--",
        label=f"OLS fit:  y = {slope:.2f}x + {intercept:.2f}")

# Annotation box
assessment = (
    "Strong" if abs(r) >= 0.7
    else "Moderate" if abs(r) >= 0.4
    else "Weak"
)
direction = "positive" if r > 0 else "negative"
sig = "significant" if p_value < 0.05 else "not significant"

ax.annotate(
    f"r  = {r:.3f}\n"
    f"R² = {r2:.3f}\n"
    f"p  = {p_value:.3e}\n"
    f"slope = {slope:.2f}\n\n"
    f"{assessment} {direction} linear\nrelationship ({sig})",
    xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="lightgray", alpha=0.9),
)

ax.set_title("MCP vs Ratio Bid (purchase / sell)", fontsize=14, pad=10)
ax.set_xlabel("Ratio Bid  (purchase_bid / sell_bid)", fontsize=11)
ax.set_ylabel("MCP  (₹/MWh)", fontsize=11)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("mcp_vs_ratio_bid.png", dpi=150, bbox_inches="tight")
plt.show()
