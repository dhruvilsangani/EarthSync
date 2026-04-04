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
data = pd.read_csv("../../Downloads/iex_dam_feb_mar_2026.csv")

# %%
data

# %%
data.info()

# %%
data.describe()

# %% [markdown]
# ### Data Visualization - naive plots

# %%
data["period_start"] = pd.to_datetime(data["period_start"])

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

# %%
## from the above plots, I am observing
## purchase bids 
### spike at around 7:30 am in the morning, and 6:30 pm in the evening
### A slight downtrend moving from february to march.
## sell bids
### spike at around 01:00 pm, then lower at 06:30 pm, and rise again at around 2:30 am, and lower at 07:00 pm
## MCV
### follows almost the same trend as sell bids, but with some significant noise.
## MCP
### whenever sell bid is low, MCP is high.
### capped at 10000

## interesting things
# 1. There is a massive simultaneous supply shortage from march 11 to 15 and demand spike during the same period.

## findings
## 1. MCV and Sell_bid are strongly correlated
## 2. MCP and ratio(purchase/sell) are linearly related (when 10,000 cap is not breached)

## what's happening with the difference of energy between sell_bid and MCV

## ideally, the energy purchase bid today at a particular period should depend on a similar period yesterday or a few days before.
## 04:00 should depend on 03:00 to 05:00 

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
data["period_enum"] = data["period_start"].dt.hour * 4 + data["period_start"].dt.minute // 15 + 1
data["weekday_enum"] = data["period_start"].dt.weekday + 1

# %%
# 1. Add +-1 and +-2 period values from the previous day
# Since 1 period = 15 mins, lag 95 is "15 mins after same time yesterday" 
# and lag 97 is "15 mins before same time yesterday"

data["pb_lag1d"] = data["purchase_bid"].shift(96)
data["sb_lag1d"] = data["sell_bid"].shift(96)

data["pb_lag2d"] = data["purchase_bid"].shift(192)
data["sb_lag2d"] = data["sell_bid"].shift(192)

## The below are not of much help
# # Neighbors (Looking at the "Shape" around the same time yesterday)
# data["pb_lag1d_plus1"] = data["purchase_bid"].shift(96 - 1)  # 15 mins after
# data["pb_lag1d_plus2"] = data["purchase_bid"].shift(96 - 2)  # 30 mins after
# data["pb_lag1d_minus1"] = data["purchase_bid"].shift(96 + 1) # 15 mins before
# data["pb_lag1d_minus2"] = data["purchase_bid"].shift(96 + 2) # 30 mins before

# # Do the same for Sell Bids to give the model a view of supply-side "shape"
# data["sb_lag1d_plus1"] = data["sell_bid"].shift(96 - 1)
# data["sb_lag1d_plus2"] = data["sell_bid"].shift(96 - 2)
# data["sb_lag1d_minus1"] = data["sell_bid"].shift(96 + 1)
# data["sb_lag1d_minus2"] = data["sell_bid"].shift(96 + 2)

# Drop the new NaNs created by the larger shifts
data_clean = data.dropna()

print(f"New feature count: {len(data_clean.columns)}")
data_clean.head()

# %% [markdown]
# ### TSA decompose

# %%
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. Ensure your index has a frequency (required for seasonal_decompose)
indexed_data = data.set_index('period_start')
indexed_data = indexed_data.asfreq('15min')

# 2. Perform Classical Decomposition
result = seasonal_decompose(indexed_data['purchase_bid'], model='additive', period=96)

# 3. Save the residuals
indexed_data['pb_resid'] = result.resid

# 4. Plot using matplotlib
plt.rcParams['figure.figsize'] = (12, 8)
result.plot()
plt.show()

# %% [markdown]
# ## 

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("ACF & PACF — Purchase Bid and Sell Bid", fontsize=14, fontweight="bold")

plot_acf(data["purchase_bid"].diff(96).diff().dropna(),  lags=400, ax=axes[0, 0], title="ACF — Purchase Bid")
plot_pacf(data["purchase_bid"].diff(96).diff().dropna(), lags=400, ax=axes[0, 1], title="PACF — Purchase Bid")
plot_acf(data["sell_bid"].diff(96).diff().dropna(),      lags=400, ax=axes[1, 0], title="ACF — Sell Bid")
plot_pacf(data["sell_bid"].diff(96).diff().dropna(),     lags=400, ax=axes[1, 1], title="PACF — Sell Bid")

# plot_acf(data["purchase_bid"].dropna(),  lags=200, ax=axes[0, 0], title="ACF — Purchase Bid")
# plot_pacf(data["purchase_bid"].dropna(), lags=200, ax=axes[0, 1], title="PACF — Purchase Bid")
# plot_acf(data["sell_bid"].dropna(),      lags=200, ax=axes[1, 0], title="ACF — Sell Bid")
# plot_pacf(data["sell_bid"].dropna(),     lags=200, ax=axes[1, 1], title="PACF — Sell Bid")

for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()

# %%

# %%

# %%

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Checking if we can fit individual time series for each period. Seems like ARIMA(1,1,1) can work.

# %%
period_enum = 64

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data[data["period_enum"] == period_enum]["period_start"],
    y=data[data["period_enum"] == period_enum]["purchase_bid"],
    mode="lines+markers",
    line=dict(color="#378ADD", width=2),
    marker=dict(size=4),
    hovertemplate="<b>%{x|%H:%M}</b><br>Purchase Bid: %{y:,.1f} MW<extra></extra>",
))

fig.add_trace(go.Scatter(
    x=data[data["period_enum"] == period_enum]["period_start"],
    y=data[data["period_enum"] == period_enum]["sell_bid"],
    mode="lines+markers",
    line=dict(color="#378ADD", width=2),
    marker=dict(size=4),
    hovertemplate="<b>%{x|%H:%M}</b><br>Sell Bid: %{y:,.1f} MW<extra></extra>",
))

fig.update_layout(
    title="Purchase Bid over Time",
    xaxis_title="Time",
    yaxis_title="MW",
    hovermode="x unified",
    plot_bgcolor="white",
    yaxis=dict(gridcolor="rgba(150,150,150,0.2)"),
)

fig.show("notebook")
plot_acf(data[data["period_enum"] == period_enum]["purchase_bid"], lags=15)

# %% [markdown]
# ### Seeing if uncleared volume can work as a predictor. It can act as a predictor but its correlated with purchase bids.

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

# %%

# %%

# %%

# %% jupyter={"outputs_hidden": true}

# %%

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
