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
2. Previous few days purchase bids of period surrounding that particular period
3. If i can get the whole previous 2-3 days to forecast next days energy bids, that should be perfect.
4. is_weekend
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
data["sb_lag1d"] = data["purchase_bid"].shift(96)

# Neighbors (Looking at the "Shape" around the same time yesterday)
data["pb_lag1d_plus1"] = data["purchase_bid"].shift(96 - 1)  # 15 mins after
data["pb_lag1d_plus2"] = data["purchase_bid"].shift(96 - 2)  # 30 mins after
data["pb_lag1d_minus1"] = data["purchase_bid"].shift(96 + 1) # 15 mins before
data["pb_lag1d_minus2"] = data["purchase_bid"].shift(96 + 2) # 30 mins before

# Do the same for Sell Bids to give the model a view of supply-side "shape"
data["sb_lag1d_plus1"] = data["sell_bid"].shift(96 - 1)
data["sb_lag1d_plus2"] = data["sell_bid"].shift(96 - 2)
data["sb_lag1d_minus1"] = data["sell_bid"].shift(96 + 1)
data["sb_lag1d_minus2"] = data["sell_bid"].shift(96 + 2)

# Drop the new NaNs created by the larger shifts
data_clean = data.dropna()

print(f"New feature count: {len(data_clean.columns)}")
data_clean.head()

# %%

# %%

# %%

# %%
period_enum = 64

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data[data["period_num"] == period_enum]["period_start"],
    y=data[data["period_num"] == period_enum]["purchase_bid"],
    mode="lines+markers",
    line=dict(color="#378ADD", width=2),
    marker=dict(size=4),
    hovertemplate="<b>%{x|%H:%M}</b><br>Purchase Bid: %{y:,.1f} MW<extra></extra>",
))

fig.add_trace(go.Scatter(
    x=data[data["period_num"] == period_enum]["period_start"],
    y=data[data["period_num"] == period_enum]["sell_bid"],
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

# %%
period_enum = 55
plot_pacf(data[data["period_num"] == period_enum]["purchase_bid"], lags=15)

# %%
plot_acf(data["purchase_bid"], lags = 100)

# %%

# %%

# %%

# %%
result = adfuller(data[data["period_num"] == 36]["purchase_bid"].diff().dropna())
result[1]

# %%
ARIMA(1, 1, 2) 

# %%

# %%

# %%

# %%
# Ensure your index has a frequency
indexed_data = data.set_index('period_start')
indexed_data = indexed_data.asfreq('15min') 

stl = STL(indexed_data['purchase_bid'], period=96, seasonal=7, trend = 96*3+1)
res = stl.fit()

# The 'resid' is your de-trended and de-seasonalized data
indexed_data['pb_resid'] = res.resid
res.plot()
plt.show()

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(res.resid, lags=100)
# plot_pacf(res.resid, lags=100)

# %%
from statsmodels.tsa.stattools import adfuller
result = adfuller(data.purchase_bid.diff().dropna())
result[1]

# %%
plot_pacf(data.purchase_bid.dropna(), lags=300)

# %%

# %%

# %%

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Prepare the components from your 'res' object
components = {
    "Observed": res.observed,
    "Trend": res.trend,
    "Seasonal": res.seasonal,
    "Residuals": res.resid
}

# 2. Create subplots
fig = make_subplots(
    rows=4, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05,
    subplot_titles=list(components.keys())
)

# 3. Add traces
for i, (name, series) in enumerate(components.items(), start=1):
    fig.add_trace(
        go.Scatter(
            x=series.index, 
            y=series, 
            name=name,
            line=dict(width=1.5),
            mode='lines'
        ),
        row=i, col=1
    )

# 4. Update layout for better readability
fig.update_layout(
    height=900, 
    title_text="STL Decomposition of Purchase Bid",
    showlegend=False,
    template="plotly_white"
)

# Optional: Add a zero-line for the Residuals subplot
fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)

fig.show()

# %%

# %%

# %%

# %%

# %%
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 1. Create a "Shock" indicator for the March 13-17 dip
data['weather_shock'] = 0
data.loc['2026-03-13':'2026-03-17', 'weather_shock'] = 1

# 2. Add weekend/weekday info (highly correlated with industrial bids)
data['is_weekend'] = (data.index.weekday >= 5).astype(int)

# 3. Fit SARIMAX with these as external inputs
# This tells the model: "The drop here is due to 'weather_shock', don't assume it's part of the normal trend."
exog_cols = ['is_weekend', 'weather_shock']
model = SARIMAX(data['purchase_bid'], 
                exog=data[exog_cols], 
                order=(1, 1, 1), 
                seasonal_order=(1, 1, 1, 96))

results = model.fit()

# %%
print(results.summary())

# %%
# 1. Prepare future exogenous data for the next 96 periods (24 hours)
last_date = data.index[-1]
forecast_index = pd.date_range(start=last_date + pd.Timedelta(minutes=15), periods=96, freq='15min')

future_exog = pd.DataFrame(index=forecast_index)
future_exog['is_weekend'] = (future_exog.index.weekday >= 5).astype(int)
future_exog['weather_shock'] = 0  # Assuming the shock has ended

# 2. Get the forecast
forecast_res = results.get_forecast(steps=96, exog=future_exog)
forecast_mean = forecast_res.summary_frame()['mean']
conf_int = forecast_res.conf_int()

# %%
fig = go.Figure()

# Plot recent actual data (last 3 days for context)
recent_data = data.tail(96 * 3)
fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['purchase_bid'], 
                         name='Actual', line=dict(color='#378ADD')))

# Plot Forecast
fig.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean, 
                         name='Forecast', line=dict(color='red', dash='dash')))

# Plot Confidence Interval (shaded area)
fig.add_trace(go.Scatter(x=forecast_mean.index.tolist() + forecast_mean.index[::-1].tolist(),
                         y=conf_int['upper purchase_bid'].tolist() + conf_int['lower purchase_bid'][::-1].tolist(),
                         fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,0,0,0)'),
                         hoverinfo="skip", name='95% Confidence Interval'))

fig.update_layout(title="24-Hour Purchase Bid Forecast", 
                  xaxis_title="Time", yaxis_title="MW",
                  template="plotly_white", hovermode="x unified")
fig.show()

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
import matplotlib.pyplot as plt


fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("ACF & PACF — Purchase Bid and Sell Bid", fontsize=14, fontweight="bold")

plot_acf(data["purchase_bid"].dropna(),  lags=100, ax=axes[0, 0], title="ACF — Purchase Bid")
plot_pacf(data["purchase_bid"].dropna(), lags=100, ax=axes[0, 1], title="PACF — Purchase Bid")
plot_acf(data["sell_bid"].dropna(),      lags=100, ax=axes[1, 0], title="ACF — Sell Bid")
plot_pacf(data["sell_bid"].dropna(),     lags=100, ax=axes[1, 1], title="PACF — Sell Bid")

for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()

# %%
data["purchase_bid"].dropna()

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
# Assuming 15-min data (96 blocks per day)
seasonal_diff = data['purchase_bid'].diff(96).dropna()
seasonal_diff.plot()

# %%
seasonal_diff

# %%

# %%

# %%

# %%

# %%

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

# x = data[data.mcp!=10000]["diff_bid_norm"].dropna()

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
