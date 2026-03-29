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

# %%
data = pd.read_csv("../../Downloads/iex_dam_feb_mar_2026.csv")

# %%
data

# %%
data.info()

# %%
data.describe()

# %%
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Plot config ─────────────────────────────────────────────────────────────
columns = ["purchase_bid", "sell_bid", "mcv", "mcp"]
titles  = ["Purchase Bid (MW)", "Sell Bid (MW)", "MCV — Market Clearing Volume (MW)", "MCP — Market Clearing Price (₹/MWh)"]
colors  = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD"]

# ── Parse datetime ──────────────────────────────────────────────────────────
data["period_start"] = pd.to_datetime(data["period_start"])

# ── Build subplots ──────────────────────────────────────────────────────────
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    subplot_titles=titles,
    vertical_spacing=0.06,
)

for i, (col, title, color) in enumerate(zip(columns, titles, colors), start=1):
    fig.add_trace(
        go.Scatter(
            x=data["period_start"],
            y=data[col],
            mode="lines+markers",
            name=title,
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
            fill="tozeroy",
            fillcolor=color.replace("#", "rgba(") + ",0.10)".replace("rgba(", "rgba(")
                if False else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.10)",
            hovertemplate=f"<b>%{{x|%H:%M}}</b><br>{title}: %{{y:,.2f}}<extra></extra>",
        ),
        row=i, col=1,
    )

# ── Y-axis labels ───────────────────────────────────────────────────────────
y_labels = ["MW", "MW", "MW", "₹/MWh"]
for i, label in enumerate(y_labels, start=1):
    fig.update_yaxes(title_text=label, row=i, col=1, showgrid=True,
                     gridcolor="rgba(150,150,150,0.2)", zeroline=False)

# ── Layout ───────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(text="IEX Market Data", font=dict(size=18), x=0.5, xanchor="center"),
    height=900,
    showlegend=False,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(t=80, b=60, l=70, r=30),
    font=dict(family="Arial, sans-serif", size=12),
)

fig.show("notebook")

# %%
import pandas as pd
import plotly.graph_objects as go

# ── Parse datetime ──────────────────────────────────────────────────────────
data["period_start"] = pd.to_datetime(data["period_start"])

# ── Plot config ─────────────────────────────────────────────────────────────
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
    xaxis=dict(
        tickformat="%H:%M",
        dtick=3600000,
        tickangle=-45,
        showgrid=True,
        gridcolor="rgba(150,150,150,0.2)",
        title="Time",
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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
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

# %%
