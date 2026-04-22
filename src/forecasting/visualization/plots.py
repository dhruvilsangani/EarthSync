"""Plotly plotting helpers for forecast overlays."""

import plotly.graph_objects as go


def forecast_overlay(history, forecast, title: str, yaxis_title: str = "Volume"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, name="history", mode="lines"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="forecast", mode="lines"))
    fig.update_layout(
        title=title,
        xaxis_title="Date time",
        yaxis_title=yaxis_title,
        width=900,
        height=420,
    )
    return fig

