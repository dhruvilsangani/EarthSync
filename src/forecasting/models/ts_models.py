"""Classical time-series model helpers."""

from statsforecast import StatsForecast
from statsforecast.models import ARIMA, HistoricAverage, Naive, SeasonalNaive, WindowAverage


def baseline_statsforecast(freq: str = "15min"):
    models = [Naive(), HistoricAverage(), WindowAverage(window_size=96 * 7), SeasonalNaive(season_length=96)]
    return StatsForecast(models=models, freq=freq)


def sarima_candidates(freq: str = "15min"):
    models = [
        ARIMA(order=(0, 1, 0), season_length=96, seasonal_order=(1, 1, 1), alias="SARIMA_010_111_96"),
        ARIMA(order=(1, 1, 1), season_length=96, seasonal_order=(1, 1, 1), alias="SARIMA_111_111_96"),
    ]
    return StatsForecast(models=models, freq=freq)

