"""Feature builders for market and weather-enriched datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from forecasting.data.schema import TIME_COL
from forecasting.utils.logging import get_logger

_LOGGER = get_logger(__name__)


def _ensure_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with `period_start` available as a datetime column."""
    out = df.copy()
    if TIME_COL not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
        else:
            raise ValueError(
                f"Input dataframe must contain `{TIME_COL}` column or a DatetimeIndex."
            )
    out[TIME_COL] = pd.to_datetime(out[TIME_COL])
    return out


def _project_root() -> Path:
    """Return project root based on current module location."""
    return Path(__file__).resolve().parents[3]


def get_nci_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch weighted NCI weather data from Open-Meteo archive API."""
    locations = {
        "delhi": {"lat": 28.61, "lon": 77.20, "weight": 0.5},
        "mumbai": {"lat": 19.07, "lon": 72.88, "weight": 0.3},
        "akola": {"lat": 20.70, "lon": 77.01, "weight": 0.2},
    }

    all_weather = pd.DataFrame()
    url = "https://archive-api.open-meteo.com/v1/archive"

    for name, loc in locations.items():
        params = {
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max",
            "timezone": "Asia/Kolkata",
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        if "daily" not in payload:
            raise ValueError(f"Open-Meteo response missing `daily` for city `{name}`.")

        temp_df = pd.DataFrame(
            {
                "date": pd.to_datetime(payload["daily"]["time"]),
                f"temp_{name}": payload["daily"]["temperature_2m_max"],
            }
        )

        if all_weather.empty:
            all_weather = temp_df
        else:
            all_weather = all_weather.merge(temp_df, on="date", how="inner")

    all_weather["NCI"] = 0.0
    for name, loc in locations.items():
        all_weather["NCI"] += all_weather[f"temp_{name}"] * loc["weight"]

    return all_weather.sort_values("date").reset_index(drop=True)


def get_weather_master(
    start_date: str,
    end_date: str,
    weather_master_path: Path | str | None = None,
    save_if_fetched: bool = True,
) -> pd.DataFrame:
    """
    Load `weather_master` from disk or fetch from API if unavailable.

    Priority:
    1) read existing CSV from `weather_master_path`
    2) fetch via `get_nci_data(start_date, end_date)`
    3) save fetched file (optional)
    """
    if weather_master_path is None:
        weather_master_path = _project_root() / "data" / "weather" / "weather_master.csv"

    weather_master_path = Path(weather_master_path)
    if weather_master_path.exists():
        _LOGGER.info("Loading weather master from %s", weather_master_path)
        weather_master = pd.read_csv(weather_master_path)
        weather_master["date"] = pd.to_datetime(weather_master["date"])
        return weather_master

    _LOGGER.warning(
        "Weather master not found at %s. Fetching from Open-Meteo API.",
        weather_master_path,
    )
    weather_master = get_nci_data(start_date=start_date, end_date=end_date)

    if save_if_fetched:
        weather_master_path.parent.mkdir(parents=True, exist_ok=True)
        weather_master.to_csv(weather_master_path, index=False)
        _LOGGER.info("Saved fetched weather master to %s", weather_master_path)

    return weather_master


def add_period_enum(df: pd.DataFrame) -> pd.DataFrame:
    """Add intra-day 15-minute block number (1..96) as `period_enum`."""
    out = _ensure_time_column(df)
    out["period_enum"] = out[TIME_COL].dt.hour * 4 + out[TIME_COL].dt.minute // 15 + 1
    return out


def add_weekday_enum(df: pd.DataFrame) -> pd.DataFrame:
    """Add weekday number (Monday=1..Sunday=7) as `weekday_enum`."""
    out = _ensure_time_column(df)
    out["weekday_enum"] = out[TIME_COL].dt.weekday + 1
    return out


def add_is_evening(
    df: pd.DataFrame,
    period_col: str = "period_enum",
    start_block: int = 73,
    end_block: int = 88,
) -> pd.DataFrame:
    """Add evening indicator based on period block range."""
    out = _ensure_time_column(df)
    if period_col not in out.columns:
        out = add_period_enum(out)
    out["is_evening"] = out[period_col].between(start_block, end_block).astype(int)
    return out


def add_nci(
    df: pd.DataFrame,
    weather_master: pd.DataFrame | None = None,
    weather_master_path: Path | str | None = None,
) -> pd.DataFrame:
    """Add daily `NCI` by merging weather data on market date."""
    out = _ensure_time_column(df)
    if weather_master is None:
        start_date = out[TIME_COL].min().date().isoformat()
        end_date = out[TIME_COL].max().date().isoformat()
        weather_master = get_weather_master(
            start_date=start_date,
            end_date=end_date,
            weather_master_path=weather_master_path,
        )
    if "date" not in weather_master.columns or "NCI" not in weather_master.columns:
        raise ValueError("`weather_master` must contain `date` and `NCI` columns.")

    weather = weather_master.copy()
    weather["date"] = pd.to_datetime(weather["date"])
    weather["date_only"] = weather["date"].dt.date

    out["date_only"] = out[TIME_COL].dt.date
    out = out.merge(weather[["date_only", "NCI"]], on="date_only", how="left")
    return out


def add_daily_clf(df: pd.DataFrame, nci_col: str = "NCI") -> pd.DataFrame:
    """Add daily cooling load factor as `daily_clf`."""
    out = _ensure_time_column(df)
    if nci_col not in out.columns:
        raise ValueError(f"Column `{nci_col}` not found. Run `add_nci()` first.")
    out["daily_clf"] = (out[nci_col] - 30).clip(lower=0) ** 2
    return out


def add_rolling_3d_clf(df: pd.DataFrame, daily_clf_col: str = "daily_clf") -> pd.DataFrame:
    """Add `rolling_3d_clf` using the last 3 daily CLF values."""
    out = _ensure_time_column(df)
    if daily_clf_col not in out.columns:
        out = add_daily_clf(out)

    out["date_only"] = out[TIME_COL].dt.date
    daily_map = (
        out[["date_only", daily_clf_col]]
        .drop_duplicates("date_only")
        .sort_values("date_only")
        .rename(columns={daily_clf_col: "daily_clf_for_roll"})
    )
    daily_map["rolling_3d_clf"] = (
        daily_map["daily_clf_for_roll"].rolling(window=3, min_periods=1).mean()
    )
    out = out.merge(daily_map[["date_only", "rolling_3d_clf"]], on="date_only", how="left")
    return out


def add_uncleared_vol(
    df: pd.DataFrame,
    purchase_bid_col: str = "purchase_bid",
    mcv_col: str = "mcv",
) -> pd.DataFrame:
    """Add `uncleared_vol` as purchase bid minus market cleared volume."""
    out = _ensure_time_column(df)
    required = [purchase_bid_col, mcv_col]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Missing columns for uncleared volume: {missing}.")
    out["uncleared_vol"] = out[purchase_bid_col] - out[mcv_col]
    return out


def add_prev_day_uncleared_vol(
    df: pd.DataFrame,
    periods_per_day: int = 96,
    source_col: str = "uncleared_vol",
) -> pd.DataFrame:
    """Add `prev_day_uncleared_vol` by lagging uncleared volume by one day."""
    out = _ensure_time_column(df)
    if source_col not in out.columns:
        out = add_uncleared_vol(out)
    out["prev_day_uncleared_vol"] = out[source_col].shift(periods_per_day)
    _LOGGER.warning(
        "Added `prev_day_uncleared_vol` with shift=%s. "
        "Truncate leading rows with NaN before model training.",
        periods_per_day,
    )
    return out


def add_prev_day_purchase_bid(
    df: pd.DataFrame,
    periods_per_day: int = 96,
    purchase_bid_col: str = "purchase_bid",
) -> pd.DataFrame:
    """Add `prev_day_purchase_bid` by lagging purchase bid by one day."""
    out = _ensure_time_column(df)
    if purchase_bid_col not in out.columns:
        raise ValueError(f"Column `{purchase_bid_col}` not found.")
    out["prev_day_purchase_bid"] = out[purchase_bid_col].shift(periods_per_day)
    _LOGGER.warning(
        "Added `prev_day_purchase_bid` with shift=%s. "
        "Truncate leading rows with NaN before model training.",
        periods_per_day,
    )
    return out
