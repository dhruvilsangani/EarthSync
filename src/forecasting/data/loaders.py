"""Data loading and indexing utilities."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from forecasting.config.defaults import DEFAULT_FREQ
from forecasting.data.schema import TIME_COL

DEFAULT_MARKET_START = date(2026, 2, 1)
DEFAULT_MARKET_END = date(2026, 4, 14)


def load_market_data(
    path: Path | str,
    freq: str = DEFAULT_FREQ,
    *,
    start_date: date | str | pd.Timestamp = DEFAULT_MARKET_START,
    end_date: date | str | pd.Timestamp = DEFAULT_MARKET_END,
) -> pd.DataFrame:
    data = pd.read_csv(path)
    data[TIME_COL] = pd.to_datetime(data[TIME_COL])
    indexed = data.set_index(TIME_COL).sort_index()
    start_ts = pd.Timestamp(start_date).normalize()
    end_exclusive = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)
    indexed = indexed.loc[(indexed.index >= start_ts) & (indexed.index < end_exclusive)]
    return indexed.asfreq(freq)

