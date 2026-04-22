"""Data loading and indexing utilities."""

from pathlib import Path

import pandas as pd

from forecasting.config.defaults import DEFAULT_FREQ
from forecasting.data.schema import TIME_COL


def load_market_data(path: Path | str, freq: str = DEFAULT_FREQ) -> pd.DataFrame:
    data = pd.read_csv(path)
    data[TIME_COL] = pd.to_datetime(data[TIME_COL])
    indexed = data.set_index(TIME_COL).sort_index()
    return indexed.asfreq(freq)

