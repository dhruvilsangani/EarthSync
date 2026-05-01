"""Export holdout-week naive forecasts to CSV (same layout as `pb_boost_*.csv`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.config.defaults import DEFAULT_HORIZON, PREDICTIONS_DIR, PROJECT_ROOT
from forecasting.data.loaders import DEFAULT_MARKET_END, DEFAULT_MARKET_START, load_market_data
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw" / "iex-dam-0201-0421.csv"
PERIOD = 96


def naive_repeat_lag(train_series: pd.Series, horizon: int, lag: int) -> np.ndarray:
    base = train_series.iloc[-lag:].values
    return np.tile(base, int(np.ceil(horizon / lag)))[:horizon]


def naive_mean_last_k_days(train_series: pd.Series, horizon: int, k: int = 4) -> np.ndarray:
    tail = train_series.iloc[-k * PERIOD :].values.reshape(k, PERIOD)
    base = np.nanmean(tail, axis=0)
    return np.tile(base, int(np.ceil(horizon / PERIOD)))[:horizon]


def naive_weekday_block_mean(train_series: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
    frame = pd.DataFrame({"y": train_series.dropna()})
    frame["wd"] = frame.index.weekday
    frame["pe"] = frame.index.hour * 4 + frame.index.minute // 15
    table = frame.groupby(["wd", "pe"])["y"].mean()
    return np.array([table.loc[(t.weekday(), t.hour * 4 + t.minute // 15)] for t in test_index])


def _baseline_fns(horizon: int):
    return [
        ("Persist t-96 (yesterday)", lambda ts, idx: naive_repeat_lag(ts, horizon, 96)),
        ("Persist t-672 (last week)", lambda ts, idx: naive_repeat_lag(ts, horizon, 672)),
        ("Mean of last 4 days", lambda ts, idx: naive_mean_last_k_days(ts, horizon, k=4)),
        ("Weekday × block mean", lambda ts, idx: naive_weekday_block_mean(ts, idx)),
    ]


def _holdout_split(df: pd.DataFrame, horizon: int):
    train, test = df.iloc[:-horizon], df.iloc[-horizon:]
    return train, test


def _best_naive_pred(
    train: pd.DataFrame, test: pd.DataFrame, target: str, horizon: int
) -> tuple[np.ndarray, str]:
    y_true = test[target].values
    train_series = train[target].dropna()
    best_name, best_mae = None, float("inf")
    best_yhat = None
    for name, fn in _baseline_fns(horizon):
        yhat = fn(train_series, test.index)
        mae = float(np.mean(np.abs(y_true - yhat)))
        if mae < best_mae:
            best_mae, best_name, best_yhat = mae, name, yhat
    assert best_yhat is not None and best_name is not None
    return best_yhat, best_name


def _suffix_from_index(test_index: pd.DatetimeIndex) -> str:
    d0 = test_index.min().strftime("%m%d")
    d1 = test_index.max().strftime("%m%d")
    return f"{d0}_{d1}"


def write_boost_style_csv(path: Path, test_index: pd.DatetimeIndex, pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = test_index.strftime("%Y-%m-%d %H:%M:%S")
    out = pd.DataFrame({"timestamp": ts, "pred": pred})
    out.to_csv(path, index=True)


def export_holdout_best_naive(
    *,
    raw_path: Path | str = DEFAULT_RAW,
    out_dir: Path | None = None,
    horizon: int = DEFAULT_HORIZON,
    start_date=DEFAULT_MARKET_START,
    end_date=DEFAULT_MARKET_END,
) -> dict[str, object]:
    """
    For the final ``horizon``-step holdout, pick the naive with lowest MAE per target
    and write ``pb_naive_{mmdd}_{mmdd}.csv`` and ``sb_naive_{mmdd}_{mmdd}.csv``.
    """
    out_dir = out_dir or PREDICTIONS_DIR
    df = load_market_data(raw_path, start_date=start_date, end_date=end_date)
    train, test = _holdout_split(df, horizon)
    suf = _suffix_from_index(test.index)

    meta: dict[str, object] = {"test_start": test.index.min(), "test_end": test.index.max()}
    for col, prefix in [("purchase_bid", "pb"), ("sell_bid", "sb")]:
        yhat, name = _best_naive_pred(train, test, col, horizon)
        meta[f"{prefix}_baseline"] = name
        path = Path(out_dir) / f"{prefix}_naive_{suf}.csv"
        write_boost_style_csv(path, test.index, yhat)
        meta[f"{prefix}_path"] = path

    return meta


if __name__ == "__main__":
    info = export_holdout_best_naive()
    print("Wrote:", info["pb_path"], info["sb_path"])
    print("Baselines:", info["pb_baseline"], "|", info["sb_baseline"])
