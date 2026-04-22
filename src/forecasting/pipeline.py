from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import select_model_frame
from .evaluation import compute_metrics
from .models import build_model_specs, fit_predict_lstm, fit_predict_standard


def _safe_mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_backtest_for_target(
    df: pd.DataFrame,
    config: dict,
    target: str,
) -> pd.DataFrame:
    timestamp_col = config["data"]["timestamp_col"]
    holdout_start = pd.Timestamp(config["forecast"]["holdout_start"])
    holdout_end = pd.Timestamp(config["forecast"]["holdout_end"])

    feature_df = select_model_frame(
        df=df,
        target_col=target,
        timestamp_col=timestamp_col,
        shared_feature_cols=config["features"]["shared"],
    )

    mask = (feature_df[timestamp_col] >= holdout_start) & (feature_df[timestamp_col] <= holdout_end)
    train = feature_df.loc[~mask].copy()
    test = feature_df.loc[mask].copy()
    if test.empty:
        raise ValueError(f"No holdout rows for target={target}. Check holdout window.")

    x_cols = [c for c in feature_df.columns if c not in {timestamp_col, target}]
    x_train = train[x_cols].to_numpy()
    y_train = train[target].to_numpy()
    x_test = test[x_cols].to_numpy()
    y_test = test[target].to_numpy()

    results = []
    specs = build_model_specs(
        candidates=config["modeling"]["candidates"],
        random_state=int(config["modeling"]["random_state"]),
    )

    for spec in specs:
        if spec.uses_naive:
            # Uses previous day value if present among lag features.
            naive_col = f"{target}_lag_96"
            if naive_col in test.columns:
                preds = test[naive_col].to_numpy()
            else:
                preds = np.full_like(y_test, y_train.mean())
        elif spec.name == "lstm":
            preds = fit_predict_lstm(x_train, y_train, x_test)
        else:
            preds = fit_predict_standard(spec.estimator, x_train, y_train, x_test)

        metrics = compute_metrics(y_test, preds)
        row = {"target": target, "model": spec.name, **metrics}
        results.append(row)

    return pd.DataFrame(results).sort_values("MAPE").reset_index(drop=True)


def train_final_and_forecast(
    df: pd.DataFrame,
    config: dict,
    target: str,
    best_model_name: str,
) -> pd.DataFrame:
    timestamp_col = config["data"]["timestamp_col"]
    holdout_start = pd.Timestamp(config["forecast"]["holdout_start"])
    holdout_end = pd.Timestamp(config["forecast"]["holdout_end"])

    feature_df = select_model_frame(
        df=df,
        target_col=target,
        timestamp_col=timestamp_col,
        shared_feature_cols=config["features"]["shared"],
    )

    forecast_frame = feature_df[
        (feature_df[timestamp_col] >= holdout_start) & (feature_df[timestamp_col] <= holdout_end)
    ].copy()
    train_frame = feature_df[feature_df[timestamp_col] < holdout_start].copy()
    x_cols = [c for c in feature_df.columns if c not in {timestamp_col, target}]

    if best_model_name == "seasonal_naive":
        preds = forecast_frame[f"{target}_lag_96"].to_numpy()
    else:
        specs = {s.name: s for s in build_model_specs(config["modeling"]["candidates"], config["modeling"]["random_state"])}
        if best_model_name not in specs:
            raise ValueError(f"Model {best_model_name} is unavailable in this environment.")
        spec = specs[best_model_name]
        x_train = train_frame[x_cols].to_numpy()
        y_train = train_frame[target].to_numpy()
        x_test = forecast_frame[x_cols].to_numpy()
        if best_model_name == "lstm":
            preds = fit_predict_lstm(x_train, y_train, x_test)
        else:
            spec.estimator.fit(x_train, y_train)
            preds = spec.estimator.predict(x_test)

            model_dir = _safe_mkdir(Path(config["paths"]["artifacts_dir"]) / "models")
            model_path = model_dir / f"{target}_{best_model_name}.joblib"
            joblib.dump(spec.estimator, model_path)

    out = forecast_frame[[timestamp_col]].copy()
    out["target"] = target
    out["model"] = best_model_name
    out["prediction"] = preds
    if target in forecast_frame.columns:
        out["actual"] = forecast_frame[target].to_numpy()
    return out
