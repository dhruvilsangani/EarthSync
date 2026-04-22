from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forecasting.config import load_config
from src.forecasting.data import load_and_validate_raw_dataset
from src.forecasting.features.feature_builder import build_feature_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build modeling dataset from raw DAM CSV.")
    parser.add_argument("--config", default="configs/base.yaml", help="YAML config path.")
    parser.add_argument("--input", default=None, help="Optional raw input override.")
    parser.add_argument("--output", default=None, help="Optional processed output override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    src = Path(args.input or config["data"]["raw_input_csv"])
    dst = Path(args.output or config["data"]["input_csv"])
    dst.parent.mkdir(parents=True, exist_ok=True)

    raw_df, metadata = load_and_validate_raw_dataset(
        csv_path=src,
        timestamp_col=config["data"]["timestamp_col"],
        expected_frequency=config["data"]["frequency"],
    )
    feature_df, feature_diagnostics = build_feature_frame(
        raw_df=raw_df,
        timestamp_col=config["data"]["timestamp_col"],
        targets=config["targets"],
        include_time=bool(config["features"]["time"]["enabled"]),
        include_market=bool(config["features"]["market"]["enabled"]),
        include_prev_day_features=bool(config["features"]["market"]["include_prev_day_features"]),
        include_weather=bool(config["features"]["weather"]["enabled"]),
        weather_cache_csv=str(config["features"]["weather"]["cache_csv"]),
        weather_refresh_cache=bool(config["features"]["weather"]["refresh_cache"]),
        weather_locations=dict(config["features"]["weather"]["locations"]),
        is_evening_blocks=list(config["features"]["time"]["is_evening_blocks"]),
        lags=list(config["features"]["lags"]),
        rolling_windows=list(config["features"]["rolling_windows"]),
        dropna_after_feature_build=bool(config["features"]["dropna_after_feature_build"]),
    )

    feature_df.to_csv(dst, index=False)
    report_path = dst.parent / "feature_build_report.json"
    report = {
        **metadata,
        **feature_diagnostics,
        "output_rows": int(len(feature_df)),
        "output_columns": int(len(feature_df.columns)),
        "null_cells_after_build": int(feature_df.isna().sum().sum()),
        "constant_columns": [c for c in feature_df.columns if feature_df[c].nunique(dropna=False) <= 1],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {len(feature_df)} rows to {dst}")
    print(f"Wrote feature build report to {report_path}")


if __name__ == "__main__":
    main()
