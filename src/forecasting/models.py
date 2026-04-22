from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class ModelSpec:
    name: str
    estimator: Any | None
    uses_naive: bool = False


def build_model_specs(candidates: list[str], random_state: int) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for name in candidates:
        key = name.lower().strip()
        if key == "seasonal_naive":
            specs.append(ModelSpec(name="seasonal_naive", estimator=None, uses_naive=True))
        elif key == "hist_gbrt":
            specs.append(
                ModelSpec(
                    name="hist_gbrt",
                    estimator=HistGradientBoostingRegressor(random_state=random_state),
                )
            )
        elif key == "lightgbm":
            if os.getenv("ENABLE_LIGHTGBM", "0") == "1":
                try:
                    from lightgbm import LGBMRegressor

                    specs.append(
                        ModelSpec(
                            name="lightgbm",
                            estimator=LGBMRegressor(random_state=random_state, verbose=-1),
                        )
                    )
                except Exception:
                    pass
        elif key == "xgboost":
            if os.getenv("ENABLE_XGBOOST", "0") == "1":
                try:
                    from xgboost import XGBRegressor

                    specs.append(
                        ModelSpec(
                            name="xgboost",
                            estimator=XGBRegressor(
                                random_state=random_state,
                                n_estimators=300,
                                max_depth=6,
                                learning_rate=0.05,
                                objective="reg:squarederror",
                            ),
                        )
                    )
                except Exception:
                    pass
        elif key == "lstm":
            # Keep LSTM optional to avoid runtime instability in some local envs.
            if os.getenv("ENABLE_TF_LSTM", "0") == "1":
                try:
                    import tensorflow as tf  # noqa: F401

                    specs.append(ModelSpec(name="lstm", estimator="lstm"))
                except Exception:
                    pass
    return specs


def fit_predict_standard(estimator: Any, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    estimator.fit(x_train, y_train)
    return estimator.predict(x_test)


def fit_predict_lstm(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    tf.random.set_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(x_train.shape[1], 1)),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mae")
    x_train_3d = x_train[..., np.newaxis]
    x_test_3d = x_test[..., np.newaxis]
    model.fit(x_train_3d, y_train, epochs=10, batch_size=64, verbose=0)
    preds = model.predict(x_test_3d, verbose=0).reshape(-1)
    return preds
