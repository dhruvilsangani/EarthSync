"""LSTM model creation wrappers."""

from skforecast.deep_learning import create_and_compile_model


def build_default_lstm(series, lags: int = 96, steps: int = 96 * 7):
    return create_and_compile_model(
        series=series,
        lags=lags,
        steps=steps,
        recurrent_layer="LSTM",
        recurrent_units=[64, 32],
        dense_units=[32],
        recurrent_layers_kwargs={"dropout": 0.2},
        compile_kwargs={"optimizer": "adam", "loss": "mae"},
        model_name="earthsync-lstm",
    )

