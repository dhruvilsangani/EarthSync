"""Rolling feature builders."""

from skforecast.preprocessing import RollingFeatures


def default_rolling_mean(window_size: int = 96 * 3) -> RollingFeatures:
    return RollingFeatures(stats=["mean"], window_sizes=window_size)

