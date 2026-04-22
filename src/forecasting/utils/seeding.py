"""Helpers to keep experiments reproducible."""

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set global random state for standard libraries."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        tf.random.set_seed(seed)
    except Exception:
        # TensorFlow is optional for non-LSTM workflows.
        pass

