"""Capped forecast accuracy used by feature selection and reporting."""

from __future__ import annotations

import numpy as np


def capped_accuracy(actual, forecast) -> np.ndarray:
    """Accuracy in ``[0, 1]`` for non-negative volumes.

    ``1 - min(actual, |actual - forecast|) / actual``. When the absolute
    error exceeds the actual, accuracy is 0 rather than negative. Zero
    actuals are left as NaN and are skipped by the mean.

    This is the offset-level score the stepwise loop originally averaged
    at ``Offset == 3``.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    error = np.abs(actual - forecast)
    capped = np.minimum(actual, error)
    out = np.full(actual.shape, np.nan, dtype=float)
    np.divide(capped, actual, out=out, where=actual != 0)
    return 1.0 - out
