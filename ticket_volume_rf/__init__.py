"""Stepwise Random Forest selection for monthly ticket-volume forecasts.

The planning loop has two modes that share one implementation:

* **SI** (``refit_each_month=False``): fit once per planning month, then walk
  the offset horizon with that forest.
* **MI** (``refit_each_month=True``): refit before every target month so
  in-horizon forecasts can influence the next fit.

Feature selection is scored on a validation window. The accuracy that is
reported uses a later holdout window the selector never sees.
"""

from ticket_volume_rf.calendar import advance_yyyymm, validate_yyyymm
from ticket_volume_rf.forecast import forecast_panel
from ticket_volume_rf.io import load_panel
from ticket_volume_rf.pipeline import run_pipeline
from ticket_volume_rf.selection import (
    CATEGORICAL_FEATURES,
    DEFAULT_FEATURES,
    greedy_select,
)
from ticket_volume_rf.synthetic import make_synthetic_panel

__all__ = [
    "CATEGORICAL_FEATURES",
    "DEFAULT_FEATURES",
    "advance_yyyymm",
    "forecast_panel",
    "greedy_select",
    "load_panel",
    "make_synthetic_panel",
    "run_pipeline",
    "validate_yyyymm",
]
