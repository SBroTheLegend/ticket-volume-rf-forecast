"""Monthly planning loop shared by single-iteration and multiple-iteration fits.

For each planning month in ``(window_start, window_end)``:

1. Restore actuals and rebuild target-dependent features from history.
2. **SI** (``refit_each_month=False``): fit one forest on
   ``year_month < planning_month``.
3. Walk ``offset`` months. After each forecast, write it back into the
   target so the next month's lags and rolling stats use that forecast
   rather than the future actual.
4. **MI** (``refit_each_month=True``): refit on
   ``year_month < target_month`` before predicting that month.

Calendar Fourier terms and exogenous ratio lags are built once per call.
They do not change when a forecast is written back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ticket_volume_rf.calendar import advance_yyyymm, validate_yyyymm
from ticket_volume_rf.features import (
    prepare_panel,
    refresh_dynamic_features,
    sanitize_columns,
)
from ticket_volume_rf.metrics import capped_accuracy

logger = logging.getLogger(__name__)

TARGET = "ticket_volume"


@dataclass
class ForecastResult:
    """Predictions, capped accuracy, and averaged tree importances."""

    accuracy: float
    score_row: pd.DataFrame
    predictions: pd.DataFrame
    feature_importances: pd.DataFrame
    n_fits: int
    refit_each_month: bool


def default_regressor(
    *,
    n_estimators: int = 50,
    random_state: int = 2,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Forest used when the caller does not pass one.

    ``n_jobs=-1`` is the default because the multiple-iteration loop refits
    on every target month. Pass ``n_jobs=1`` when bit-for-bit repeats matter
    more than runtime.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def _matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    matrix = frame.loc[:, list(columns)].astype("float64").fillna(0.0)
    matrix.columns = sanitize_columns(columns)
    return matrix


def _average_importances(parts: list[pd.Series]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["feature", "importance"])
    averaged = pd.concat(parts, axis=1).mean(axis=1).sort_values(ascending=False)
    return averaged.rename("importance").rename_axis("feature").reset_index()


def _fit(model, train: pd.DataFrame, columns: list[str], target: str):
    matrix = _matrix(train, columns)
    model.fit(matrix, train[target].to_numpy(dtype=float))
    names = list(matrix.columns)
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=names, name="importance")
    return importance


def forecast_panel(
    df: pd.DataFrame,
    predictors: list[str] | None = None,
    qual_preds: list[str] | None = None,
    *,
    window_start: int,
    window_end: int,
    refit_each_month: bool = False,
    offset: int = 4,
    score_offset: int = 3,
    model=None,
    n_estimators: int = 50,
    n_jobs: int = -1,
    random_state: int = 2,
    demand_lag: int = 9,
    feature_no: int = 0,
    target: str = TARGET,
) -> ForecastResult:
    """Forecast every planning month in ``(window_start, window_end)``.

    Both bounds are exclusive YYYYMM values, matching the original test
    window. ``201901`` .. ``201906`` scores planning months 201902 through
    201905. ``window_end`` must be a real month: ``201924`` is rejected.

    Accuracy is the mean capped accuracy of rows whose offset equals
    ``score_offset`` (default 3, and ``offset`` must be larger than that).
    Feature importances are the mean across every fit in this call. The
    input frame is not modified.
    """
    if not 0 <= score_offset < offset:
        raise ValueError("score_offset must satisfy 0 <= score_offset < offset")
    window_start = validate_yyyymm(window_start)
    window_end = validate_yyyymm(window_end)
    if window_end <= window_start:
        raise ValueError("window_end must be after window_start")

    predictors = list(predictors or ["demand_lag_1"])
    qual_preds = list(qual_preds or [])
    forbidden = {target, "_actual"}
    leaked = sorted(forbidden & set(predictors))
    if leaked:
        raise ValueError(f"Target columns cannot be predictors: {leaked}")

    if model is None:
        model = default_regressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    work, dummy_cols = prepare_panel(df, qual_preds, demand_lag=demand_lag, target=target)
    # One dynamic refresh so missing engineered predictors fail before the loop.
    refresh_dynamic_features(work, target=target, demand_lag=demand_lag)
    missing = [col for col in predictors if col not in work.columns]
    if missing:
        raise KeyError(
            "Predictors not available after feature engineering: "
            f"{missing}. Lags, rolling stats, change ratios, and Fourier "
            "terms are created from ticket_volume, Date, and "
            "Quarterly_Ratio_Jan. Other columns must already be on the panel."
        )
    columns = list(predictors) + [col for col in dummy_cols if col not in predictors]

    months_present = set(int(value) for value in work["year_month"])
    planning_months = sorted(
        month for month in months_present if window_start < month < window_end
    )

    importance_parts: list[pd.Series] = []
    prediction_frames: list[pd.DataFrame] = []
    n_fits = 0

    for planning_month in planning_months:
        # Each origin starts from actuals. Forecasts written for a previous
        # origin must not leak into this one.
        work[target] = work["_actual"]
        refresh_dynamic_features(work, target=target, demand_lag=demand_lag)

        horizon = [advance_yyyymm(planning_month, step) for step in range(offset)]
        if not any(month in months_present for month in horizon):
            continue

        if not refit_each_month:
            train = work[work["year_month"] < planning_month]
            if train.empty:
                logger.info("Skipping %s: no history to fit", planning_month)
                continue
            importance = _fit(model, train, columns, target)
            n_fits += 1
            if importance is not None:
                importance_parts.append(importance)

        for step, target_month in enumerate(horizon):
            if step > 0:
                refresh_dynamic_features(work, target=target, demand_lag=demand_lag)
            test = work[work["year_month"] == target_month]
            if test.empty:
                continue
            if refit_each_month:
                train = work[work["year_month"] < target_month]
                if train.empty:
                    continue
                importance = _fit(model, train, columns, target)
                n_fits += 1
                if importance is not None:
                    importance_parts.append(importance)

            raw = np.clip(model.predict(_matrix(test, columns)), 0, None)
            # Unrounded values feed the next lag. Reported forecasts are rounded,
            # which is what the capped accuracy was originally computed on.
            work.loc[test.index, target] = raw
            reported = np.round(raw)
            actual = test["_actual"].to_numpy(dtype=float)
            frame = {
                "series_id": test["series_id"].to_numpy(),
                "Actual": actual,
                "Offset": step,
                "planning_month": planning_month,
                "target_month": target_month,
                "Fcst": reported,
                "acc": capped_accuracy(actual, reported),
            }
            if "region" in test.columns:
                frame["region"] = test["region"].to_numpy()
            prediction_frames.append(pd.DataFrame(frame))

    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions["Error"] = np.minimum(
            predictions["Actual"],
            (predictions["Actual"] - predictions["Fcst"]).abs(),
        )
    else:
        predictions = pd.DataFrame(
            columns=[
                "series_id",
                "Actual",
                "Offset",
                "planning_month",
                "target_month",
                "Fcst",
                "acc",
                "Error",
            ]
        )

    if predictions.empty:
        scored = pd.Series(dtype=float)
    else:
        scored = predictions.loc[predictions["Offset"] == score_offset, "acc"]
    accuracy = float(scored.mean()) if len(scored) else float("nan")

    score_row = pd.DataFrame(
        [
            {
                "Num_feat": ", ".join(predictors),
                "Cat_feat": ", ".join(qual_preds) if qual_preds else "No Cat Feat",
                "Acc": accuracy,
                "Feature_no": feature_no,
            }
        ]
    )
    logger.debug(
        "forecast mode=%s accuracy=%s fits=%s",
        "MI" if refit_each_month else "SI",
        accuracy,
        n_fits,
    )
    return ForecastResult(
        accuracy=accuracy,
        score_row=score_row,
        predictions=predictions,
        feature_importances=_average_importances(importance_parts),
        n_fits=n_fits,
        refit_each_month=refit_each_month,
    )
