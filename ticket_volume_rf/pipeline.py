"""Per-series stepwise selection on a validation window, scored on a holdout.

The selector only sees rows with ``year_month < selection_end_month``.
Reported accuracy is a second pass on planning months in
``(holdout_start_month, holdout_end_month)``, which must start at or after
``selection_end_month``. The holdout pass may train on earlier actuals,
including the selection period. It does not choose features.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ticket_volume_rf.calendar import validate_yyyymm
from ticket_volume_rf.forecast import ForecastResult, forecast_panel
from ticket_volume_rf.io import load_panel
from ticket_volume_rf.selection import (
    DEFAULT_FEATURES,
    SelectionOutcome,
    greedy_select,
)

logger = logging.getLogger(__name__)

_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class SeriesResult:
    """Selected features and the two window scores for one series."""

    series_id: object
    region: str
    numeric_features: list[str]
    categorical_features: list[str]
    validation_accuracy: float
    holdout_accuracy: float
    selection_trace: pd.DataFrame
    validation_predictions: pd.DataFrame
    holdout_predictions: pd.DataFrame
    feature_importances: pd.DataFrame
    holdout_fits: int


@dataclass
class PipelineResult:
    """Results for every series in the panel."""

    mode: str
    refit_each_month: bool
    series: list[SeriesResult]

    @property
    def holdout_accuracy(self) -> float:
        """Unweighted mean of per-series holdout accuracies."""
        if not self.series:
            return float("nan")
        return float(np.nanmean([item.holdout_accuracy for item in self.series]))

    @property
    def validation_accuracy(self) -> float:
        """Unweighted mean of per-series validation accuracies."""
        if not self.series:
            return float("nan")
        return float(np.nanmean([item.validation_accuracy for item in self.series]))

    def summary(self) -> pd.DataFrame:
        """One row per series: selected features and both accuracies."""
        rows = []
        for item in self.series:
            rows.append(
                {
                    "series_id": item.series_id,
                    "region": item.region,
                    "mode": self.mode,
                    "validation_acc": item.validation_accuracy,
                    "holdout_acc": item.holdout_accuracy,
                    "numeric_features": ", ".join(item.numeric_features),
                    "categorical_features": (
                        ", ".join(item.categorical_features) if item.categorical_features else "No Cat Feat"
                    ),
                    "n_features": len(item.numeric_features) + len(item.categorical_features),
                    "holdout_fits": item.holdout_fits,
                }
            )
        return pd.DataFrame(rows)


def resolve_refit(mode: str | None, refit_each_month: bool | None) -> bool:
    """Resolve SI/MI. ``mode`` and ``refit_each_month`` must agree when both are set."""
    from_mode = None
    if mode is not None:
        key = mode.strip().upper()
        if key not in {"SI", "MI"}:
            raise ValueError("mode must be 'SI' or 'MI'")
        from_mode = key == "MI"
    if from_mode is None and refit_each_month is None:
        return False
    if from_mode is not None and refit_each_month is not None and from_mode != refit_each_month:
        raise ValueError("mode and refit_each_month disagree")
    if refit_each_month is not None and from_mode is None:
        return bool(refit_each_month)
    return bool(from_mode)


def _check_window(start: int, end: int, label: str) -> tuple[int, int]:
    start_month = validate_yyyymm(start)
    end_month = validate_yyyymm(end)
    if end_month <= start_month:
        raise ValueError(f"{label} end month must be after the start month")
    return start_month, end_month


def _safe_segment(name: object) -> str:
    cleaned = _SEGMENT.sub("_", str(name)).strip("._")
    return cleaned or "series"


def _select_on_window(
    frame: pd.DataFrame,
    candidates: list[str],
    forecast_kwargs: dict,
    *,
    shuffle_candidates: bool,
    backward_prune: bool,
    random_state: int,
) -> tuple[SelectionOutcome, ForecastResult]:
    def evaluate(numeric: list[str], categorical: list[str], feature_no: int) -> float:
        result = forecast_panel(
            frame,
            numeric,
            categorical,
            feature_no=feature_no,
            **forecast_kwargs,
        )
        logger.info(
            "selection feature_no=%s acc=%s numeric=%s categorical=%s",
            feature_no,
            None if not np.isfinite(result.accuracy) else round(result.accuracy, 4),
            numeric,
            categorical,
        )
        return result.accuracy

    outcome = greedy_select(
        candidates,
        evaluate,
        shuffle_candidates=shuffle_candidates,
        backward_prune=backward_prune,
        random_state=random_state,
    )
    # Rescore the kept set. The last forward trial may have been rejected,
    # and a backward pass may have dropped features after that trial.
    final = forecast_panel(
        frame,
        outcome.numeric_features,
        outcome.categorical_features,
        feature_no=len(outcome.trace),
        **forecast_kwargs,
    )
    return outcome, final


def run_pipeline(
    data: pd.DataFrame | str | Path,
    *,
    mode: str | None = None,
    refit_each_month: bool | None = None,
    selection_start_month: int,
    selection_end_month: int,
    holdout_start_month: int,
    holdout_end_month: int,
    candidates: list[str] | None = None,
    shuffle_candidates: bool = False,
    backward_prune: bool = False,
    offset: int = 4,
    score_offset: int = 3,
    n_estimators: int = 50,
    n_jobs: int = -1,
    random_state: int = 2,
    demand_lag: int = 9,
    output_root: str | Path | None = None,
) -> PipelineResult:
    """Select features per series, then score the frozen set on the holdout.

    ``mode="SI"`` fits once per planning month. ``mode="MI"`` (or
    ``refit_each_month=True``) refits every target month. The default model
    uses ``n_jobs=-1``.

    When ``output_root`` is set, each series writes a selection trace, the
    validation and holdout predictions, and the holdout feature importances.
    Importances are averaged across the holdout fits — the models behind the
    reported accuracy — not discarded after fitting.
    """
    refit = resolve_refit(mode, refit_each_month)
    resolved_mode = "MI" if refit else "SI"
    selection_start, selection_end = _check_window(
        selection_start_month, selection_end_month, "selection"
    )
    holdout_start, holdout_end = _check_window(holdout_start_month, holdout_end_month, "holdout")
    if holdout_start < selection_end:
        raise ValueError(
            "holdout_start_month must be >= selection_end_month so the "
            "reported window is not used to choose features."
        )

    panel = load_panel(data) if not isinstance(data, pd.DataFrame) else data.copy()
    if "year_month" not in panel.columns:
        raise KeyError("Panel is missing year_month")
    panel["year_month"] = panel["year_month"].astype(int)
    if "series_id" not in panel.columns:
        raise KeyError("Panel is missing series_id")

    feature_list = list(DEFAULT_FEATURES if candidates is None else candidates)
    forecast_kwargs = {
        "refit_each_month": refit,
        "offset": offset,
        "score_offset": score_offset,
        "n_estimators": n_estimators,
        "n_jobs": n_jobs,
        "random_state": random_state,
        "demand_lag": demand_lag,
    }
    selection_kwargs = {
        **forecast_kwargs,
        "window_start": selection_start,
        "window_end": selection_end,
    }
    holdout_kwargs = {
        **forecast_kwargs,
        "window_start": holdout_start,
        "window_end": holdout_end,
    }

    series_results: list[SeriesResult] = []
    for series_id, frame in panel.groupby("series_id", sort=True):
        region = str(frame["region"].iloc[0]) if "region" in frame.columns else str(series_id)
        # Cut the future off so an offset that would land in the holdout has
        # no actuals to score (and no future actuals to lag).
        selection_frame = frame[frame["year_month"] < selection_end].copy()
        outcome, validation = _select_on_window(
            selection_frame,
            feature_list,
            selection_kwargs,
            shuffle_candidates=shuffle_candidates,
            backward_prune=backward_prune,
            random_state=random_state,
        )
        if not validation.predictions.empty:
            latest = int(validation.predictions["target_month"].max())
            if latest >= selection_end:
                raise RuntimeError(
                    "Selection scored target_month "
                    f"{latest}, which is outside the selection window ending {selection_end}."
                )

        holdout_frame = frame[frame["year_month"] < holdout_end].copy()
        holdout = forecast_panel(
            holdout_frame,
            outcome.numeric_features,
            outcome.categorical_features,
            **holdout_kwargs,
        )
        logger.info(
            "%s %s validation_acc=%s holdout_acc=%s features=%s",
            resolved_mode,
            region,
            None if not np.isfinite(validation.accuracy) else round(validation.accuracy, 4),
            None if not np.isfinite(holdout.accuracy) else round(holdout.accuracy, 4),
            outcome.numeric_features + outcome.categorical_features,
        )
        series_results.append(
            SeriesResult(
                series_id=series_id,
                region=region,
                numeric_features=list(outcome.numeric_features),
                categorical_features=list(outcome.categorical_features),
                validation_accuracy=float(validation.accuracy),
                holdout_accuracy=float(holdout.accuracy),
                selection_trace=outcome.trace,
                validation_predictions=validation.predictions,
                holdout_predictions=holdout.predictions,
                feature_importances=holdout.feature_importances,
                holdout_fits=holdout.n_fits,
            )
        )

    result = PipelineResult(mode=resolved_mode, refit_each_month=refit, series=series_results)
    if output_root is not None:
        _write_outputs(result, Path(output_root))
    return result


def _write_outputs(result: PipelineResult, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    result.summary().to_csv(output_root / "summary.csv", index=False)
    for item in result.series:
        segment = _safe_segment(f"{item.region}_{item.series_id}")
        path = output_root / segment
        path.mkdir(parents=True, exist_ok=True)
        item.selection_trace.to_csv(path / "selection_trace.csv", index=False)
        item.validation_predictions.to_csv(path / "validation_predictions.csv", index=False)
        item.holdout_predictions.to_csv(path / "holdout_predictions.csv", index=False)
        item.feature_importances.to_csv(path / "feature_importances.csv", index=False)
