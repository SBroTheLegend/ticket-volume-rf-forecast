"""Planning loop: forecast write-back, SI vs MI, and the holdout split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ticket_volume_rf.calendar import advance_yyyymm
from ticket_volume_rf.forecast import default_regressor, forecast_panel
from ticket_volume_rf.pipeline import run_pipeline
from ticket_volume_rf.synthetic import make_synthetic_panel


class _CaptureModel:
    """Stand-in forest that records every fit and predicts a constant."""

    def __init__(self, value: float = 7.0):
        self.value = value
        self.fits: list[tuple[pd.DataFrame, np.ndarray]] = []
        self.predicts: list[pd.DataFrame] = []

    def fit(self, matrix, target):
        self.fits.append((matrix.copy(), np.asarray(target, dtype=float)))
        return self

    def predict(self, matrix):
        self.predicts.append(matrix.copy())
        return np.full(len(matrix), self.value, dtype=float)


def _known_panel() -> pd.DataFrame:
    months = [advance_yyyymm(202001, step) for step in range(8)]
    return pd.DataFrame(
        {
            "series_id": [1] * len(months),
            "region": ["Region_APAC"] * len(months),
            "year_month": months,
            "ticket_volume": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        }
    )


def test_horizon_lag_uses_the_forecast_not_the_future_actual():
    panel = _known_panel()
    model = _CaptureModel(7.0)
    result = forecast_panel(
        panel,
        predictors=["demand_lag_1"],
        window_start=202003,
        window_end=202005,
        refit_each_month=False,
        offset=2,
        score_offset=1,
        model=model,
        demand_lag=3,
    )
    # One planning month (202004), one SI fit, two offsets.
    assert result.n_fits == 1
    assert len(model.predicts) == 2
    assert model.predicts[0]["demand_lag_1"].iloc[0] == pytest.approx(30.0)
    # 202005 must see the forecast written for 202004 (7), not the actual 40.
    assert model.predicts[1]["demand_lag_1"].iloc[0] == pytest.approx(7.0)
    assert np.allclose(model.fits[0][1], [10.0, 20.0, 30.0])
    pd.testing.assert_series_equal(
        panel["ticket_volume"],
        pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], name="ticket_volume"),
    )


def test_mi_refits_each_target_month_and_trains_on_the_written_forecast():
    panel = _known_panel()
    model = _CaptureModel(7.0)
    result = forecast_panel(
        panel,
        predictors=["demand_lag_1"],
        window_start=202003,
        window_end=202005,
        refit_each_month=True,
        offset=2,
        score_offset=1,
        model=model,
        demand_lag=3,
    )
    assert result.n_fits == 2
    assert len(model.fits[1][1]) == 4
    assert model.fits[1][1][-1] == pytest.approx(7.0)
    assert model.predicts[1]["demand_lag_1"].iloc[0] == pytest.approx(7.0)
    assert result.refit_each_month is True


def test_default_regressor_uses_all_cores_and_importances_are_returned():
    model = default_regressor()
    assert model.n_jobs == -1
    panel = make_synthetic_panel(n_months=16, regions=("Region_APAC",), seed=0)
    result = forecast_panel(
        panel,
        predictors=["demand_lag_1", "moving_avg_2"],
        window_start=201806,
        window_end=201810,
        offset=2,
        score_offset=1,
        n_estimators=8,
        n_jobs=-1,
        random_state=2,
        demand_lag=4,
    )
    assert result.n_fits > 0
    assert not result.feature_importances.empty
    assert result.feature_importances["importance"].sum() == pytest.approx(1.0)
    assert set(result.feature_importances["feature"]) == {"demand_lag_1", "moving_avg_2"}


def test_same_seed_replays_forecasts():
    panel = make_synthetic_panel(n_months=14, regions=("Region_EU",), seed=3)
    kwargs = dict(
        predictors=["demand_lag_1", "sin365"],
        window_start=201806,
        window_end=201809,
        offset=2,
        score_offset=0,
        n_estimators=6,
        n_jobs=1,
        random_state=2,
        demand_lag=3,
    )
    panel["Date"] = pd.to_datetime(panel["Date"])
    first = forecast_panel(panel, **kwargs)
    second = forecast_panel(panel, **kwargs)
    pd.testing.assert_series_equal(first.predictions["Fcst"], second.predictions["Fcst"])


def test_selection_does_not_see_holdout_and_scrambling_holdout_does_not_change_it(tmp_path):
    panel = make_synthetic_panel(n_months=24, regions=("Region_APAC",), seed=4)
    kwargs = dict(
        mode="SI",
        selection_start_month=201806,
        selection_end_month=201811,
        holdout_start_month=201811,
        holdout_end_month=201902,
        candidates=["demand_lag_2", "moving_avg_2", "Jan_flag"],
        backward_prune=True,
        offset=2,
        score_offset=1,
        n_estimators=8,
        n_jobs=1,
        random_state=2,
        demand_lag=4,
        output_root=tmp_path,
    )
    original = run_pipeline(panel, **kwargs)
    scrambled = panel.copy()
    scrambled.loc[scrambled["year_month"] > 201811, "ticket_volume"] *= 5
    # Same selection window; do not rewrite files.
    kwargs.pop("output_root")
    altered = run_pipeline(scrambled, **kwargs)

    assert len(original.series) == 1
    series = original.series[0]
    assert series.numeric_features == altered.series[0].numeric_features
    assert series.categorical_features == altered.series[0].categorical_features
    assert int(series.validation_predictions["target_month"].max()) < 201811
    assert int(series.holdout_predictions["planning_month"].min()) > 201811
    assert int(series.holdout_predictions["planning_month"].max()) < 201902
    # Reported actuals are the holdout actuals, so scrambling them changes the score.
    assert series.holdout_predictions["Actual"].mean() != pytest.approx(
        altered.series[0].holdout_predictions["Actual"].mean()
    )
    assert series.holdout_accuracy != pytest.approx(altered.series[0].holdout_accuracy)

    importance_path = tmp_path / "Region_APAC_1" / "feature_importances.csv"
    assert importance_path.exists()
    saved = pd.read_csv(importance_path)
    assert "importance" in saved.columns
    assert not saved.empty
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert summary.loc[0, "mode"] == "SI"
    assert "validation_acc" in summary.columns
    assert "holdout_acc" in summary.columns


def test_mi_pipeline_fits_more_often_than_si_and_rejects_a_bad_split():
    panel = make_synthetic_panel(n_months=18, regions=("Region_EU",), seed=5)
    common = dict(
        selection_start_month=201806,
        selection_end_month=201810,
        holdout_start_month=201810,
        holdout_end_month=201901,
        candidates=["demand_lag_2"],
        offset=2,
        score_offset=1,
        n_estimators=5,
        n_jobs=1,
        random_state=2,
        demand_lag=3,
    )
    si = run_pipeline(panel, mode="SI", **common)
    mi = run_pipeline(panel, refit_each_month=True, **common)
    assert si.mode == "SI" and si.refit_each_month is False
    assert mi.mode == "MI" and mi.refit_each_month is True
    assert mi.series[0].holdout_predictions["Offset"].nunique() >= 1
    # MI writes a prediction per fit; SI fits once per planning month.
    assert mi.series[0].holdout_fits > si.series[0].holdout_fits
    assert mi.series[0].holdout_fits >= len(mi.series[0].holdout_predictions)

    with pytest.raises(ValueError, match="holdout_start_month"):
        run_pipeline(
            panel,
            mode="SI",
            selection_start_month=201806,
            selection_end_month=201810,
            holdout_start_month=201809,
            holdout_end_month=201901,
            candidates=["demand_lag_2"],
            offset=2,
            score_offset=1,
            n_estimators=4,
            n_jobs=1,
            demand_lag=3,
        )


def test_forecast_rejects_the_old_invalid_end_month():
    panel = _known_panel()
    with pytest.raises(ValueError, match="201924"):
        forecast_panel(
            panel,
            window_start=201901,
            window_end=201924,
            offset=2,
            score_offset=1,
            model=_CaptureModel(),
            demand_lag=2,
        )
