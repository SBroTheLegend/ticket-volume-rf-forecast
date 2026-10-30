"""Feature engineering: alignment, finite ratios, static vs dynamic columns."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ticket_volume_rf.calendar import advance_yyyymm, validate_yyyymm
from ticket_volume_rf.features import (
    engineer_snapshot,
    safe_divide,
    sanitize_columns,
)
from ticket_volume_rf.io import load_panel
from ticket_volume_rf.metrics import capped_accuracy
from ticket_volume_rf.synthetic import make_synthetic_panel

ROOT = Path(__file__).resolve().parents[1]


def test_invalid_yyyymm_is_rejected():
    with pytest.raises(ValueError, match="201924"):
        validate_yyyymm(201924)
    with pytest.raises(ValueError, match="201924"):
        advance_yyyymm(201924, 1)
    assert validate_yyyymm(201912) == 201912
    assert advance_yyyymm(201912, 1) == 202001
    assert advance_yyyymm(201901, 0) == 201901


def test_package_has_no_invalid_month_default_or_regex_package():
    import inspect

    from ticket_volume_rf.forecast import forecast_panel
    from ticket_volume_rf.pipeline import run_pipeline

    for fn in (forecast_panel, run_pipeline):
        for param in inspect.signature(fn).parameters.values():
            assert param.default != 201924
    for path in (ROOT / "ticket_volume_rf").rglob("*.py"):
        text = path.read_text()
        assert "import regex" not in text


def test_rolling_features_align_when_rows_are_shuffled():
    frame = pd.DataFrame(
        {
            "series_id": [1, 1, 1, 2, 2, 2],
            "year_month": [202001, 202002, 202003, 202001, 202002, 202003],
            "ticket_volume": [10.0, 20.0, 30.0, 1.0, 2.0, 3.0],
        }
    )
    shuffled = frame.sample(frac=1, random_state=1)
    out = engineer_snapshot(shuffled, demand_lag=3)

    series_one = out[(out["series_id"] == 1) & (out["year_month"] == 202003)].iloc[0]
    series_two = out[(out["series_id"] == 2) & (out["year_month"] == 202003)].iloc[0]
    # Window 2, shift 1: mean of the two preceding points.
    assert series_one["moving_avg_2"] == pytest.approx(15.0)
    assert series_two["moving_avg_2"] == pytest.approx(1.5)
    # Must not pick up the other series, and must not be all-missing.
    assert series_one["moving_avg_3"] == pytest.approx(15.0)
    assert np.isfinite(out["quarterly_demand_std"]).all()


def test_change_features_stay_finite_when_the_baseline_is_zero():
    frame = pd.DataFrame(
        {
            "series_id": [1, 1, 1, 1],
            "year_month": [202001, 202002, 202003, 202004],
            "ticket_volume": [0.0, 0.0, 0.0, 8.0],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = engineer_snapshot(frame, demand_lag=3)
        ratios = safe_divide(pd.Series([1.0, 0.0, 5.0]), pd.Series([0.0, 0.0, 2.0]))
    for column in ("yearly_change", "quaterly_change", "recent_change"):
        assert np.isfinite(out[column]).all()
    assert ratios.tolist() == pytest.approx([0.0, 0.0, 2.5])


def test_calendar_and_ratio_lags_ignore_the_target():
    panel = make_synthetic_panel(n_months=8, regions=("Region_APAC",), seed=1)
    baseline = engineer_snapshot(panel, demand_lag=4)
    scaled = panel.copy()
    scaled["ticket_volume"] = scaled["ticket_volume"] * 10
    changed = engineer_snapshot(scaled, demand_lag=4)
    pd.testing.assert_series_equal(baseline["sin365"], changed["sin365"])
    pd.testing.assert_series_equal(baseline["demandratio_lag_1"], changed["demandratio_lag_1"])
    assert not np.allclose(baseline["moving_avg_3"], changed["moving_avg_3"])


def test_capped_accuracy_matches_the_original_score():
    acc = capped_accuracy([100.0, 100.0, 0.0], [80.0, 250.0, 5.0])
    assert acc[0] == pytest.approx(0.8)
    assert acc[1] == pytest.approx(0.0)
    assert np.isnan(acc[2])


def test_sanitize_columns_and_sample_csv_roundtrip():
    assert sanitize_columns(["plain", "a[b]", "x<y"]) == ["plain", "a_b_", "x_y"]
    panel = load_panel(ROOT / "examples" / "sample_series.csv")
    assert {"series_id", "region", "ticket_volume", "year_month"} <= set(panel.columns)
    snapshot = engineer_snapshot(panel, demand_lag=4)
    assert snapshot["demand_lag_1"].notna().all()
    assert len(snapshot) == len(panel)


def test_input_panel_is_not_mutated_by_engineering():
    panel = make_synthetic_panel(n_months=6, regions=("Region_EU",), seed=2)
    before = panel.copy()
    engineer_snapshot(panel)
    pd.testing.assert_frame_equal(panel, before)
