"""Forward selection, order, and the backward prune. No forest fits."""

from __future__ import annotations

import pytest

from ticket_volume_rf.pipeline import resolve_refit
from ticket_volume_rf.selection import greedy_select


def test_forward_keeps_the_seed_when_nothing_helps():
    def evaluate(numeric, categorical, feature_no):
        return 0.3

    outcome = greedy_select(["noise", "also_noise"], evaluate)
    assert outcome.numeric_features == ["demand_lag_1"]
    assert outcome.categorical_features == []
    assert outcome.validation_accuracy == pytest.approx(0.3)
    tried = outcome.trace.loc[outcome.trace["stage"] == "forward", "feature_in_model"]
    assert not tried.any()


def test_backward_prune_drops_a_redundant_feature():
    def evaluate(numeric, categorical, feature_no):
        # ``redundant`` helps only while ``useful`` is absent. Forward order
        # [redundant, useful] therefore keeps both; the prune should drop
        # the one that no longer adds accuracy.
        if "useful" in numeric:
            return 0.6
        if "redundant" in numeric:
            return 0.45
        return 0.4

    outcome = greedy_select(
        ["redundant", "useful"],
        evaluate,
        backward_prune=True,
    )
    assert outcome.numeric_features == ["demand_lag_1", "useful"]
    assert outcome.validation_accuracy == pytest.approx(0.6)
    backward = outcome.trace[outcome.trace["stage"] == "backward"]
    dropped = backward.loc[backward["tried_feature"] == "redundant", "feature_in_model"]
    assert dropped.tolist() == [False]


def test_shuffle_is_seeded_and_changes_order():
    def _order(seed: int) -> list[str]:
        seen: list[str] = []

        def evaluate(numeric, categorical, feature_no):
            if feature_no > 0 and len(numeric) == 2:
                seen.append(numeric[-1])
            return 0.1

        greedy_select(
            ["a", "b", "c", "d"],
            evaluate,
            shuffle_candidates=True,
            random_state=seed,
        )
        return seen

    first = _order(0)
    assert _order(0) == first
    assert _order(1) != first
    assert sorted(first) == ["a", "b", "c", "d"]


def test_seed_score_must_be_finite():
    def evaluate(numeric, categorical, feature_no):
        return float("nan")

    with pytest.raises(RuntimeError, match="finite validation accuracy"):
        greedy_select(["a"], evaluate)


def test_mode_and_refit_flag_must_agree():
    assert resolve_refit(None, None) is False
    assert resolve_refit("SI", None) is False
    assert resolve_refit("MI", None) is True
    assert resolve_refit(None, True) is True
    assert resolve_refit("mi", True) is True
    with pytest.raises(ValueError, match="disagree"):
        resolve_refit("SI", True)
    with pytest.raises(ValueError, match="SI"):
        resolve_refit("monthly", None)
