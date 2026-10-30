"""Greedy forward feature selection with an optional backward prune.

The walk starts from ``demand_lag_1`` and tries each candidate once. A
feature is kept only when validation accuracy strictly improves, which is
the original ``model_manager`` rule. That path depends on list order.

``shuffle_candidates`` draws a seeded permutation before the forward pass.
``backward_prune`` then drops any selected feature, other than the seed,
whose removal does not lower validation accuracy. Together they take the
dead-end features an unlucky order would otherwise freeze in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

# Candidates the original stepwise loop tried, in its original order.
# ``quaterly_change`` keeps the historical spelling on purpose.
DEFAULT_FEATURES: list[str] = [
    "demand_lag_2",
    "demand_lag_3",
    "demand_lag_4",
    "demand_lag_5",
    "demand_lag_6",
    "demand_lag_7",
    "demand_lag_8",
    "demand_lag_9",
    "yearly_demand_std",
    "quarterly_demand_std",
    "half_yearly_demand_std",
    "yearly_demand_avg",
    "quarterly_demand_avg",
    "half_yearly_demand_avg",
    "yearly_change",
    "quaterly_change",
    "recent_change",
    "moving_avg_2",
    "moving_avg_3",
    "moving_avg_2_lag2",
    "moving_avg_3_lag2",
    "moving_avg_2_lag3",
    "moving_avg_3_lag3",
    "seasonality_num",
    "Year",
    "yearly_seasonality",
    "sin365",
    "cos365",
    "sin365_2",
    "cos365_2",
    "demandratio_lag_1",
    "demandratio_lag_2",
    "demandratio_lag_3",
    "demandratio_lag_4",
    "demandratio_lag_5",
    "demandratio_lag_6",
    "demandratio_lag_7",
    "demandratio_lag_8",
    "demandratio_lag_9",
    "sin365_lag1",
    "sin365_lag2",
    "sin365_lag3",
    "sin365_lag4",
    "sin365_lag5",
    "cos365_lag1",
    "cos365_lag2",
    "cos365_lag3",
    "cos365_lag4",
    "cos365_lag5",
    "Peak_flag",
    "Dec_flag",
    "Quarterly_Ratio_seasonality",
    "Ratio_Jan",
    "Jan_flag",
]

CATEGORICAL_FEATURES = {"seasonality_num", "Jan_flag", "Dec_flag", "Peak_flag"}

EvaluateFn = Callable[[list[str], list[str], int], float]


@dataclass
class SelectionOutcome:
    """Feature set chosen on the validation window, plus the trial trace."""

    numeric_features: list[str]
    categorical_features: list[str]
    validation_accuracy: float
    trace: pd.DataFrame = field(default_factory=pd.DataFrame)


def feature_kind(name: str) -> str:
    """Return ``Categorical`` for calendar flags, otherwise ``Numerical``."""
    if name in CATEGORICAL_FEATURES:
        return "Categorical"
    return "Numerical"


def build_feature_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Column catalog used by the original selector (numeric vs categorical)."""
    catalog = pd.DataFrame({"Features": list(df.columns), "Variable_type": "Numerical"})
    catalog["Variable_type"] = np.where(
        catalog["Features"].isin(CATEGORICAL_FEATURES),
        "Categorical",
        "Numerical",
    )
    return catalog


def _joined(values: list[str]) -> str:
    return ", ".join(values)


def greedy_select(
    candidates: list[str],
    evaluate: EvaluateFn,
    *,
    shuffle_candidates: bool = False,
    backward_prune: bool = False,
    random_state: int = 2,
    seed_numeric: list[str] | None = None,
    feature_kinds: Callable[[str], str] | None = None,
) -> SelectionOutcome:
    """Forward-select ``candidates``, then optionally prune.

    ``evaluate(numeric, categorical, feature_no)`` returns the validation
    accuracy of that set. It is called for the seed set, for every forward
    trial, and for every backward trial. Accuracy must be finite for the
    seed; later non-finite scores are treated as "do not keep".
    """
    kind_of = feature_kinds or feature_kind
    numeric = list(seed_numeric or ["demand_lag_1"])
    categorical: list[str] = []
    protected = set(numeric)

    seen = set(numeric)
    order: list[str] = []
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        order.append(name)
    if shuffle_candidates and order:
        rng = np.random.default_rng(random_state)
        order = [order[int(index)] for index in rng.permutation(len(order))]

    rows: list[dict] = []

    def _record(
        feature_no: int,
        tried: str,
        kind: str,
        feature_in_model: bool,
        acc: float,
        stage: str,
    ) -> None:
        rows.append(
            {
                "Feature_no": feature_no,
                "stage": stage,
                "tried_feature": tried,
                "kind": kind,
                "feature_in_model": feature_in_model,
                "Acc": acc,
                "Num_feat": _joined(numeric),
                "Cat_feat": _joined(categorical) if categorical else "No Cat Feat",
            }
        )

    baseline = float(evaluate(list(numeric), list(categorical), 0))
    _record(0, "", "seed", True, baseline, "seed")
    if not np.isfinite(baseline):
        raise RuntimeError(
            "The seed feature set has no finite validation accuracy. "
            "Check that the selection window contains planning months whose "
            "target month at score_offset is still inside the window, and "
            "that actuals are non-zero."
        )

    best = baseline
    feature_no = 0
    for name in order:
        feature_no += 1
        kind = kind_of(name)
        if kind == "Categorical":
            categorical.append(name)
        else:
            numeric.append(name)
        acc = float(evaluate(list(numeric), list(categorical), feature_no))
        kept = bool(np.isfinite(acc) and acc > best)
        if kept:
            best = acc
        elif kind == "Categorical":
            categorical.pop()
        else:
            numeric.pop()
        in_model = name in numeric or name in categorical
        _record(feature_no, name, kind, in_model, acc, "forward")

    if backward_prune:
        changed = True
        while changed:
            changed = False
            pool = [("Numerical", name) for name in numeric if name not in protected]
            pool += [("Categorical", name) for name in categorical]
            for kind, name in pool:
                feature_no += 1
                trial_numeric = [item for item in numeric if item != name]
                trial_categorical = [item for item in categorical if item != name]
                acc = float(evaluate(trial_numeric, trial_categorical, feature_no))
                drop = bool(np.isfinite(acc) and acc >= best)
                if drop:
                    numeric = trial_numeric
                    categorical = trial_categorical
                    best = acc
                in_model = name in numeric or name in categorical
                _record(feature_no, name, kind, in_model, acc, "backward")
                if drop:
                    changed = True
                    break

    trace = pd.DataFrame(rows)
    return SelectionOutcome(
        numeric_features=list(numeric),
        categorical_features=list(categorical),
        validation_accuracy=float(best),
        trace=trace,
    )
