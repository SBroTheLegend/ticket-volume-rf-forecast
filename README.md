# Ticket-volume RF forecast (portfolio glance)

**Generalized portfolio illustration** of stepwise Random Forest feature
selection for **multi-region monthly ticket / demand forecasting**.

> Original employer project details and proprietary data are **excluded**.
> The runnable code in this repository uses **synthetic panels only**.

Author: [Sourav Kumar Kashyap](https://github.com/SBroTheLegend) ·
[smartagent88@gmail.com](mailto:smartagent88@gmail.com)

---

## Problem (generalized)

Operations and planning teams need a forward view of **monthly ticket volume**
(or analogous demand) across regions. A useful model should:

1. Respect hierarchy (one series per region).
2. Use lag / seasonality / rolling statistics without leaking future actuals.
3. Select features **stepwise** so each region keeps only what improves accuracy.
4. Support a **planning loop**: freeze a planning month, then forecast several offset months ahead.

## Approach

| Piece | Role |
|---|---|
| `RandomForestRegressor` | Non-linear regressor on lag + engineered features (`n_jobs=-1` by default) |
| Lag / rolling / Fourier features | Demand memory and seasonality |
| Optional categorical dummies | Calendar flags (Jan / Dec / peak) |
| **Single-Iteration (SI)** | Fit once per planning month, then walk offsets |
| **Multiple-Iteration (MI)** | Re-fit each target month inside the offset window |
| `greedy_select` | Forward selection, optional shuffle, optional backward prune |

The SI and MI loops are one module, `ticket_volume_rf`, switched with
`mode="SI"|"MI"` or `refit_each_month=True|False`.

## Single-Iteration vs Multiple-Iteration

Both modes share the same horizon. For each planning month they restore
actuals, build features from history, forecast `offset` months ahead
(default 4), and **write each forecast back** into `ticket_volume`. The next
offset's demand lags and rolling statistics therefore see that forecast, not
the future actual. Calendar Fourier terms and lags of exogenous ratios do
not depend on the write-back, so they are computed once per fit cycle.

| | SI (`refit_each_month=False`) | MI (`refit_each_month=True`) |
|---|---|---|
| When the forest is fit | Once, on `year_month < planning_month` | Again before every target month, on `year_month < target_month` |
| In-horizon forecasts | Change later lags and rolling features only | Also change the training rows of the next refit |
| Cost | One fit per planning month | One fit per planning month × offset |
| Typical use | Freeze a plan, then roll the horizon | Let the horizon adapt as imputed history updates |

```python
from ticket_volume_rf import make_synthetic_panel, run_pipeline

panel = make_synthetic_panel(n_months=30, seed=7)
result = run_pipeline(
    panel,
    mode="SI",  # or mode="MI" / refit_each_month=True
    selection_start_month=201906,
    selection_end_month=202001,
    holdout_start_month=202001,
    holdout_end_month=202006,
    candidates=["demand_lag_2", "moving_avg_3", "sin365", "recent_change"],
    backward_prune=True,
)
print(result.summary())
```

## Validation design

Feature selection and the published accuracy are **not** measured on the same
months. Using one window for both lets the selector keep whatever happened to
fit those months, so the reported accuracy is optimistic.

| Window | Planning months (both ends exclusive) | Allowed to |
|---|---|---|
| Selection | `(selection_start_month, selection_end_month)` | Add or drop features |
| Holdout | `(holdout_start_month, holdout_end_month)` | Score the frozen feature set |

`holdout_start_month` must be **≥** `selection_end_month`. During selection
the series is cut at `selection_end_month`, so a long offset cannot score
holdout actuals or read them as lags. Example: selection `(201906, 202001)`
chooses features using planning months 201907–201912, and only when
`planning_month + score_offset` is still inside that cut. Holdout
`(202001, 202006)` then reports accuracy for the later planning months.

The holdout pass may **train** on earlier actuals, including the selection
period. It does not choose features. The number that goes in the summary as
`holdout_acc` is that second pass. `validation_acc` is what the selector saw.

**Score.** Capped accuracy at `score_offset` (default 3, so the horizon must
be at least 4 months), averaged over the rows at that offset:

`1 - min(actual, |actual - forecast|) / actual`

Zero actuals are skipped. When the absolute error is larger than the actual,
accuracy is 0 rather than negative. This is the same offset score the
original stepwise loop reported.

**Change features.** `yearly_change`, `quaterly_change` (historical spelling),
and `recent_change` are ratios. A zero baseline is stored as 0. The ratio is
never left as `inf` and then overwritten.

**Order of the greedy walk.** Forward selection tries the candidate list from
top to bottom and keeps a feature only when validation accuracy **strictly**
improves, starting from `demand_lag_1`. That path depends on order.

* `shuffle_candidates=True` permutes the list with `random_state` before the forward pass.
* `backward_prune=True` then drops any selected feature, other than the seed, whose removal does not lower validation accuracy.

**Importances.** Averaged `feature_importances_` from the holdout fits are
returned on each series and written next to the predictions. They are the
importances of the model that produced the reported accuracy.

## Quick synthetic demo

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python examples/run_demo.py
pytest
```

`examples/run_demo.py` builds a 30-month synthetic panel (two regions), runs
SI and MI with a short candidate list, prints validation vs holdout accuracy,
and writes traces under `outputs/demo/` (gitignored).

`examples/sample_series.csv` is a tiny schema sketch, not a series long
enough for the two windows.

## Layout

| Path | What it is |
|---|---|
| `ticket_volume_rf/` | Planning loop, selection, synthetic panel |
| `examples/run_demo.py` | SI and MI on synthetic data |
| `examples/sample_series.csv` | Tiny synthetic schema sample |
| `snippets/forecasting_core_excerpt.py` | Short excerpt of one planning month |
| `tests/` | Synthetic-data tests for alignment, leakage, and the two windows |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the planning-loop diagram.
Contact: [ACCESS.md](ACCESS.md).

## Disclaimer

Synthetic demonstration and methodology overview only. No employer customer
data, no proprietary notebooks with outputs, and no claim of clearance to
publish production IP. Suitable for recruiter review of **approach and
communication**.
