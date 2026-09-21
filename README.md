# Ticket-volume RF forecast (portfolio glance)

**Generalized portfolio illustration** of stepwise Random Forest feature
selection for **multi-region monthly ticket / demand forecasting**.

> Original employer project details and proprietary data are **excluded**.
> This public repository is a **glance + synthetic demo only**.
> Full cleaned source is available on request (see [ACCESS.md](ACCESS.md)).

Author: [Sourav Kumar Kashyap](https://github.com/SBroTheLegend) ·
[smartagent88@gmail.com](mailto:smartagent88@gmail.com)

---

## Problem (generalized)

Operations and planning teams need a forward view of **monthly ticket volume**
(or analogous demand) across regions. A useful model should:

1. Respect hierarchy (one series per region).
2. Use lag / seasonality / rolling statistics without leaking future actuals.
3. Select features **stepwise** so each region keeps only what improves hold-out accuracy.
4. Support a **planning loop**: freeze a planning month, then forecast several offset months ahead.

## Approach (high level)

| Piece | Role |
|---|---|
| `RandomForestRegressor` | Non-linear regressor on lag + engineered features |
| Lag / rolling / Fourier features | Demand memory and seasonality |
| Optional categorical dummies | Calendar flags (e.g. Jan / Dec / peak) |
| **Single-Iteration (SI)** | Fit once per planning month, then walk offsets |
| **Multiple-Iteration (MI)** | Re-fit each target month inside the offset window |
| `model_manager` wrapper | Greedy forward feature selection by accuracy |

Public repo = glance + synthetic demo only. The full SI / MI modules
(employer-safe, path-sanitized) are **not** pushed here.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a diagram of the planning loop
and the SI vs MI fit timing.

## Quick synthetic demo

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python examples/run_demo.py
```

Uses `examples/sample_series.csv` (tiny **synthetic** panel only).

## Snippets

`snippets/forecasting_core_excerpt.py` — short excerpt of the planning /
lag idea (~60 lines). Not the full `model_manager`.

## Request full cleaned source

Email **smartagent88@gmail.com** — template in [ACCESS.md](ACCESS.md).

Mailto:
[Request full source](mailto:smartagent88@gmail.com?subject=Request%3A%20ticket-volume-rf-forecast%20full%20source&body=Hi%20Sourav%2C%0A%0AI%20reviewed%20https%3A%2F%2Fgithub.com%2FSBroTheLegend%2Fticket-volume-rf-forecast%20and%20would%20like%20the%20full%20sanitized%20SI%2FMI%20modules%20for%20recruiter%20%2F%20hiring-manager%20review.%0A%0AName%3A%0ACompany%3A%0ARole%3A%0A%0AThanks%21)

## Disclaimer

Synthetic demonstration and methodology overview only. No employer customer
data, no proprietary notebooks with outputs, and no claim of clearance to
publish production IP. Suitable for recruiter review of **approach and
communication**.
