#!/usr/bin/env python3
"""Run SI and MI stepwise selection on a synthetic multi-region panel.

The sample CSV in this folder is a schema sketch (15 months). The validation
window plus the holdout need a longer history, so this demo builds one with
``make_synthetic_panel``. Nothing here is a real ticket series.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ticket_volume_rf import make_synthetic_panel, run_pipeline


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    panel = make_synthetic_panel(n_months=30, seed=7)
    windows = dict(
        selection_start_month=201906,
        selection_end_month=202001,
        holdout_start_month=202001,
        holdout_end_month=202006,
        candidates=["demand_lag_2", "moving_avg_3", "sin365", "recent_change", "Jan_flag"],
        backward_prune=True,
        offset=4,
        score_offset=3,
        random_state=2,
        n_jobs=-1,
        demand_lag=6,
    )
    print("Synthetic panel rows:", len(panel), "regions:", sorted(panel["region"].unique()))
    print(
        "Selection planning months are inside (201906, 202001); "
        "holdout planning months are inside (202001, 202006)."
    )
    for mode, trees in (("SI", 20), ("MI", 12)):
        result = run_pipeline(
            panel,
            mode=mode,
            n_estimators=trees,
            output_root=ROOT / "outputs" / "demo" / mode,
            **windows,
        )
        print(f"\n=== {mode} (refit_each_month={result.refit_each_month}) ===")
        print(result.summary().to_string(index=False))
        top = result.series[0].feature_importances.head(5)
        print(f"Top holdout importances, {result.series[0].region}:")
        print(top.to_string(index=False))
    print("\nWrote selection traces, predictions, and importances under outputs/demo/.")


if __name__ == "__main__":
    main()
