#!/usr/bin/env python3
"""
Short synthetic demo — simplified API, not the full SI/MI model_manager.

Demonstrates: lag features + RandomForestRegressor on a tiny multi-region panel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

HERE = Path(__file__).resolve().parent
DATA = HERE / "sample_series.csv"


def add_lags(df: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    out = df.sort_values(["series_id", "year_month"]).copy()
    for lag in range(1, max_lag + 1):
        out[f"demand_lag_{lag}"] = out.groupby("series_id")["ticket_volume"].shift(lag)
    return out.fillna(0)


def demo_holdout(df: pd.DataFrame, holdout_from: int = 202101) -> None:
    df = add_lags(df)
    feature_cols = [c for c in df.columns if c.startswith("demand_lag_")]
    rows = []
    for sid, g in df.groupby("series_id"):
        train = g[g["year_month"] < holdout_from]
        test = g[g["year_month"] >= holdout_from]
        if train.empty or test.empty:
            continue
        model = RandomForestRegressor(n_estimators=50, random_state=2)
        model.fit(train[feature_cols], train["ticket_volume"])
        pred = np.clip(model.predict(test[feature_cols]), 0, None)
        mae = mean_absolute_error(test["ticket_volume"], pred)
        rows.append(
            {
                "series_id": sid,
                "region": g["region"].iloc[0],
                "holdout_n": len(test),
                "mae": round(float(mae), 2),
                "actual_mean": round(float(test["ticket_volume"].mean()), 1),
                "pred_mean": round(float(pred.mean()), 1),
            }
        )
    summary = pd.DataFrame(rows)
    print("=== Synthetic ticket-volume RF demo (simplified) ===")
    print(summary.to_string(index=False))
    print("\nNote: full stepwise SI/MI feature-selection modules are private;")
    print("see ../ACCESS.md to request the sanitized full source.")


if __name__ == "__main__":
    panel = pd.read_csv(DATA)
    demo_holdout(panel)
