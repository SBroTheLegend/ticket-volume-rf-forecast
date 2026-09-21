"""
Illustrative excerpt (~60 lines) — planning month + demand lags + RF predict.
Not the full model_manager / SI-MI wrappers. Synthetic-demo friendly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def add_demand_lags(df: pd.DataFrame, target: str = "ticket_volume", max_lag: int = 3) -> pd.DataFrame:
    out = df.sort_values(["series_id", "year_month"]).copy()
    for lag in range(1, max_lag + 1):
        out[f"demand_lag_{lag}"] = (
            out.groupby("series_id")[target].shift(lag).fillna(0)
        )
    return out


def plan_one_month(
    panel: pd.DataFrame,
    planning_month: int,
    feature_cols: list[str],
    target: str = "ticket_volume",
    horizon: int = 3,
) -> pd.DataFrame:
    """Fit once on history before planning_month; walk `horizon` months ahead."""
    df = add_demand_lags(panel, target=target)
    train = df[df["year_month"] < planning_month]
    model = RandomForestRegressor(n_estimators=40, random_state=2)
    model.fit(train[feature_cols], train[target])

    working = df.copy()
    rows = []
    y, m = divmod(planning_month, 100)
    for offset in range(horizon):
        mm = m + offset
        yy = y
        while mm > 12:
            mm -= 12
            yy += 1
        target_month = yy * 100 + mm

        # refresh lag-1 from (possibly forecast) history
        working = add_demand_lags(working, target=target)
        test = working[working["year_month"] == target_month]
        if test.empty:
            continue
        pred = np.clip(model.predict(test[feature_cols]), 0, None).round()
        working.loc[test.index, target] = pred
        rows.append(
            pd.DataFrame(
                {
                    "series_id": test["series_id"].values,
                    "planning_month": planning_month,
                    "target_month": target_month,
                    "offset": offset,
                    "fcst": pred,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


if __name__ == "__main__":
    # Tiny inline synthetic check
    demo = pd.DataFrame(
        {
            "series_id": [1] * 8,
            "year_month": [202001 + i for i in range(8)],
            "ticket_volume": [100, 110, 105, 120, 115, 130, 125, 140],
        }
    )
    feats = ["demand_lag_1", "demand_lag_2"]
    demo = add_demand_lags(demo)
    print(plan_one_month(demo, planning_month=202005, feature_cols=feats, horizon=2))
