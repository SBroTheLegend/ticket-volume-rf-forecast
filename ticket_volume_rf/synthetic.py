"""Synthetic multi-region monthly demand. No real tickets, names, or paths."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ticket_volume_rf.calendar import advance_yyyymm, validate_yyyymm


def make_synthetic_panel(
    *,
    start_year: int = 2018,
    start_month: int = 1,
    n_months: int = 36,
    regions: tuple[str, ...] = ("Region_APAC", "Region_EU"),
    seed: int = 0,
) -> pd.DataFrame:
    """Build a seasonal panel long enough for a validation window and a holdout.

    Volumes are a region level, a yearly wave, a small trend, and Gaussian
    noise, clipped to stay positive. Calendar flags and ratio columns are
    stand-ins so the original candidate catalog has something to select.
    """
    if n_months < 2:
        raise ValueError("n_months must be at least 2")
    start = validate_yyyymm(start_year * 100 + start_month)
    year_months = [advance_yyyymm(start, offset) for offset in range(n_months)]
    months = np.array([value % 100 for value in year_months], dtype=int)
    years = np.array([value // 100 for value in year_months], dtype=int)
    dates = [f"{year:04d}-{month:02d}-15" for year, month in zip(years, months)]

    rng = np.random.default_rng(seed)
    frames = []
    for index, region in enumerate(regions):
        seasonal = 15.0 * np.sin(2.0 * np.pi * (months - 1) / 12.0)
        trend = 0.35 * np.arange(n_months)
        noise = rng.normal(0.0, 3.0, size=n_months)
        volume = np.clip(80.0 + 25.0 * index + seasonal + trend + noise, 1.0, None)
        ratio = 1.0 + 0.08 * np.cos(2.0 * np.pi * (months - 1) / 12.0)
        frames.append(
            pd.DataFrame(
                {
                    "series_id": index + 1,
                    "region": region,
                    "year_month": year_months,
                    "Date": dates,
                    "ticket_volume": np.round(volume, 2),
                    "seasonality_num": months,
                    "Jan_flag": (months == 1).astype(int),
                    "Dec_flag": (months == 12).astype(int),
                    "Peak_flag": np.isin(months, (4, 7, 11, 12)).astype(int),
                    "Year": years,
                    "yearly_seasonality": np.sin(2.0 * np.pi * months / 12.0),
                    "Quarterly_Ratio_Jan": np.round(ratio, 4),
                    "Quarterly_Ratio_seasonality": np.round(
                        1.0 + 0.1 * np.cos(2.0 * np.pi * months / 3.0), 4
                    ),
                    "Ratio_Jan": np.where(months == 1, 1.0, 0.9),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
