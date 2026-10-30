"""Lag, rolling, change, and calendar features for one planning origin.

``GroupBy.apply`` + ``Series.rolling`` is intentionally not used. On current
pandas that ``apply`` returns a MultiIndex and assigning it back does not
line up with the frame (the new column becomes all missing). Rolling
statistics go through ``GroupBy.rolling``, which keeps the original index.

Calendar Fourier terms and lags of exogenous ratios do not depend on the
forecast write-back, so they are built once. Lags, rolling stats, and change
ratios are refreshed whenever in-horizon forecasts replace actuals.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

SERIES = "series_id"
TARGET = "ticket_volume"

# Feature names preserved from the stepwise catalog, including the original
# spelling of the quarterly change ratio.
ROLLING_SPECS: tuple[tuple[str, int, int, str], ...] = (
    ("yearly_demand_std", 12, 1, "std"),
    ("quarterly_demand_std", 3, 1, "std"),
    ("half_yearly_demand_std", 6, 1, "std"),
    ("yearly_demand_avg", 12, 1, "mean"),
    ("quarterly_demand_avg", 3, 1, "mean"),
    ("half_yearly_demand_avg", 6, 1, "mean"),
    ("moving_avg_2", 2, 1, "mean"),
    ("moving_avg_3", 3, 1, "mean"),
    ("moving_avg_2_lag2", 2, 2, "mean"),
    ("moving_avg_3_lag2", 3, 2, "mean"),
    ("moving_avg_2_lag3", 2, 3, "mean"),
    ("moving_avg_3_lag3", 3, 3, "mean"),
)

_COLUMN_SANITIZER = re.compile(r"\[|\]|<", re.IGNORECASE)


def sanitize_columns(columns) -> list[str]:
    """Replace ``[``, ``]``, and ``<`` so tree feature names stay plain text."""
    cleaned = []
    for col in columns:
        text = str(col)
        if any(token in text for token in ("[", "]", "<")):
            text = _COLUMN_SANITIZER.sub("_", text)
        cleaned.append(text)
    return cleaned


def safe_divide(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Divide and map a zero or missing denominator to 0.

    A zero baseline used to become ``inf`` and was then replaced with 0,
    which hides the bad ratio and raises a divide-by-zero warning. Zero
    denominators are masked before the division, so the result is finite.
    """
    aligned = denom.reindex(numer.index)
    usable = aligned.mask(aligned.eq(0) | aligned.isna())
    return numer.div(usable).fillna(0.0)


def _rolling_shift(df: pd.DataFrame, column: str, window: int, shift: int, how: str) -> pd.Series:
    """Rolling mean/std, shifted within each series, aligned to ``df.index``."""
    roller = df.groupby(SERIES, sort=False)[column].rolling(window=window, min_periods=1)
    stat = roller.mean() if how == "mean" else roller.std()
    shifted = stat.groupby(level=0).shift(shift)
    return shifted.reset_index(level=0, drop=True)


def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Add day-of-year Fourier terms. No-op when ``Date`` is absent."""
    if date_col not in df.columns:
        return df
    out = df.copy()
    dates = pd.to_datetime(out[date_col])
    # day_of_year is the non-deprecated name; dayofyear remains for pandas 1.5.
    try:
        day_of_year = dates.dt.day_of_year.to_numpy(dtype=float)
    except AttributeError:
        day_of_year = dates.dt.dayofyear.to_numpy(dtype=float)
    out["sin365"] = np.sin(2 * np.pi * day_of_year / 365.25)
    out["cos365"] = np.cos(2 * np.pi * day_of_year / 365.25)
    out["sin365_2"] = np.sin(4 * np.pi * day_of_year / 365.25)
    out["cos365_2"] = np.cos(4 * np.pi * day_of_year / 365.25)
    return out


def add_exogenous_lags(
    df: pd.DataFrame,
    demand_lag: int = 9,
    ratio_col: str = "Quarterly_Ratio_Jan",
) -> pd.DataFrame:
    """Lag an exogenous ratio. Skipped when the source column is absent.

    These lags do not read ``ticket_volume``, so forecast write-back must
    not recompute them.
    """
    if ratio_col not in df.columns:
        return df
    out = df.copy()
    grouped = out.groupby(SERIES, sort=False)[ratio_col]
    for lag in range(1, demand_lag + 1):
        out[f"demandratio_lag_{lag}"] = grouped.shift(lag).fillna(0.0)
    return out


def refresh_dynamic_features(
    df: pd.DataFrame,
    *,
    target: str = TARGET,
    demand_lag: int = 9,
) -> pd.DataFrame:
    """Recompute target-dependent features in place.

    ``df`` must already be sorted by ``series_id``, ``year_month`` and have
    a unique index. Call this again after writing forecasts into ``target``
    so the next offset sees those forecasts in its lags and rolling windows.
    """
    grouped = df.groupby(SERIES, sort=False)[target]
    for lag in range(1, demand_lag + 1):
        df[f"demand_lag_{lag}"] = grouped.shift(lag).fillna(0.0)

    for name, window, shift, how in ROLLING_SPECS:
        df[name] = _rolling_shift(df, target, window, shift, how).fillna(0.0)

    previous = grouped.shift(1)
    previous_2 = grouped.shift(2)
    df["yearly_change"] = safe_divide(previous, df["yearly_demand_avg"])
    df["quaterly_change"] = safe_divide(previous, df["quarterly_demand_avg"])
    df["recent_change"] = safe_divide(previous, previous_2)

    if "sin365" in df.columns:
        for lag in range(1, 6):
            lag_col = f"demand_lag_{lag}"
            if lag_col not in df.columns:
                continue
            df[f"sin365_lag{lag}"] = df["sin365"] * df[lag_col]
            df[f"cos365_lag{lag}"] = df["cos365"] * df[lag_col]
    return df


def encode_categoricals(
    df: pd.DataFrame,
    qual_preds: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categorical predictors with the first level dropped.

    Numeric codes are cast to int first so dummy names stay stable
    (``Jan_flag_1`` rather than ``Jan_flag_1.0``).
    """
    if not qual_preds:
        return df, []
    missing = [col for col in qual_preds if col not in df.columns]
    if missing:
        raise KeyError(f"Categorical features not on the panel: {missing}")

    out = df.copy()
    for col in qual_preds:
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().all():
            out[col] = numeric.astype(int)

    before = set(out.columns)
    out = pd.get_dummies(out, columns=list(qual_preds), drop_first=True)
    dummy_cols = sorted(col for col in out.columns if col not in before)
    rename = {}
    for col in dummy_cols:
        cleaned = sanitize_columns([col])[0]
        if cleaned != col:
            rename[col] = cleaned
    if rename:
        out = out.rename(columns=rename)
        dummy_cols = [rename.get(col, col) for col in dummy_cols]
    for col in dummy_cols:
        out[col] = out[col].astype("float64")
    return out, dummy_cols


def prepare_panel(
    df: pd.DataFrame,
    qual_preds: list[str] | None = None,
    *,
    demand_lag: int = 9,
    target: str = TARGET,
) -> tuple[pd.DataFrame, list[str]]:
    """Copy the panel, encode categoricals, and add static features.

    Dynamic features are not built here. The planning loop refreshes them
    from actuals, then again after each forecast write-back.
    """
    required = {SERIES, "year_month", target}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Panel is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Panel is empty")

    work = df.copy()
    work["year_month"] = work["year_month"].astype(int)
    work[target] = work[target].astype(float)
    work = work.sort_values([SERIES, "year_month"]).reset_index(drop=True)
    if work.duplicated([SERIES, "year_month"]).any():
        raise ValueError("Panel has duplicate series_id / year_month rows")

    work["_actual"] = work[target]
    work, dummy_cols = encode_categoricals(work, list(qual_preds or []))
    work = add_calendar_features(work)
    work = add_exogenous_lags(work, demand_lag=demand_lag)
    return work, dummy_cols


def engineer_snapshot(
    df: pd.DataFrame,
    *,
    demand_lag: int = 9,
    target: str = TARGET,
) -> pd.DataFrame:
    """Sorted frame with static and dynamic features from the current target.

    Useful for tests and for inspecting one feature row. The planning loop
    uses :func:`prepare_panel` plus :func:`refresh_dynamic_features` so
    calendar columns are not rebuilt on every offset.
    """
    work, _ = prepare_panel(df, qual_preds=[], demand_lag=demand_lag, target=target)
    refresh_dynamic_features(work, target=target, demand_lag=demand_lag)
    return work
