# Architecture

## Panel schema (generic)

| Column | Meaning |
|---|---|
| `series_id` | Hierarchy key (one series per region in the demo) |
| `region` | Region label (e.g. `Region_APAC`) |
| `year_month` | YYYYMM integer. Month must be 01..12 |
| `Date` | Calendar date for Fourier terms |
| `ticket_volume` | Target (monthly tickets / demand) |
| engineered cols | Lags, rolling std/mean, change ratios, flags |

`load_panel` also accepts the aliases `Ticket Volume`, `Sales Market+`, and
`iter`. The panel is synthetic in this repository.

## Two windows

```text
history .........| selection planning months | holdout planning months
                  (feature decisions)         (reported accuracy)

holdout_start >= selection_end
selection frame is cut at selection_end, so offsets cannot score the holdout
```

Accuracy inside either window is the mean capped accuracy at `score_offset`
(default 3).

## Planning loop (`forecast_panel`)

```text
static features once: Fourier terms, exogenous ratio lags

for each planning_month in (window_start, window_end):
    restore actuals
    refresh lags, rolling stats, change ratios from the current target
    if SI:
        fit on year_month < planning_month
    for offset in 0 .. H-1:
        target_month = planning_month + offset
        if offset > 0:
            refresh target-dependent features (forecasts already written)
        if MI:
            fit on year_month < target_month
        predict target_month
        write the unrounded forecast into ticket_volume
        record the rounded forecast
    average feature_importances_ across the fits
```

Rolling stats use `GroupBy.rolling`, which keeps the row index. A
`groupby.apply` of `Series.rolling` does not: on current pandas the apply
result is a MultiIndex and the assigned column does not line up.

In-horizon lags use the written forecast. They are not left pointing at the
future actual.

## Feature selection (`greedy_select`)

```text
start with predictors = [demand_lag_1]
optionally shuffle the candidate list (seeded)
for candidate in that order:
    tentatively add (numeric, or categorical dummy)
    score ONLY the selection window
    if accuracy strictly improves: keep
    else: remove
optional backward pass:
    drop a selected feature (never the seed) when accuracy does not fall
rescore the kept set on the selection window
freeze that set and score the holdout window
```

Nothing that fails to improve is retained just because it was tried. The
baseline `demand_lag_1` model remains when every candidate is flat.

## SI vs MI

| Mode | When the forest is fit |
|---|---|
| **Single-Iteration** | Once per planning month |
| **Multiple-Iteration** | Every target month inside the offset loop |

MI is slower. The default forest uses `n_jobs=-1`, and features that do not
depend on the forecast (Fourier terms, exogenous ratio lags, dummy encoding)
are not rebuilt on every offset.

## What is in this repository

| Artifact | In the repo |
|---|---|
| README / architecture / access | yes |
| Synthetic CSV, synthetic generator, demo | yes |
| Core excerpt snippet | yes |
| SI / MI planning loop and stepwise selection | yes (`ticket_volume_rf`) |
| Employer data or internal paths | no |
