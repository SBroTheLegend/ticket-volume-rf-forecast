# Architecture

## Panel schema (generic)

| Column | Meaning |
|---|---|
| `series_id` | Hierarchy key (one series per region) |
| `region` | Region label (e.g. `Region_APAC`) |
| `year_month` | YYYYMM integer |
| `Date` | Calendar date for seasonality |
| `ticket_volume` | Target (monthly tickets / demand) |
| engineered cols | Lags, rolling std/mean, Fourier, flags |

## Planning loop

```text
for each planning_month in hold-out window:
    rebuild demand lags from history (+ prior forecasts in-horizon)
    if SI:
        fit model on year_month < planning_month
        for offset in 0..H-1:
            engineer rolling / seasonal features
            predict target_month = planning_month + offset
            write prediction back into ticket_volume (for next lag)
    if MI:
        for offset in 0..H-1:
            engineer features
            re-fit on year_month < target_month
            predict target_month
            write prediction back
    score accuracy (e.g. mean acc at Offset == 3)
```

## Feature selection (`model_manager`)

```text
start with predictors = [demand_lag_1]
for candidate in candidate_list:
    tentatively add (num or categorical)
    run demnd_hp / planning loop
    if Acc improves: keep
    else: remove
persist per-region accuracy trail + best forecast file
```

## SI vs MI

| Mode | When the forest is fit |
|---|---|
| **Single-Iteration** | Once per planning month |
| **Multiple-Iteration** | Every target month inside the offset loop |

MI is slower and can adapt mid-horizon; SI is cheaper and matches a
“freeze plan then roll” workflow.

## What is public vs private

| Artifact | Public repo | Private / on request |
|---|---|---|
| README / architecture / access | yes | — |
| Synthetic CSV + short demo | yes | — |
| Core excerpt snippet | yes | — |
| Full SI / MI modules | **no** | yes (email) |
