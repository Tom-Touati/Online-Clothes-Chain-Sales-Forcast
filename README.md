# Online clothes chain sales forecast

Monthly unit-sales forecast for 100 style-colours of an online fashion retailer, July to October 2026, from two and a half years of
order lines (January 2024 to June 2026).

## What is here

| Path | Contents |
|---|---|
| `Edikted_Forecasting_Exercise_filled.xlsx` | The submitted workbook: forecast, confidence, explanations, model explanation |
| `Edikted_Forecasting_Exercise_candidate_*.xlsx` | Earlier and later candidate workbooks (nx7 is the latest) |
| `log.md` | The decision log: every modelling decision and experiment, append-only, with scores |
| `eda/` | All scripts: features, models, backtests, corrections, ranges, workbook builders, page builders |
| `agent_eval/` | The hand-written planning rules, one version per backtest month-end |
| `data/` | Derived tables: features, backtest predictions of every model, calibration and scoring outputs |
| `*_REPORT.md`, `SYNC.md` | Reports and working notes |

The raw order-line file (`Dataset.parquet`, 249 MB) is not included: it is larger than GitHub's file limit.

## The forecast in short

Each item's four-month total is forecast relative to its recent rate by six models (planning rules, LightGBM with a launch-weighted
version for new items, a neural network, two TabPFN v2 models, and a per-location recommender at a quarter weight), averaged, then
corrected by two small models trained on earlier forecast errors. On a rolling backtest at 13 month-ends the error on the nine most
recent ones is 0.38 of a recency baseline's (last three months x 4), 0.61 of a seasonal naive's and 0.78 of the hand rules'.

The builder of the latest workbook is `eda/181n_final_nx7_2026.py`; the audit of the submission is entry D389u in `log.md`.
