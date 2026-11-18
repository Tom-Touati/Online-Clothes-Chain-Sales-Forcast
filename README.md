# Online clothes chain: sales forecast, July to October 2026

Monthly unit-sales forecast for 100 style-colours of an online fashion retailer for July to October 2026, from the order lines of
January 2024 to June 2026. Each item also gets a confidence level (1 to 3), and 18 items get a written explanation.

**The submission is [`Edikted_Forecasting_Exercise_filled.xlsx`](Edikted_Forecasting_Exercise_filled.xlsx)**: 630,254 units
(July 177,694, August 187,685, September 130,828, October 134,047); confidence 10 items at 1, 86 at 2, 4 at 3.

## How the forecast is made

Each item's July–October total is forecast relative to its own recent rate, then split by month.

1. **Six models.** Planning rules written by hand, LightGBM (a launch-weighted version for items under six months old), a neural
   network, two TabPFN models, and a model that forecasts each item at each place it sells (a quarter weight).
2. **Their weighted average, corrected** by two small models trained on earlier forecast errors: biases by volume, age and discount,
   and signs of breakouts and fading items (including how each item's photo and text look-alikes ramped last year).
3. **Monthly split** by the rules' and the neural network's monthly shapes.
4. **Judgement in two places**, both recorded in the workbook: the two new site sales in the campaign plan (August, September) are
   added by each item's edikted.com share (+1.4%); hand rules for three World Cup items and one out-of-stock item.
5. **Ranges, confidence, buying quantity** calibrated on the system's own backtest errors.

The workbook's "Model explanation" and "Method and decisions" sheets give the reasoning and the evidence for each choice.

## How it was tested

A rolling backtest at 13 month-ends (February 2025 to February 2026): everything is trained only on data before the month-end
and scored on the next four months. On the nine most recent month-ends the error (WAPE of item totals) is 0.38 of a recency
forecast's (last three months × 4), 0.61 of a naive seasonal forecast's and 0.78 of the hand rules', better at every month-end.

## What is where

| Folder | Contents |
|---|---|
| [`pipeline/`](pipeline/) | The scripts of the current solution, in stage order, with a map of what each one does and how it was run |
| [`data/`](data/) | The tables those scripts read and write: member forecasts, corrections, ranges, backtests |
| [`rule_agents/`](rule_agents/) | The hand-written planning rules, one folder per backtest month-end and for June 2026 |
| [`exercise/`](exercise/) | What the retailer provided: the exercise workbook and the item attributes |
| [`model_submissions/`](model_submissions/) | Every other workbook: single-model forecasts and the earlier versions of the submission |
| [`docs/`](docs/) | The decision log (every experiment and decision, with scores), the data report, three interactive pages |
| [`archive/`](archive/) | Past attempts, kept as they were: about 380 other scripts, their run logs, tables and pages |

The raw order lines (`Dataset.parquet`, 249 MB) are not in the repository (GitHub's file limit); put the file at the repository
root to rerun stage 1. All scripts run from the repository root; see [`pipeline/README.md`](pipeline/README.md).
