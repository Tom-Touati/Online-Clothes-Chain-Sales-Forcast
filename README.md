# Online clothes chain: sales forecast, July to October 2026

Monthly unit-sales forecast for 100 style-colours of an online fashion retailer for July to October 2026, from the order lines of
January 2024 to June 2026 (about 1,500 items, sold on edikted.com, in the chain's own stores and through partners). Each item also
gets a confidence level (1 to 3), and 18 items get a written explanation.

**The submission is [`Edikted_Forecasting_Exercise_filled.xlsx`](Edikted_Forecasting_Exercise_filled.xlsx)** (version nx18): 630,254 units
(July 177,694, August 187,685, September 130,828, October 134,047); confidence 10 items at 1, 86 at 2, 4 at 3. Its sheets:

| Sheet | What it holds |
|---|---|
| Forecast | Monthly units, total, confidence and the model's raw total for each item; explanations for 18 items |
| Model explanation | A table of every metric for SYS and, as a ratio to it, for R3, SN, AR and each of SYS's six models alone (all lower-is-better; white = as good as SYS, deeper blue = SYS better by more), shown under "How you tested it" and again below the boxes, and three boxes, one topic per line, each opening with a bold summary: the approach (summary, real-world evaluation, models, information, data artefacts), how it was tested (protocol, leakage checks, one metric per aspect, the buying simulation, explainability), where it is weak |
| Method and decisions | Each major choice with its evidence, by area: framing, models, information, judgement, metrics (one row per aspect: volume-weighted error, per-item error, hit rate, bias, R², ranges, confidence, decision value, how sure the scores are), evaluation, explainability (the cascade, SHAP) and a six-step improvement roadmap |
| Metrics vs baselines | Every metric of the submitted forecast next to R3, the seasonal naive and the agents' rules, and how much better it is than each (the table below) |
| Cascade (items) | Each item's July–October total step by step: R3 and a seasonal baseline, the six models (model 1 is the planning rules), their weighted average, the total after the corrections (the raw model total), the share sold on edikted.com, the campaign uplift, the final total (World Cup and out-of-stock rules included), the buying quantity, the highest / lowest model ratio, confidence, P10 / P25 / P75 / P90 |
| Cascade (hierarchy) | The same steps, from R3 to the buying quantity, summed for the total, each product type and each style |
| Ranges (totals) | P10, P25, P50, P75 and P90 of the July–October total for all 100 items, each of the 13 product types and the two styles with three items |

## Interactive pages

Three self-contained HTML pages, in [`docs/pages/`](docs/pages/). GitHub shows an HTML file as source code, so download the pages (or
clone the repository) and open them in a browser: no server is needed, only an internet connection for the fonts. The atlas and the
error study link to each other when both files are in the same folder; the item explorer stands alone.

| Page | What it shows |
|---|---|
| [**Item explorer**](docs/pages/item_explorer.html) | One item at a time, for each of the 100: its history, past forecasts and the July–October projection with its ranges; the steps from the recent rate to the forecast; what pushes the tree model up or down; weekly history; where it sells; its discount on edikted.com; the item against its product type and group; its nearest neighbours; what the item is and the business around it |
| [**Forecast atlas**](docs/pages/atlas.html) | The project end to end: from order lines to a filled workbook; what the data holds (six-fold growth that is mostly more items, a site that is always on sale, each product type's own year, the launch curve, fifteen data traps); every model from the first quantile model to the final stack; the evaluation against recency; the forecast for the 100 items; every input of every model; the decision log |
| [**Error study**](docs/pages/error_study.html) | Where the backtest forecast goes wrong and what would fix it: how concentrated, one-sided and shared the error is; 34 cohort axes and pairs of them; which missing information would have helped most (the roll-out plan); how much of the improvement is real; what the large misses have in common; how the 100 targets sit against the risky cohorts |

A fourth page, the scoreboard of every system scored in the project, is not in the repository: it loads its numbers from claude.ai.
Its table is [`data/scoreboard.json`](data/scoreboard.json): 648 rows, 72 systems (from R3 and single models to each version of the
submission) on 9 subsets of the backtest (all 13 month-ends, the nine recent ones, the 100 targets, each season, ...), with WAPE and
the per-item scores as ratios to R3, bias, the share within ±20%, pinball loss, bootstrap intervals and the buying-simulation loss.

## How the forecast is made

Each item's July–October total is forecast relative to its own recent rate, then split by month.

1. **Six models that see the problem differently.**
   - Planning rules written by AI agents: recent daily rate × seasonal and event indices by product type × launch ramp and fade. At
     each backtest month-end a fresh agent saw only the data up to that date, then revised the previous rules.
   - LightGBM, with a launch-weighted version for items under six months old.
   - A neural network that learns an item embedding and reads the weekly sales curve.
   - Two TabPFN v2 models, one of them with calibrated ranges.
   - A model that forecasts each item at each place it sells and adds them up. It has a quarter weight, because it leans most on
     last year.
2. **Their weighted average, corrected** by two small models trained on earlier forecast errors. One corrects biases by volume, age
   and discount. The other looks for signs of breakouts and fading items, including how each item's photo and text look-alikes
   ramped last year.
3. **Monthly split** by the rules' and the neural network's monthly shapes.
4. **Judgement in two places**, both recorded in the workbook (the "Model raw total" column is the forecast before both):
   - The campaign plan has two new site sales (August, September). They are added at half the measured lift of a past campaign day,
     scaled by each item's edikted.com share (+1.4% in total).
   - Hand rules for three World Cup items and one out-of-stock item.
5. **Ranges, confidence and a buying quantity**, calibrated on the system's own backtest errors.
   - Confidence follows the exercise's own definitions. Level 1 goes only to items with the narrowest range that are also rising and
     sold in the same months last year. Level 3 goes to the four hand-ruled items, and level 2 to the rest.
   - The buying quantity is about 8% above the forecast, because a buying simulation showed that forecasting too low costs more
     than forecasting too high.

The models use momentum; seasonality read from like-for-like items; each item's place on its launch curve; where it sells; items
bought together; fashion attributes and looks (language-model scores, photo and text embeddings); about 30 US events; and temperature
where each item sells: actual monthly temperatures for every US state from NOAA (Hawaii and DC from Open-Meteo), weighted by where
each item sells, observed up to each month-end, with normals for the coming months.

## The data and its traps

The order lines have artefacts that a naive model would learn as signal (all fifteen, with their treatment, in
[`docs/EDA_REPORT.md`](docs/EDA_REPORT.md)):

- **A growing survivor panel.** The 1,500 items are survivors and the panel grows over time, so total units grow sixfold mostly
  because there are more items. Trend and seasonality are read from like-for-like items, never from raw totals.
- **A mass cancellation** on 14–15 October 2025 (145k units). Cancelled lines are excluded, as the target says.
- **More own stores.** They grew from 1 to 16 during the period, which is handled through store shares and the location model.
- **Partial launch months.** Rates are per day on sale, and 85 items sold before their published date.
- **Possible stock-outs** hide about 0.4% of units.

## How it was tested

A rolling backtest at 13 month-ends (February 2025 to February 2026). Everything, including the corrections, is trained only on
data before the month-end and scored on the next four months. Error is the WAPE of item totals, always read against simple
baselines:

**Real-world evaluation first.** The main baseline is planning rules written by AI agents: at each month-end a fresh agent saw only
the data up to that date (never anything later, apart from whatever general knowledge its language-model pretraining holds, which
cannot be audited) and revised the previous rules, as a planner would. And the forecast is also judged by what it is worth in the
decision it serves: a daily buying simulation (orders every 7 days at most, 21-day lead time) scores the profit lost against perfect
foresight.

Every metric on the nine most recent month-ends (9,031 item forecasts), all scored the same way (`pipeline/232_metrics_table.py`;
the workbook's "Metrics vs baselines" sheet):

| Metric | Submitted system (SYS) | R3 | Seasonal naive (SN) | Agents' rules (AR) | SYS better than R3 | SYS better than SN | SYS better than AR |
|---|---|---|---|---|---|---|---|
| Volume-weighted error (WAPE) | 26.5% | 69.9% | 43.6% | 34.1% | 62% lower | 39% lower | 22% lower |
| Per-item error (median) | 24.2% | 57.2% | 38.8% | 30.3% | 58% lower | 38% lower | 20% lower |
| Hit rate: items within ±20% | 42.3% | 18.1% | 27.1% | 34.6% | +24.3 pts | +15.2 pts | +7.7 pts |
| Hit rate: units within ±20% | 49.1% | 19.2% | 29.9% | 37.9% | +30.0 pts | +19.2 pts | +11.2 pts |
| Bias | -5.6% | +15.4% | -6.7% | -6.4% | +9.8 pts closer to 0 | +1.0 pts closer to 0 | +0.8 pts closer to 0 |
| Error plus bias | 33.6% | 113.7% | 50.4% | 45.0% | 70% lower | 33% lower | 25% lower |
| Variance explained (R², units) | 0.85 | 0.15 | 0.64 | 0.76 | +0.71 | +0.21 | +0.10 |
| Variance explained (R², log units) | 0.92 | 0.76 | 0.84 | 0.90 | +0.16 | +0.09 | +0.02 |
| Pinball loss, P10 to P90 | 10.5% | 28.1% | 16.7% | 13.2% | 63% lower | 37% lower | 20% lower |
| Buying loss (decision value) | 10.6% | 19.5% | 14.5% | 12.4% | 46% lower | 27% lower | 14% lower |
| Win rate |  |  |  |  | 76% of items | 68% of items | 61% of items |

On last July–October, the same months as this forecast, the error is 0.68 of R3's; against the three baselines averaged, 0.59.

- **Against the baselines.** The system beats each baseline at every one of those month-ends, with 95% intervals (resampling
  styles and whole month-ends) well below 1. About four in ten item totals land within ±20%.
- **Selection.** In the later rounds a change was kept only if it also held on month-ends not used to tune it, and with last year's
  same season removed from training. Dozens of ideas, many in several variants, were rejected.
- **Leakage checks passed.** Features rebuilt from data cut at two test month-ends (June and October 2025) are identical. A model trained on shuffled outcomes
  collapses. A model that never saw an item's style is only 1% worse, so the model generalises rather than memorising items.

## Where it is weak, and what would improve it

The largest misses are young launches that break out, with sales growing 3–10× within the window. Every model misses them
together: items bought together rise and fall together, and past sales do not foresee that shared shock. The forecast ran about 6%
below actual sales on the recent month-ends.

Measured in the backtest, these would help most:
- a re-forecast after the first weeks of sales (half of the breakouts show within a month);
- the retailer's per-item plan for stores, partners, launches and discounts, as known inputs;
- demand signals that come before sales: page views, add-to-cart, wishlists, social views of a look;
- pricing tests, to measure price response causally;
- a third season of history.

## What is where

| Folder | Contents |
|---|---|
| [`pipeline/`](pipeline/) | The scripts of the current solution, in stage order, with a map of what each one does and how it was run |
| [`data/`](data/) | The tables those scripts read and write: member forecasts, corrections, ranges, backtests |
| [`rule_agents/`](rule_agents/) | The planning rules written by AI agents, one folder per backtest month-end and for June 2026 |
| [`exercise/`](exercise/) | What the retailer provided: the exercise workbook and the item attributes |
| [`model_submissions/`](model_submissions/) | Every other workbook: two early candidates (the quantile model with the neural network, and with the network and the agents' rules) and the earlier versions of the submission |
| [`docs/`](docs/) | The decision log (every experiment and decision, with scores), the data report, the three interactive pages |
| [`archive/`](archive/) | Past attempts, kept as they were: about 400 other scripts, their run logs, tables and pages |

## Rerunning it

All scripts run from the repository root with Python 3.12 (`python pipeline/<script>`); [`pipeline/README.md`](pipeline/README.md)
gives the stages and the exact commands.
- The workbook builder (`pipeline/181z_final_nx18_2026.py`) reruns from the tables in `data/` and reproduces the submission cell for
  cell.
- Rerunning the earlier stages needs the raw order lines (`Dataset.parquet`, 249 MB) at the root. They are not in the repository
  because of GitHub's file limit.
- The neural network, TabPFN and recommender steps need a GPU.

The decision log, [`docs/decision_log.md`](docs/decision_log.md), records every decision and experiment of the project (entries
`D…`) with the commands, the scores and the verdict. [`docs/FILE_MAP.tsv`](docs/FILE_MAP.tsv) maps its file paths to this
repository.
