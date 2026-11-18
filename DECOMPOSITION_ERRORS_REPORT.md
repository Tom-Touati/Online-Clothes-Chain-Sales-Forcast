# How to decompose, and where we miss: overnight report (17–18 Nov 2026)

You asked: *"the true challenge is to know how to decompose. the levels … also deep error analysis. what are the items we ain't good at, why and when. try PCA like hierarchical."*

This report covers decisions D238–D250 in log.md. The scripts are `eda/179*.py` and `eda/175i–j`.

- **Forecast scored:** the final system, p2c32a. It is the arithmetic mean of five members: rules, the D198 LightGBM, the NN, conformal TabPFN and TabPFN dyn.
- **Backtest:** 13 origins, scored on the nine recent ones. Each score is WAPE ÷ R3 on 4-month totals.
- **Time-honest:** every correction is trained only on origins whose outcomes were known at that point.

## 1. Short answer

1. **The right hierarchy is site → look → type within look → style → colour, not site → macro → type.**
   - A look is a cluster of items bought together in carts, with similar photos and text.
   - At an equal number of groups, looks hold about twice the shared error that product types hold (table in 2.3).
   - Most of the variation is in style; colour is almost free of error.
2. **Knowing the totals at the right level would cut error by a third, but those totals cannot be forecast.**
   - If the type × look cell totals (about 250 cells) were known, WAPE ÷ R3 would fall from 0.398 to 0.260.
   - Every cell-level model we built lost to simply adding up the item forecasts.
   - So the hierarchy is the right way to explain the forecast, but not a way to improve it.
3. **The error is concentrated, and almost all of it is a shared miss.**
   - The worst 5% of item-origins hold 42% of the error.
   - Large misses hold 44% of the error. In those, every model is wrong in the same direction (all under: 31% of total error; all over: 13%).
   - Reweighting the models cannot fix these misses.
4. **Who we miss:**
   - **Young launches** (< 3 months): 22% of the error, under-forecast by 17%.
   - **Small and mid-volume items:** under-forecast by 17–28%.
   - **Dresses:** WAPE ÷ R3 0.55.
   - **Items falling fast:** 0.57.
   - **Items where the models disagree:** 0.52.
5. **When we miss:**
   - **Black Friday windows:** 56% of the error.
   - **Early 2025 origins,** when training history was short.
   - **By month:** November and January are under-forecast; September is over-forecast by 28%. The monthly split puts some of the autumn peak too early.
6. **Why: breakouts.**
   - The single largest failure is a young launch that breaks out into the autumn and holiday peak. Sales grow 3–10× while the models extrapolate a gentle ramp. Examples: a black-and-white hoodie missed at four origins in a row, a sweater-and-shorts set, black pants, a tank top.
   - Breakouts are only weakly visible at the origin (AUC 0.66, from weekly momentum and look-peer growth).
   - **Half of them show within the first month** (actual ≥ 1.5× forecast).
   - Misses do not persist: an item badly missed now is no more likely to be missed again four months later (rank correlation 0.06).
7. **What moved the score:**
   - Two item-level corrections, combined by geometric mean with each origin's total kept: past segment biases (volume × age × discount) and a breakout model.
   - **p2c32a_geo: 0.3902 against 0.3977** (bootstrap −0.0073 to −0.0031); all 13 origins 0.4482 against 0.4542; bias unchanged.
   - **Buying loss is level** (10.78% against 10.74%): it improves accuracy, not buying.

## 2. Decomposition

### 2.1 Oracle: if one level's 4-month total were known (F rescaled to it)

| Level known | WAPE ÷ R3 |
|---|---|
| none (as forecast) | 0.398 |
| site | 0.395 |
| macro group | 0.384 |
| product type | 0.351 |
| look | 0.349 |
| **type × look** | **0.260** |
| style | 0.078 |

### 2.2 Where the variation sits (nested shares)

| | macro | type | look | style | colour |
|---|---|---|---|---|---|
| Growth vs R3, path A (macro → type → style) | 0.22 | 0.13 | | 0.59 | 0.06 |
| Growth vs R3, path B (look → style) | | | **0.53** | 0.44 | 0.03 |
| Final-system error, path A | 0.04 | 0.11 | | 0.77 | 0.08 |
| Final-system error, path B | | | 0.23 | 0.72 | 0.04 |

### 2.3 Which grouping, at a matched number of groups (D250)

Each value is the share of the final system's residual variance that the group means explain, minus the same share for random groups of the same sizes.

| Grouping | Groups | Over random |
|---|---|---|
| macro | 6 | 0.036 |
| co-surprise clusters (items whose surprises move together) | 26 | 0.129 |
| **look (carts + visuals)** | 34 | **0.192** |
| co-surprise within type | 46 | 0.146 |
| product type | 48 | 0.102 |
| co-surprise clusters | 106 | 0.119 |
| **type × look** | 250 | **0.214** |
| co-surprise clusters | 256 | −0.013 |

Co-surprise clusters hold less shared error than looks, and at about 250 groups they are no better than random. The reason is that they are unstable: item loadings correlate only 0.48 between the two halves of the history. Looks built from carts and content are stable, so they are the better middle layer.

### 2.4 Hierarchical PCA of the monthly surprise panel (D241)

Panel: 482 items × 24 months. A surprise is an item's month against its own 3-month level × its product type's seasonal step.

- **Global factors:** three global factors explain 36% / 52% / 61% of the variance (cumulative).
  - **PC1 is a recurring seasonal factor:** its next 4 months are predictable from last year (R² 0.81).
  - **PC3 is momentum:** R² 0.51 from the last 3 months.
  - PC2 is not predictable.
- **Within-look factor:** one factor per look explains another 17.8%. Per product type gives 13.6%; random groups give 14.2%.
- **Item-specific:** 21% remains.
- **Forecast correction from these factors (D242):** the momentum projection correlates +0.15 with the residual, but WAPE does not move (0.3979). The factors explain the past well; their forward value is already inside the members.

Figure: `eda/figs/hpca_factors.png`

### 2.5 Middle-out through cells (D240)

At every level, adding up the item forecasts beats a model of the cell total (cell-level WAPE):

| Cell level | Sum of item forecasts | Best cell model |
|---|---|---|
| type × look | 0.199 | 0.244 |
| look | 0.157 | 0.218 |
| type | 0.135 | 0.208 |

Item-level middle-out versions were level or worse (best 0.4003). **We keep the hierarchy as an explanation cascade**, in the "Cascade (hierarchy)" sheet of the candidate workbook.

## 3. Error analysis (D244)

Figure: `eda/figs/err_final_worst.png`, the 12 largest misses.

| Segment | Unit share | Error share | WAPE ÷ R3 | Bias |
|---|---|---|---|---|
| launches < 3 months | 16% | 22% | 0.58 | −17% |
| 3–6 months | 16% | 16% | 0.34 | +5% |
| smallest volume fifth | 1% | 2% | 0.72 | −14% |
| 2nd volume fifth | | | 0.64 | −28% |
| largest volume fifth | | | 0.31 | +1% |
| dresses | 5% | 6% | 0.55 | −14% |
| warm items | 25% | 24% | 0.28 | −7% |
| models disagree (top fifth) | | | 0.52 | |
| falling fast | | | 0.57 | |
| Black Friday windows | 51% | 56% | 0.51 | |

**When.**
- **By origin:** 2025-02 to 2025-06 score 0.60–0.76, over-forecast by up to 37%. December 2025 to February 2026 score 0.19–0.27.
- **By target month:** September +28% (over), November −19%, January −14%.

**Why.** The large misses are shared misses: every model wrong in the same direction. They do not persist: an item missed now is no more likely to be missed four months later (rank correlation 0.06), and only 11% of items are chronically off. The pattern is "the wrong item breaks out", not "a bad item".

## 4. Breakouts (D247–D248)

A breakout is an item whose actual is at least 1.65× the forecast, with at least 20 units.

- **Rates by age:**
  - **under 3 months:** 10–20% break out;
  - **3–6 months:** 4–8% (the calm band);
  - **6–12 months:** 9–16%;
  - **12 months or older:** 5–9%.
- **Signals at the origin (AUC among young items):**
  - weekly signals: last 4 weeks' share of life sales 0.66, weekly momentum 0.65;
  - look-peer signals: 3-month growth 0.59, recent launches 0.58;
  - more age, more weeks live and deeper discount mean fewer breakouts.
- **After the origin:** breakouts run at 1.47× forecast in month 1, then 2.0× and 2.2×.
  - A rule flagging items at ≥ 1.5× in month 1 catches 49% of breakouts (61% of young ones) and flags 7% of the rest (precision 0.41).
  - **Operational suggestion:** a monthly re-forecast or chase-buy check on young items. Rescaling months 2–4 by the month-1 ratio would cut their WAPE from 0.326 to 0.303.

## 5. Systems that moved (all production-licensable, time-honest)

| System | WAPE ÷ R3 recent | All 13 | Bias recent | Buying loss |
|---|---|---|---|---|
| p2c32a (final system) | 0.3977 | 0.4542 | −4.6% | 10.74% |
| + segment re-allocation (D245) | 0.3918 | 0.4495 | −4.6% | 10.76% |
| + breakout model, total kept (D247) | 0.3930 | 0.4505 | −4.6% | |
| **+ geometric mean of both (D248)** | **0.3902** | **0.4482** | −4.6% | 10.78% |
| qsc32a (previous accuracy best) | 0.3964 | | | |
| newsvendor ladder u65 (buying best) | | | | 10.38% |

**On halving the error:** the decomposition shows the remaining error lives in which styles break out. That is weakly visible at the origin and much more visible one month in. The forecast itself gains about 2%. The bigger lever is operational: re-forecast or chase-buy monthly on young items.

The Excel candidate (`Edikted_Forecasting_Exercise_candidate_final.xlsx`) is still p2c32a. Whether to rebuild it with the geo correction, and whether to switch the submission, are your decisions.
