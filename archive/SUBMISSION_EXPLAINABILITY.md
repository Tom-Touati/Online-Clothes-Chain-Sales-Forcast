# Explainability, evidence and future work

*Fork's sections for the submission report. Decisions D191–D210 in `log.md`. All numbers are time-honest backtests: 13 monthly origins from Feb 2025 to Feb 2026, each forecasting the next four months. "Recent" means the nine origins from Jun 2025.*

## 1. How to read a forecast

Each item's forecast is built as a cascade. Main's workbook shows every layer as a log contribution:

1. **R3**: the recency baseline, 4/3 × the last three months.
2. **Type seasonality**.
3. **Members**:
   - the rule agents;
   - the D198 LightGBM quantile model: lags, like-for-like history, events, temperature, LLM fashion scores, and time-aware co-purchase neighbours;
   - the neural net;
   - TabPFN v2, conformally calibrated;
   - TabPFN v2 with 32 estimators.
4. **Arithmetic stack** of the members: the point forecast.
5. **Buying column**: the same stack with the quantile members at P65.

The **"why" column** explains each item's deviation from R3. It comes from `data/fk_2026_explain.parquet`, which gives each item's top three drivers with their percentage effects.

How the "why" column is built:
- The tree member is refitted at June 2026: the volume-weighted D198 LightGBM median.
- TreeSHAP splits its predicted deviation from R3 exactly into feature contributions.
- An LLM categoriser grouped those contributions into 11 named categories (D194).

**Why one member's SHAP can explain the whole stack.** On the nine recent backtest origins, the 11 SHAP category sums explain **94%** of the final system's deviation from R3 (R² 0.941, leave-one-origin-out ridge). The direction is right for **92%** of items. The tree member's own deviation alone gives R² 0.917. The other members mostly agree with the tree on direction and differ on size, so the category story carries over (D210, `eda/explain_final_output.txt`).

**Final numbers** (Main, D237, workbook `Edikted_Forecasting_Exercise_candidate_final.xlsx`; the submitted workbook is untouched):
- **Point forecast** (p2c32a): **629,372** units over the 100 targets. By month: Jul 177,820, Aug 188,423, Sep 128,951, Oct 134,178.
- **Buying quantity** (members at P65): 659,950.
- **Accuracy alternative** (qsc32a): 620,365.
- **Cascade over the targets:** R3 473,374 → type seasonality 593,665 → members: rules 620,269, D198 quantile 633,461, NN 698,591, conformal TabPFN 571,510, TabPFN dyn 622,443 → forecast 629,385 (before monthly rounding).

## 2. What drives the July–October 2026 forecast

Tree member at the June 2026 origin, for the 99 targets present at the origin. R3 is 473,373 and the tree member's median is 538,834 (+13.8%). Contributions are R3-weighted averages over the targets:

| Category | Effect |
|---|---|
| Portfolio baseline: items typically drift below their recent pace | −19.4% |
| Event-inflated recent sales: May/June prom and graduation lift fading, read as seasonality | +7.1% |
| Upcoming events and occasions: back-to-school, Labor Day, Halloween | +5.4% |
| Item age and sales scale | +4.7% |
| Shopper-cluster seasonality: last year's ramp of co-purchase neighbours | +4.4% |
| Last year's like-for-like pattern | +2.6% |
| Regional climate mix | +1.8% |
| Short-term momentum | +1.4% |
| Life cycle, weather, product style | within ±1% |

Examples of item stories:

| Item | R3 | Tree member | Main drivers |
|---|---|---|---|
| Roman Low Rise Baggy Jeans | 17,812 | 23,843 | recent buzz +21%, structural denim trend +12%, portfolio drift −19% |
| Stephanie Lace Mini Dress | 15,715 | 7,476 | cooling momentum −29%, life cycle −14% |
| Vikkey Lacey Satin Tank Top | 11,305 | 24,906 | strong recent buzz +48%, a young item still scaling +45% |
| Washed Faux Leather Micro Cargo Shorts and Aisha Low Rise Mini Skirt | — | — | upcoming events +21–24% |

## 3. Evidence that the explanation layer is faithful

**(a) Deviations from R3, SHAP (D191).** The model's predicted deviation has R² 0.60 against the actual deviation (0.74 on the targets). Its direction is right 83% of the time.

**(b) LLM categoriser and three judges (D194).** One LLM read the SHAP summaries and the highest-gain trees and proposed 16 categories: 11 explained and 5 unexplained. Three LLM judges (Opus, Sonnet, Haiku) scored 120 item cards:
- Their scores agree with each other (mean pairwise r 0.88) and with the exact SHAP sums (r 0.94).
- They agree on the residual's main cause with Fleiss' kappa 0.66.

**(c) Simple surrogates (D195).** A linear model on the 11 category sums recovers **72%** of the model's lift over R3; on the 100 targets, 80%. A linear model on the 15 largest SHAP values recovers **85%**. A linear model on the raw features recovers nothing (−20%). So the effects are non-linear in the inputs, and the categories describe them well.

**(d) Final system (D210).** The categories explain 94% of the final stack's deviation, as in section 1.

**What remains unexplained.** The judges' diagnosis of the residual is a label read with hindsight, not a forecast. On the 120 cards:
- promotions or markdowns not in the data: 38 cards;
- none: 33 cards;
- launch, sell-through or assortment decisions: 25 cards, half of them young items;
- viral spikes: 7 cards;
- supply limits: 7 cards;
- data artefacts: 4 cards.

## 4. How the final system was chosen

Production-licensable systems only. WAPE ÷ R3, recent / all 13 origins; buying loss from Main's buying simulation (11 windows).

| System | WAPE recent | WAPE all 13 | Bias recent | Buying loss |
|---|---|---|---|---|
| R3 + rules | 0.4879 | 0.5090 | −6.4% | — |
| p2 (champion at the start of the night) | 0.4100 | 0.4586 | −4.1% | 10.78% |
| **p2c32a**: point forecast (rules, D198 q, NN, conformal TabPFN v2, TabPFN v2 dyn 32; arithmetic mean) | **0.3977** | **0.4542** | −4.6% | 10.74% |
| qsc32a: p2c32a + feature-pruned LightGBM | 0.3964 | 0.4520 | −6.6% | 10.93% |
| **Buying column**: w03c32a with the quantile members at P65 | 0.4208 | 0.4980 | +4.5% | **10.38%** |

Three findings shaped the choice:
- **Bias drives buying loss.** Buying loss tracks bias more than WAPE. The geometric-mean stacks were the most accurate but forecast low. Arithmetic means and upper-quantile members buy better.
- **The best order quantity is an upper quantile.** Along the ladder of quantile levels, buying loss bottoms out flat between P65 and P70 (10.38%). Hence a separate buying column next to the accuracy point forecast.
- **TabPFN-3.5 isn't worth it here.** Its stacks add about 0.005 WAPE and nothing on buying, and its licence is non-commercial and non-production. It is therefore not used.

## 5. What was tried and not adopted (fork, Nov 16–17)

| Idea | Result |
|---|---|
| Cluster-history, fashion-attribute trend, trend-cycle and regional-diffusion features (D205) | All level or worse in the trees, even though several carry residual signal. The trees already carry it through the item's own seasonality and the cart neighbours. |
| 5-nearest-neighbour features in the dynamic and content embeddings (D205) | Level or worse |
| CatBoost, XGBoost, LightGBM-DART, Tweedie on units (D206, D207) | Worse as extra members |
| Optuna tuning on early origins (D206) | The current parameters stayed best |
| Seed and training-window bagging (D206, D207) | Level |
| TabPFN v2 fine-tuned per origin (D207) | Worse at every origin it completed |
| A time-honest level multiplier (D207) | Makes bias worse |
| Graph neural network style embedding (GraphSAGE over photo/text-neighbour and cart edges, D208) | Passes the cart gate against the raw product features (future carts: rank correlation 0.284 vs 0.159). Against the best content embedding v6 it is level: +0.004 rank correlation on future carts, worse on past carts. Not used as features, since neighbour features gave nothing in the trees (D205). |

Time-series foundation models (Chronos-2, Bolt, TimesFM) and other tabular foundation models (TabDPT, Mitra, Nori, TabICL) were rejected earlier (D192–D201).

## 6. Future work, in order of expected value

1. **Promotion and markdown calendar per item.** It is the largest unexplained cause: 38 of 120 diagnosed cards, mean residual +23%. A known-future discount plan per item and week would turn it into an input. The site-level campaign calendar alone gave nothing (D173), so item-level depth and timing are what's needed.
2. **Launch, assortment and inventory data.** Young items and sell-through decisions are the second cause (25 of 120 cards, with the largest residuals). Inputs that would help:
   - planned launch dates and buy depth;
   - on-hand inventory, for true stock-out censoring.
   The stock-out-corrected level already gives a small gain (D174); real inventory would replace that proxy.
3. **Buying policy on real economics.** The P65–P70 bottom comes from the simulation's assumed margins. With per-item cost, price and salvage values, the order quantile can be set per item (newsvendor critical ratio). The calibrated quantiles needed for this already exist.
4. **Monitoring and recalibration.** Bias drifts with the season. Four routines would keep it in check:
   - recalibrate the conformal layer monthly, on the latest known outcomes;
   - track bias by segment;
   - retrain the members monthly; TabPFN needs no training;
   - keep the backtest harness (`eda/199d_stack_any.py`) for any new member.
5. **Cold-start items.** For new launches without history, the content embedding v6 (photos and text) is the best similarity measure on the cart gate. A launch-analog member could use it; it was not needed here, because every target has history.
6. **Social and trend signals for viral spikes.** Google Trends did not help (D183); viral spikes are 7 of 120 cards. Creator, TikTok or Instagram mention counts, if available, are the candidate inputs.
7. **A commercial TabPFN-3.x licence.** It is worth about 0.005 WAPE and nothing on buying. Low priority.
