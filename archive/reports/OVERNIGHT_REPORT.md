# Overnight work: fixing the shortcomings (13 Nov 2026, 01:40–09:00)

Every step is in `log.md`, entries D074–D097.

## Bottom line
- **Recommended for the assignment: switch to the candidate workbook.** The decision is yours; the submitted workbook (580,968) is unchanged.
  - **The candidate:** `Edikted_Forecasting_Exercise_candidate_nn.xlsx`, total **627,253 (+8%)**: Jul 181,571, Aug 181,734, Sep 131,540, Oct 132,408. Confidence 6 / 68 / 26.
  - **How it differs:** the production quantile model is re-centred on a neural net with hierarchical embeddings and a 26-week sales-curve encoder. All rules apply unchanged.
  - **Replay with all rules at past origins (D097):**

    | Origin | Submitted WAPE | Candidate WAPE |
    |---|---|---|
    | June 2025 (the deliverable's situation) | 0.298 | 0.298 (tie) |
    | July 2025 | 0.302 | 0.312 (slightly worse) |
    | May 2025 | 0.387 | 0.341 |
    | August 2025 | 0.362 | 0.350 |

    It loses less buying profit at 10 of 10 origins.
  - **Common sense:** July/June is 1.51, against 1.48–1.51 expected from the 2024/2025 group ratios on the 2026 mix (submitted 1.40). Jul–Oct/June is 5.21 (2025: 5.21). This closes the two points left open earlier: the low July/June ratio and the 3–12-month items sitting below their benchmark.
- **Buying decisions should use the blended forecast**, spread over each month with the campaign calendar, with the 14-day sales adjustment switched off for orders placed Sep–Dec.
  - Lost profit across the 11 simulated windows: **about 12.0% against 14.5%** before, and 17.8% for the simple run-rate rule (D083, D090, D097).

## What was tried, by shortcoming

| Shortcoming | What I did | Result |
|---|---|---|
| Monthly totals are the main error; peak months come in low | Diagnosis (D075) | The November miss is not a surprise campaign. Each item's median is predicted well, but in a surge month the totals are dominated by a few items. Knowing the site total exactly would only cut WAPE from 0.363 to 0.346. |
| | Hierarchical-embedding NN (type → style → item, plus occasion, collection and attributes), as a blend partner (D076, D081, D088, D095) | **Kept.** The first seed set overstated the gain (D088). With 6 seeds the production model re-centred on it scores 0.315 against 0.322 on the representative origins, significant (D095). The monthly blend alone is not significant. |
| | Sales-curve encoder: a small recurrent layer over the last 26 weeks, inside the NN (D096) | **Kept.** Consistently better than the NN without it on both seed sets. Its summer upward bias is gone: June 2025 median bias +0.7%, against +4.9% for the plain NN. |
| | Campaign calendar as model inputs (D079; NN variant in D088) | Rejected. Better in spring and autumn, worse in summer; about 20 site-months are too few to learn from. |
| | Carrying each item's last miss into the next forecast (D078) | Rejected. Misses are site-wide and reverse the next month. |
| | Other embeddings: SVD of sales curves, NN vectors as tree features, product-name words, an NN on 4-month totals (D080, D087, D092, D093) | All rejected (neutral or worse). |
| | Learning the blend weight over time (D094) | Worse than a fixed 50/50, which is also the best fixed weight in hindsight at the production level. |
| No shape within the month | Calendar daily profile: day of week × deep-discount-day lift (D077) | Small, consistent gain (better in 10 of 11 windows). Most within-month error is item-level noise. |
| Recent-sales adjustment chases peaks | Switching it off for Sep–Dec orders (D083, D090) | The largest buying gain: peak windows drop from about 24% to about 16% lost profit. It holds for any off-window that covers Nov–Dec. |
| Hand rules for segment biases | Checked whether the NN learns them (D081) | Partly: swimwear +43% → +35%, young Homecoming/Halloween −20% → −16%. The rules are still needed. |
| Safety stock too thin at peaks | The NN's own uncertainty range for safety stock (D082); a higher service level tested (D090) | The NN's range is too narrow (much worse). The 85th percentile beats the 80th here, but it was not adopted, because that would mean choosing on the test windows. |

## What the embeddings learned (D091)
The model was given only type labels. The learned type vectors still group types by how they sell:
- accessories together;
- knitwear and loungewear together;
- denim and bottoms together;
- tops together.

Dresses and swimwear stand apart, because their seasonality is their own.

## How to reproduce the candidate
1. **NN backtest, two seed sets.** `python eda/57_embed_nn.py seeds=3 inter=0 seq=1 tag=_S13`, then the same with `seed0=10 tag=_S13s10`. The 6-seed file is their average (`data/nn_backtest_preds_S13x6.parquet`).
2. **NN for the June 2026 origin.** The same command with `origins=2026-06`, tags `_finalS` and `_finalS_s10`.
3. **Candidate forecast.** `python eda/65_candidate_nn.py nn_backtest_preds_S13x6 nn_backtest_preds_finalS nn_backtest_preds_finalS_s10` runs eda/22 → 32 → 41 and writes `data/cand_nn_final_forecast_q4.csv`.
4. **Candidate workbook and checks.** `python eda/68_fill_candidate.py`.
5. **Buying simulation.** `python eda/59_sim_v2.py nn=nn_backtest_preds_S13x6`; for the rules replay, `python eda/66_rules_replay_nn.py <origin>` and then `summary`.

## Caveats
- **Gains in unit accuracy are modest.** The candidate is +8% on the total, a material change built on about one point of WAPE.
- **The Sep–Dec switch was chosen after seeing the windows.** No earlier peak season has backtest forecasts to confirm it.
- **Money assumptions are unchanged:** price 1, cost 0.40, salvage 0.20.
- **Compute:** GPU runs used the shared T4 agreed with the all nighter session (`~/m5`, at most one process, under 1 GB). Nothing of theirs was touched.
