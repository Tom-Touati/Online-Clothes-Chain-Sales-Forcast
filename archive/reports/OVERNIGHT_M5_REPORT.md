# Overnight: the M5 lessons, tested (16–17 Nov 2026)

**Goal set by the user:** implement every M5-derived suggestion, with several versions each, plus augmentation, aiming to halve the error. In WAPE ÷ R3 on the nine recent origins that means going from about 0.41 to about 0.20.

**Systems used:** the laptop and the GPU server (Tesla T4). The fork session worked alongside on gradient-boosting diversity, tuning, bagging and TabPFN (its D205+); `SYNC.md` records the split.

Decisions are D170–D183 in `log.md`. Every number below is WAPE ÷ R3, on the same item-origins as the scoreboard (below 1 beats recency).

## Where we started

The best production-usable system is **p2**: rules + the dynamic co-occurrence quantile model + the NN + TabPFN v2, combined with equal weights. It scores **0.4100** recent and 0.4586 on all 13 origins, with a buying loss of 10.78%.

## Results by direction

| # | Direction | Versions tried | Best result | Verdict |
|---|---|---|---|---|
| 1 | Hierarchical reconciliation, item / style / type (D172) | top-down; partial 0.3 / 0.5 / 0.7; MinT; three levels | p2 partial 0.3: 0.4094 (level) | The style-level model is weaker than the summed items (style WAPE 0.292 vs 0.273). It helps only a weaker base (D159 three-way 0.4234 → 0.416). |
| 2 | Planned campaign calendar (D173) | site intensity; + item discount response; + LLM interactions | worse (three-way +0.001 to +0.003) | The events calendar already carries the big campaigns; site discount levels drift. |
| 3 | Stock-out-corrected inputs and targets (D174) | availability features; demand-corrected level; censored targets down-weighted | quantile alone 0.4320 vs 0.4355; all 13 origins 0.500 vs 0.511; pinball better | A small, real gain for the quantile model, but level inside p2. |
| 4 | Learned combination weights, FFORMA (D175) | best-member classifier; direct-loss network; shrunk | worse (0.4147 / 0.4400 / 0.4201) | Which member is right is not stable over time. |
| 5 | Weekly origins (D179); mid-month origins in the main model (D184) | weekly pipeline: month / every 2 weeks / every week; main model + mid-month rows at weight 1 / 0.5 | weekly pipeline 0.585 → 0.521 with biweekly cutoffs; main model three-way 0.4219 vs 0.4234 (mid rows at half weight) | More origins help a little on recent origins, not on all 13. Inside p2, marginal. |
| 6a | Product photos and text, public feed (D170, D171, D180) | CLIP image, mpnet text, CLIP text; gate per the user's rule | Cart gate, future carts: v6 = v4 + photos + text, rho 0.281 / precision 0.223, against raw product features 0.159 / 0.142 and v4 0.273 / 0.206 | Significantly better embedding. Photos alone are a strong naive baseline. As tree features the fork found no gain. |
| 6b | Google Trends, 74 terms (D181, D183) | momentum / year-on-year; + search ramp; + interactions | worse (three-way +0.0001 to +0.0022) | US-level search for the looks does not lead item sales at 4 months. |
| 7 | Neural multi-horizon models on weekly series (D177) | NHITS, TFT, DeepAR; + style embedding; + log target | TFT + embedding 0.547 alone; every one hurts p2 | As in M5, boosted trees on engineered features beat the global networks. |
| 8 | Augmentation (D178) | style / type aggregate rows; scale jitter; origin bagging; mixup | all worse alone (0.442–0.460 vs 0.4355); three-way level or worse | The limit is information, not sample size or scale coverage. |
| 9 | Anything hierarchical, deeper (D185, D186) | proper style features; member stacking at style level; style residual model; learned within-style shares; MinT over item / style / type; partial pooling toward style / type | best: style mix 0.4128, pooling toward style 0.4100 (level) | No style model beats summing p2's items (style WAPE 0.273 vs 0.298 / 0.315 / 0.317). The hierarchy holds no extra signal. |

## What moved the error tonight

- **Gains in the backtest came from adding strong, different learners, not from new features or tricks.**
  - The fork's TabPFN v2 with co-occurrence features (32 estimators) and a pruned LightGBM.
  - Together they give the new accuracy champion **sel_d32: 0.4037 recent, 0.4458 on all 13 origins** (p2: 0.4100 / 0.4586). Its buying loss is 11.05%.
  - The buying champion stays **p2 at 10.78%**. Both are production-licensable.
- **Small real gains:**
  - the stock-out-corrected level (quantile model 0.4355 → 0.4320; all 13 origins 0.511 → 0.500);
  - mid-month training rows at half weight (three-way 0.4234 → 0.4219).
  - Neither moves p2.
- **The best content embedding is v6** (v4 + product photos + text). It predicts carts significantly better than every naive baseline, and is the best similarity for cold-start items.

## Why halving is out of reach with this data

- **Halving would need about 0.20 × R3; the best is 0.40.** Every direction tested tonight moved the error by 0.000–0.006, or made it worse.
- **The remaining error is about which looks sell, not single styles** (D114, corrected by D189).
  - If each style's true total were known, raw WAPE would fall from 0.27 to about 0.06.
  - An item's error moves with its neighbours' errors in the same window: correlation +0.41 for cart and visual neighbours, 0.00 for random items.
  - But nothing in the past months predicts that shared shock: neighbours' recent surprises correlate 0.0 to 0.07.
  - Launches in the same look explain part of it (+0.15): items whose look gets 4 or more new launches come in about 6% over the forecast, those with none about 15% under.
- **No historical, hierarchical, external-search, campaign or content signal tried here predicts those style-level surprises:**
  - the style-level models lose to the summed items;
  - Trends and campaigns hurt;
  - photos help similarity but not the forecast;
  - neural nets and augmentation add noise.
- **What would move it is information the data doesn't contain.** The launch plan, with launches placed into looks by their photos (D189), is the first piece. Others: the buyers' launch and marketing plan (D144's halo is one example), early demand signals (sell-through in the first days of a launch, wishlists, add-to-cart counts, traffic), and planned stock levels.

## The final 2026 forecast (D237)

`Edikted_Forecasting_Exercise_candidate_final.xlsx` is a new file; the submitted workbook is unchanged.

**Point forecast:** the arithmetic mean of five members: the rule agent, the dynamic co-occurrence LightGBM quantile model, the embedding NN, conformalised TabPFN v2 and TabPFN v2 with co-occurrence features. In the backtest it scores 0.398 × R3's error on the recent origins and loses 10.74% of profit in the buying simulation.

**Total: 629,372 units** (Jul 177,820 · Aug 188,423 · Sep 128,951 · Oct 134,178), +8.3% against the submitted forecast.

**Buying quantity:** the same stack with the volatile members at their 65th percentile. It is the bottom of the newsvendor ladder, losing 10.38% in the simulation. Total 659,950.

**Cascade over the 100 targets:**

| Layer | Units |
|---|---|
| R3 (recent rate × 4) | 473,374 |
| × product-type seasonality | 593,665 |
| Rule agent | 620,269 |
| Dynamic quantile model | 633,461 |
| Embedding NN | 698,591 |
| Conformal TabPFN | 571,510 |
| TabPFN with co-occurrence features | 622,443 |
| **Forecast (average)** | **629,385** |
| Buying quantity | 659,950 |

The workbook's "Cascade" sheets show the same layers per item and rolled up by style and product type.

**Confidence:** 20 targets at level 1, 65 at level 2, 15 at level 3.

**Explanations:** written for 18 targets, built from the cascade plus SHAP categories of the tree model.

**Alternative:** the most accurate stack, qsc32a, totals 620,365 (0.396 × R3, but 6.6% under and 10.93% buying loss).
