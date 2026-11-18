# Error analysis, probes and fixes (night of 15–16 Nov 2026)

Decisions D114–D139 in `log.md`. All numbers are on the 13 backtest origins (Feb 2025 – Feb 2026), the same items throughout, 4-month item totals unless stated. "Recent" = the nine origins Jun 2025 – Feb 2026.

**How every system is scored:** as a ratio to R3 (each item's last 3 months × 4); below 1 beats recency. **R3 + rules** is the rule-agent system, shown as its own row. **Typical item** = the geometric mean of the per-item error ratio.

The reference system is **rules + quantile + NN** (equal-weight geometric mean). On the recent origins it scores WAPE ÷ R3 0.440, typical item 0.452, pinball ÷ R3 0.415, and buying-simulation loss 0.61× R3's.

Figures are in `eda/figs/`: `err_under.png`, `err_over.png`, `err_consensus.png`, `err_structure.png`, `err_locations.png`.

## 1. Where the error is

| Finding | Evidence |
|---|---|
| Almost all error is about which styles sell. The site level and the product-group mix matter little. | Oracle rescaling: site total known 0.307 → 0.306; type totals known → 0.258; style totals known → 0.056 (D114) |
| Concentrated | 7% of item-origins carry half of the error; 22% carry 80% |
| **Black Friday windows are the largest mass** | 40% of item-origins, 51% of units, **57% of the error**; bias −12.8%; WAPE ÷ R3 0.577, against 0.335 outside (D123) |
| November 2025 under-forecast by every member | −17% to −32%, worst one and two months ahead. Last year's like-for-like Nov/Oct was 1.56; this year's 1.81 |
| Average Black Friday lift stable, item lift not | Median 5.7×, same as last year; an item's own lift repeats with r = 0.18 |
| Seasonal turns of strongly seasonal groups | Warm items −35% to −48% entering winter, +15% to +26% leaving it. Swimwear's own last-year pattern called its turns; the models did not (D128) |
| Young launches | 12% of item-origins, 21% of the error, bias −18.5% (−26% in Black Friday windows) |
| Spikes at the origin are *not* a systematic problem | Bias −1.6%, WAPE ÷ R3 0.373 |
| Member disagreement predicts error size | Lowest-disagreement fifth 0.32 ÷ R3, highest 0.64. The actual lands mid-way, so equal weights are right on average |
| Consensus misses | Where the members agree, 26% of items are still off by more than 50% (17% of all error), mostly Black Friday spikes (D137) |
| Catalogue growth | Launches take 6–16% of each window but do not dilute existing items' forecasts (D116) |
| **Launches lift their siblings** | When a new colour of the style launches in the window, the existing colours are under-forecast by about 20% (−5% otherwise). A time-honest uplift is stable at 1.08–1.14. Usable if the launch plan is known (D144) |

## 2. What each kind of information is worth

**Stage by stage from R3** (recent, WAPE ÷ R3): R3 1 → × type seasonality 0.62 → quantile model 0.47 → + NN 0.457 → + rules 0.440 → + lean model with occasion timing 0.436 → + Black Friday fix 0.433 (D118, D141; figure `eda/figs/value_ladder.png`).

**Time:** the model needs at least one full year of origins to learn the seasons. Training on the last 6 origins scores 0.745, the last 12 is level with all of history, and down-weighting older origins is worse (D139).

**Stability:** between consecutive origins, the NN revises an item-month least (median 17%), the combination 20% and the rules most (26%) (D142).

**Removing one feature group from the quantile model** (base 0.471; D134):

| Group | Change when removed |
|---|---|
| Seasonality | **+0.166** |
| Recent momentum | +0.044 |
| Product type and attributes | +0.040 |
| Size | +0.008 |
| Channel mix, discount, occasion flags | about 0 |
| Lifecycle (age, days live, colour number) | −0.003 |

## 3. Fixes that worked

| Fix | Result (recent unless stated) |
|---|---|
| **Lean quantile model + occasion timing** (drop the zero-value groups, add days to the item's occasion) | Alone 0.471 → 0.465; three-way 0.4397 → 0.4365 (significant); targets 0.428 → 0.421; buying loss 11.84% → 11.60% (9 of 11 windows). 2026 candidate built: `data/cand_rlos_final_forecast_q3.csv`, 615,140 (D134–D135) |
| **Rules + NN monthly split, no own-shape mix** | Shape error 0.240 → 0.177 (−26%). Removes the July under-allocation that the July calibration patched. Applied in `data/cand_rnns_final_forecast_q3.csv` (D120–D121) |
| **Partly recursive monthly trees** | Monthly WAPE 0.406 → 0.386 at every horizon; totals −0.008 to −0.019; in the combination level on recent origins, better on all 13 and on the targets (D131) |
| **Black Friday: November level from last year + item-lift allocation** | An item's lift is partly predictable from October data (r = 0.42; size, type, TikTok share, discount). Black Friday origins 0.577 → 0.567 (significant); recent 0.4354; November item-month WAPE 0.412 → 0.387. Buying simulation: Sep–Nov windows lose 12–13% instead of 15%; mean 11.84% → 11.43%. The August window looks worse only because that simulation stops on the eve of Black Friday, so stock bought for it is written down unsold (D141c). For any window containing Black Friday; not the Jul–Oct deliverable (D136–D137c) |
| **Both stacked** (lean model + Black Friday fix) | Recent 0.4397 → **0.4333** (significant); MAE + abs bias 0.344 → 0.328; bias −8.2% → −4.5%; targets 0.428 → 0.414. Buying simulation 11.84% → **11.20%** loss (8 of 11 windows; peaks 15% → 12%). The best backtest system measured (D141, D141b) |
| **Binned conformal ranges** (by disagreement × model IQR) | Pinball −2.3%, interval score −6.5%, coverage on target, ranges 15% narrower. For reporting, not safety stock (it under-states peak risk). 2026 quantiles: `data/cand_rnn_conformal_2026.csv` (D117, D133) |
| **Year-on-year rule (D052)** | Validated: helps the quantile model significantly; on the three-way level on WAPE and better on the typical item (D127) |

## 4. Probes that did not work (and why)

- **Time-honest combination weights, stacking, segment and group bias corrections (D115, D138):** which member is right, and each group's bias, flip over time. Untuned equal weights are the robust choice.
- **Last-year calibration of every month (D124):** removes bias but costs WAPE. Last-year ratios run high for this year's older items. Calibrating only the event month works (above).
- **Bottom-up per location, and per location × month (D125–D126):** worse, even with expected-value (Tweedie) per location. TikTok and new dropship partners ramped and store assortments turned over; those errors don't cancel in the sum. At item level they average out.
- **Retail rollout features (D122b):** store counts trend over time, so trees learn relationships that don't hold later.
- **Matryoshka hierarchical embeddings as tree features (D113):** leakage (+0.05 to +0.07). Out-of-fold by style fixes it but adds nothing. Their value stays inside the NN.
- **Discount momentum (D112), analog launch curves (D129), style-level features (D145):** all worse or level.
- **TabDPT foundation model on CPU (D130, D130b):** 0.530 with a 2,048-row context, 0.515 with 4,096, against 0.471. More context helps, but its median runs about 21% low. TabPFN was not tried: its licence needs your sign-in.
- **Training-window length and recency weighting (D139):** fewer than 12 origins loses the seasons; down-weighting old data is worse.
- **Young items' momentum (D140):** the members damp young items' pace both ways (falling-fast launches +10%, rising −20%). A pace-extrapolation member is far worse, because raw pace is too noisy; the damping is right on average.

## 5. For the Jul–Oct 2026 deliverable

- **Candidates, totals:** submitted 580,968 · NN candidate 627,253 · rules + quantile + NN 615,937 · the same with the rules + NN split 616,037 · lean model version 615,140.
- **Choice of candidate:** the last three differ by at most 7.7% on any target. The split matters more than the model change: July 27.8% and August 29.9% of the four months, against 29.0% / 29.0% before.
- **No event-month warning:** the candidate's monthly ratios sit 9–15% under last year's like-for-like ratios, which is the normal ageing gap.
- **The June 2025 analogue's misses don't repeat** (D146):
  - That origin under-forecast Halloween items by 33% and over-forecast winter hoodies.
  - In the candidate, the 14 Halloween targets total 7.56× June, against 7.02× for all Halloween items in 2025.
  - Warm targets total 8.37×, against 8.85× in 2025.
- **Ranges:** conformal P10–P90 for every target are in `data/cand_rnn_conformal_2026.csv` (median P90/P10 3.2×).
- **Items worth a buyer's look:** 18 targets where the members are more than 2× apart (83k units), listed with their drivers in `data/review_list_2026.csv` (D143).
- **Unchanged:** the submitted workbook. Switching is your decision.
