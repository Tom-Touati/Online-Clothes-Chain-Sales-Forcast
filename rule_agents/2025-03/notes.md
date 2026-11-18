# Forecast notes: Apr–Jul 2025 per style_color (snapshot 2025-03-31)

`python rules.py` reads `data/` and writes `predictions.csv` (654 items × 4 months, ~2 s).
Research scripts are in `research/`: `r1_review.py` (scoreboard review + site/channel/store exploration),
`r2_explore.py` (new items, stores, channels, factors for this origin), `r3_retail_july.py` (retail,
FSM opening, July-24 web jump), `bt1.py` (backtests of variants). `rules_v1.py` is a copy of last month's rules.

## 0. Review of v1 (scoreboard: origin 2025-02, March-25 scored, 1 month only)

- Overall: pred 392k vs actual 411k (bias −4.5%), item WAPE 0.256. Web level was about right
  (rules assumed web/day ×0.87; actual Feb→Mar LFL web/day 0.84). The miss came from **retail**
  (assumed ×1.03, actual LFL retail/day ×1.49) and was partly offset by "other" (assumed 1.0, actual 0.84).
- **Young items under-forecast (systematic).** Jan-25 launches: actual/pred 1.79 (n = 52); Dec-24
  launches 1.17; Oct–Nov 1.07; 12+ months 1.02. The in-data backtest (bt1, 7 oracle origins in 2024 +
  this origin) shows the same every time: relative to old items, actual/pred was 1.0–2.9 for items launched
  inside the base month, 1.0–2.15 (median ~1.4) for launches in the month before, 1.0–1.4 for two months before.
- Retail-heavy items (web share 0.3–0.5) were under by 19%, which matches the retail miss.
- Season groups: WINTER over (actual/pred 0.89: Hoodie 0.80, Cardigan 0.75, Sweater 0.89), SUMMER under
  (1.14: Tank Top 1.26), CORE under 1.09 (Corset 1.33, Top 1.23). This is one month and partly the overall
  level miss; shrinkage K variants pointed in opposite directions (see 3), so no change.
- Small items under, big items over: pred quintiles actual/pred 1.34 / 1.19 / 1.17 / 1.18 / 0.99. The same
  shape appears at every backtest origin (top quintile 0.87–0.98 relative). Not fixed (see 3, POW).
- Occasion tags: St. Patrick's (n = 8) ×2.2 under in March; this was a holiday-month spike, not relevant now.
  Other tags were within ±20%.

## 1. Key findings (numbers from the data)

**The panel.** 654 items to forecast: 26 launched in Feb-25, 4 in Mar-25, 51 in Jan-25. March-25 these items
sold 428k (web 295k, retail 72k, other 60k).

**March-25 site level** (per day): units 13.9k (Feb 14.5k), web 9.7k (Feb 10.8k) at web discount 0.57 (Feb 0.61),
retail 2.33k (Feb 1.42k), dropship 1.18k (Feb 1.04k). TikTok is falling: 31.9k / 28.1k / 19.9k in Jan / Feb / Mar,
but Nordstrom dropship jumped to 660–840/day in the last week; "other" in the last week (~2.1k/day) is at or
above the March mean, so "other" stays flat.

**Discount drives daily web volume** (again). Late March: discount 0.61 → 12.9k web/day; 0.56 → 10–11k;
0.48–0.52 → 3.5–4.9k. The weekly mean discount slid 0.61 → 0.57 → 0.53 (week to Mar-30) and was 0.56 on Mar-31.

**Retail is the new swing factor.**
- A new store, FSM, opened in March (first real day Mar-18, opening-day spike 1.9k on Mar-22, then
  ~450–600/day; median of last 14 days ≈ 490/day). Its March total (8.7k) understates a full month (~15k).
- Existing stores also jumped: SoHo flagship 465/day in Feb → 775/day in March (late March ~860–890/day),
  MN and LA Grove up 40–100%. Retail discount fell (0.09 → 0.06), so it is not promo-driven.
- In 2024 SoHo went 244 / 268 / 280 / 344 / 592 / 706 per day (Feb–Jul). Feb-24 was a retail promo month
  (retail discount 0.38 vs 0.18 in Mar-24), so the Feb-24 → Mar-24 retail move understates spring seasonality.
  The Mar-24-anchored deflated flagship factors are Apr 0.90, May 1.04, Jun 1.61, Jul 1.78.

**July 2024 is a regime shift, not a season.** LFL web units/day vs June-24 by week: 0.80–1.33 in June;
1.70 in the July-4 week (discount 0.55); **1.11 / 1.14 on Jul 8–21 (discount 0.45, same as June's 0.46)**;
then 2.25–2.32 from Jul 22 to mid-August (discount 0.56–0.57). The web discount never went back:
0.50–0.52 in Aug–Oct, 0.56–0.61 since November. Web/day stayed 8–11k from Aug-24 on vs ~4k in spring 2024.
2025 is already in the deep-discount regime, so the raw Mar→Jul-24 web factor (×1.49 per day) would not recur.
The discount-matched Jul 8–21 lift over June is 1.12.

**The July mix shift is seasonal and kept.** Relative to June shares, Jul 1–21 (before the regime jump) already shows
Jeans ×1.24, WINTER ×1.36, Dress ×0.70, SUMMER ×0.89; August continues it (Jeans 1.77, WINTER 1.96). This is
back-to-school/fall transition, so July relative type factors from all of July-24 are kept.

**Seasonal profile for this origin** (Mar-24 → m, relative to panel): SUMMER 1.15 / 1.24 / 1.30 / 1.11,
WINTER 0.67 / 0.39 / 0.32 / 0.44, CORE 0.98 / 1.00 / 0.96 / 1.11. Shorts 1.54 / 1.93 / 2.67 / 2.60 (n = 11),
Hoodie 0.52 / 0.25 / 0.19 / 0.27 (n = 5, shrunk to group), Jeans 1.03 / 0.94 / 0.85 / 1.16 (n = 29).

Older findings still valid (from v1 notes): items are flat after month 2 (no lifecycle decay for survivors);
discount at item level does not explain residuals; the 4th of July tag overlaps summer types; Easter moves
from Mar-31 (2024) to Apr-20 (2025), ignored.

## 2. Rules (rules.py, v2)

`pred(i,m) = base_rate_i × days(m) × level_i(m) × season_seg(m) × ramp_i(h)`

1. **Base rate (units/day)** — unchanged: 0.6 × last-month + 0.4 × last-14-day rate rescaled to the last-month
   total, then 80/20 with the rescaled last-91-day rate. Live days only; zero if nothing sold in 28 days.
2. **Channel level per item.** The item's base-month mix (web / retail / other), shrunk to the panel with 20
   pseudo-units, weights per-channel per-day factors from the same months of 2024:
   - **web**: 0.6 × base-month-anchored LFL chain + 0.4 × (month-before anchored chain ÷ this year's
     per-day move). **New:** the Jun→Jul-24 link of the chain is replaced by the discount-matched Jul 8–21 lift.
     Result: 0.94 / 0.94 / 0.88 / 0.99 (raw July would have been 1.43).
   - **retail**: flagship per day, deflated for panel growth (with the raw web chain). **New:** two anchors,
     0.75 × Mar-24-anchored + 0.25 × Feb-24-anchored ÷ this year's Feb→Mar retail LFL (×1.47).
     **New:** stores opened inside the base month count at their run-rate (median day of last 14 days × days)
     → ×1.089. Result: 0.91 / 1.05 / 1.62 / 1.80.
   - **other**: flat 1.0.
3. **Seasonal profile** — unchanged: segment = product_type (long-sleeve tops split), chained LFL relative to
   panel, shrunk to WINTER / SUMMER / CORE with K = 8, capped to [0.15, 4].
4. **Ramp for young items — stronger and graded (new).** Age class at the base month:
   launched in it ×1.6, the month before ×1.4, two months before ×1.15; the first two classes get ×1.1 more
   from h = 2 on. (v1: 1.25 / 1.25 / 1.1, ×1.1.)

Forecast totals (654 items):

| Month | Units |
|---|---|
| Apr-25 | 391.5k |
| May-25 | 398.4k |
| Jun-25 | 420.7k |
| Jul-25 | 485.5k |
| 4-month total | 1.70M |

For comparison these items sold 428k in March-25 (31 days). By group, March → Apr / May / Jun / Jul:
SUMMER 188k → 205 / 237 / 270 / 273k; WINTER 116k → 73 / 44 / 37 / 59k; CORE 123k → 113 / 118 / 114 / 153k.

## 3. Backtests (research/bt1.py)

A. **Honest out-of-time**: origin 2025-02-28 using only data up to then, March-25 scored (items selling in Jan
or Feb). B. **Oracle-level 2024**: origins May–Nov 2024, factors measured in-sample on the target months
(all items), 4-month totals; mean over 7 origins. `wape_sc` = after scaling to the actual total.

| Variant | A WAPE | A bias | B WAPE | B wape_sc | B ±50% |
|---|---|---|---|---|---|
| v1-like (no retail changes, ramp 1.25/1.1) | 0.265 | −6.2% | 0.328 | 0.333 | 0.70 |
| v2 retail changes, old ramp | 0.264 | −4.7% | 0.328 | 0.333 | 0.70 |
| **v2 + graded ramp 1.6/1.4/1.15 (chosen)** | **0.261** | −4.0% | **0.326** | 0.326 | 0.69 |
| graded ramp 1.8/1.5/1.2 | 0.260 | −3.4% | 0.328 | 0.324 | 0.69 |
| graded ramp 2.0/1.6/1.25, ×1.15 | 0.260 | −2.8% | 0.333 | 0.323 | 0.68 |
| no ramp | 0.271 | −5.9% | 0.336 | 0.354 | 0.71 |
| K = 4 / K = 12 | 0.276 / 0.259 | | 0.321 / 0.333 | | |
| WLONG 0.1 / 0.3 | 0.258 / 0.273 | | 0.330 / 0.328 | | |
| graded + POW 0.05 (regression to mean) | 0.255 | | 0.328 | 0.326 | 0.65 |
| graded + POW 0.10 | 0.257 | | 0.340 | 0.338 | 0.59 |

- The graded ramp is the only change that improves both tests on unscaled WAPE; stronger ramps help scaled
  WAPE but hurt unscaled in B and lower the ±50% share.
- K and WLONG trade off between the two tests, so they stay unchanged.
- POW 0.05 helps the 1-month test but costs 4–5 points of the ±50% share; not used (parameter left at 0).
- The oracle tests have a −7% to −9% bias. That is expected: "other" was onboarding in 2024 (other LFL/day
  ×1.6 in Apr-24) and the regime fix removes the true July-24 jump from the May/Jun-24 origins.
- The retail changes cannot be tested on 2024 (no prior year, no new store in a base month with history);
  their only test is A, where they cut bias from −6.2% to −4.7%.

## 4. Known weaknesses

- **Web level is still the main risk.** It hinges on the discount: late-March days at 0.48–0.52 ran 3.5–5k
  web/day vs 10–13k at 0.56–0.61. If the late-March pullback becomes the norm, web could run 20–30% below the
  forecast. Every 0.1 on the web factor ≈ 7% of total units.
- **July** carries a judgment call: the web move uses the discount-matched mid-July lift (1.12 × June). If a
  July-4/summer sale drives a 2024-like jump, July will be under; if not, the back-to-school mix still lifts
  Jeans/winter. Future planners: the Aug-24 link (Jul→Aug) is also distorted by the regime — the code
  carries later months forward from the corrected July, check it when August becomes a target.
- **Retail**: June/July ×1.6/×1.8 assumes the 2024 SoHo summer tourist peak recurs and applies it to all stores
  (LA, MN, CA, FSM have no summer history; CA/FSM are tourist-light). The FSM run-rate may fade after opening.
- Small items remain under-forecast and the top quintile ~5–10% over (systematic in every backtest); no rule
  found that helps WAPE without hurting the ±50% share.
- One month of feedback: winter types fell faster than forecast in March and summer/core tops ran ahead.
  Not acted on; watch next month.
- Seasonal profiles rest on one year and small segments (Shorts n = 11, Hoodie n = 5).
- Store openings or new marketplaces after March are not anticipated (upside risk); TikTok decline is not
  modelled (it is offset by Nordstrom growth in the last week).
- Sibling colours, occasion tags and item-level discount are not used. Easter shift ignored.

## Changelog
- **v1 (2025-02-28 snapshot)**: first version. Base blend, channel-weighted level (two-anchor web, deflated
  flagship retail, flat other), product-type × sleeve seasonal profile shrunk to season group, young-item ramp,
  zero for items with no sales in 28 days.
- **v2 (2025-03-31 snapshot)**:
  1. *Web July fix*: the Jun→Jul-24 link of the web LFL chain is replaced by the discount-matched Jul 8–21 lift
     (1.12). Evidence: the late-July-24 ×2.3 jump coincides with the permanent move to deep discounts
     (0.45 → 0.57), which 2025 already has. Raw factor 1.43 → 0.99 for July.
  2. *Retail two-anchor* (0.25 weight on the Feb-24 anchor ÷ this year's Feb→Mar retail move) — hedges against
     March-25 retail (+47% LFL/day) being a spike; Feb-24 was a retail promo month, so weight is low.
     Retail deflation now uses the raw web chain, so the July fix does not leak into retail.
  3. *New-store run-rate*: stores opened inside the base month (FSM, March) count at their median daily
     rate over the last 14 days for a full month (×1.089 on retail). Evidence: v1 retail miss (×1.49 actual vs
     ×1.03 assumed) and FSM selling ~490/day vs 281/day averaged over March.
  4. *Graded, stronger ramp* (1.6 / 1.4 / 1.15, ×1.1 from h = 2): young items under-forecast in the
     scoreboard (Jan launches ×1.79) and at every backtest origin; improves both backtests (table in 3).
  5. Parameters moved into one dict `P`; POW (regression to the mean) added but switched off.
