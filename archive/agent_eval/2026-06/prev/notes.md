# Forecast notes: Mar-26 – Jun-26 per style_color (snapshot 2026-02-28), rules v13

`python rules.py` reads `data/` and writes `predictions.csv` (1262 items × 4 months, ~2 s, 402 lines).
Research scripts in `research/`: `r1_review.py` (scoreboard Feb-26 by group / age / tier / type / tag / channel share;
4-month totals by origin), `r2_site.py` (site / channel per month, Feb aligned YoY, LY weekly Feb–Jul, Feb daily,
channel accounts), `r3_groups.py` (group LFL moves TY vs LY; factors from the Feb-26 base), `r4_bt.py` (oracle-site
backtest harness, origins Feb-25..Jan-26 with the scoreboard item lists; `run(name, overrides)`), `r5_tags.py` /
`r5b_tags_ty.py` (tag / tier residuals vs a segment oracle from Jan-25 / Feb-25 bases, and TY Dec→Feb),
`r6_diag.py` (h=1 type residuals over origins, young-item share, launches), `r7_types.py` (backtest residuals by
type per origin), `r8_bt2.py` (variant runs), `r9_fc.py` (forecast summary).

Convention: **act/pred** = actual ÷ prediction, level-neutral (relative to the month's overall ratio) unless "raw".

## 0. Review of v12 (origin 2026-01) and older origins

- **Feb-26 at h=1: v12 557.8k vs actual 542.8k (1176 items), bias +2.8%, WAPE 0.221** (best h=1 so far). Older
  origins for Feb: v11 (h=2) −0.6% / 0.305, v10 (h=3) +9.6% / 0.378, v9 (h=4) −17% / 0.379.
- **Site plan vs actual**: web 13.83k/day vs 14.5k planned (−4.6%; YoY 1.28 against LY's deep-promo February),
  other 4.18k vs 3.3k (**+27%, second month in a row**: TikTok 3.05k/day and still rising), retail 2.50k. Under
  site actuals the item logic was **+3.8% high at h=1** (oracle backtest; Nov / Dec origins +4.6% / +2.7%) — the
  recent origins run high while 2025's ran low (−7% mean). Not acted on (DRIFT_ADJ 0.98: WAPE worse, §3).
- Segments Feb h=1: **WINTER 1.12 / SUMMER 0.97 / CORE 0.86** (Sweatpants 1.21, Hoodie 1.16, Cardigan 1.24, Short
  Jeans 1.28; Jeans 0.81, Pants 0.83, Bracelet 0.76, Skirt 0.89). Second month with WINTER under (Jan 1.15): group
  anchoring (0.75) bet that the Dec→Jan gap would close; TY Jan→Feb WINTER moved 0.895 vs LY 0.855, i.e. like LY,
  so the gap stayed. CORE's anchoring (+9%) failed the same way.
- Tags h=1 (level-neutral): St. Patrick 1.42 (13 items), 4th of July 1.07, Homecoming 1.08, NYE 1.06, Graduation
  1.04, Halloween 0.96, Valentine's 0.96. Tiers: items without a tier 0.80 (88), Resort-Festival 1.24, Loungewear
  1.15, Basics 1.05, Evening 0.94, Party 1.03. Age bands 0.96–1.07 (young items fine). Retail share > 50% 0.86 (32),
  items without dropship / TikTok sales 0.79 (133 small items).
- **Type residuals across origins (r6 / r7)**: Skirt over-forecast since Jul-25 (0.72–0.90) but UNDER in the spring
  2025 origins (1.07–1.32) → seasonal, no rule. Pants swung from 0.73 to 1.87 and back → noise. Sweatpants under in
  10 of 12 oracle origins (mean ~1.12) → a type-trend rule was tested (§3) and gained nothing.
- 4-month WAPE by origin (scoreboard, live plans) 0.30–0.40; Oct-25 origin (Nov–Feb) 0.322, bias −19% (BF plan).

## 1. Key findings (new month)

**Site Feb** (per day): web 13.83k (disc 0.608 vs LY 0.606), other 4.18k, retail 2.50k. Web YoY by month Sep–Feb
1.42 / 1.60 / 1.65 / 1.37 / 1.47 / 1.28; aligned 2-week YoY Feb 1–14 1.20, Feb 8–21 1.22, Feb 15–28 1.41 (LY dipped
to 8.3k/day in the week to Mar 1). TY web has been flat at ~13.6–14.1k/day since late January with a strong
Sun/Mon pattern (20–22k on drop days); discount fell to 0.57 on Feb 27–28. LY Mar 9.66k (weeks 7.1–11.5k), Apr 7.9k
(Easter Apr 20), May 9.1k (Memorial Day week 10.5–12.4k), Jun 8.8k (mid-June week 11.5k).
**Other by account** (per day): TikTok Sep→Feb 1.27 / 1.50 / 2.30 / 2.38 / 2.55 / 3.05k (LY fell 1.0k → 0.53k
Feb→Apr); wholesale 1.13k (Macy's 328, Nordstrom 281, Pacsun 242, Bloomingdale's 120, Urban 120, Tillys 41) vs LY
Mar–Jun 1.06–1.25k. Feb 15–28 other 4.41k/day.
**Group moves** (LFL rel. to panel): TY Jan→Feb WINTER 0.895 / SUMMER 1.21 / CORE 0.98 vs LY 0.855 / 1.149 / 1.058.
From the Feb base: amplitude 0.7 (floor), anchoring WINTER 0.88 / SUMMER 1.09 / CORE 1.09; LY rel. profile Mar–Jun
WINTER 0.75 / 0.52 / 0.42 / 0.38, SUMMER 1.28 / 1.53 / 1.66 / 1.69. LY drift (panel vs site web) 0.94 / 0.86 /
0.79 / 0.72; retail factor 1.23 / 1.14 / 0.94 / 1.15.
**Tags / tiers from a Feb-25 base** (segment oracle, raw Mar / Apr / May / Jun in the rules.py comments); Jun-25:
4th of July 1.14, Graduation 0.74, Homecoming 0.51, Halloween 0.89, NYE 0.78, Valentine's 0.88, Party 0.78, Evening
0.60, Resort-Festival 0.96, Loungewear 0.82, Basics 1.30. The Jan-25 base gives the same shapes (except my Evening
read 0.74–0.85 vs v12's 1.07–1.12). TY Jan→Feb-26 tags follow LY's shapes (Graduation 1.17 vs 1.28, Halloween 0.98
vs 1.00, Valentine's 0.95 vs 0.98); TY Evening held (0.98 vs LY 0.85).
**Young items**: Jan-26 had 106 launches (2× normal), Feb 38; young-item (< 60 days) unit share Feb-26 0.086 vs LY
0.080.

## 2. Rules (rules.py, v13)

`pred(i,m) = base_rate_i × days(m) × level_i(m) × season_seg(m) × ramp_i(h) × fade_i(h) × occasion_i(m)`
(unchanged structure; see the docstring).

1. **Site plan** (`SITE_WEB_PD`): Feb actual 13,827; **Mar 12.8k** (YoY 1.33), **Apr 10.9k** (1.38, weak LY April /
   Easter shift), **May 12.0k** (1.32), **Jun 11.5k** (1.30, new). Below v12's 13.15 / 11.2 / 12.4k after Feb came in
   4.6% under plan; YoY between the Feb monthly 1.28 and the late-Feb aligned 1.41, drifting down as TY laps LY's
   spring growth.
2. **Other plan** (`OTHER_PD`): Feb actual 4,184; **Mar 4.2k, Apr 3.95k, May 3.95k, Jun 3.8k** (v12: 3.1 / 2.95 /
   3.15k). Two consecutive +27% misses from applying LY's spring decline to a TikTok business rising against it →
   now near the Feb level with a mild decline only.
3. **Occasion / tier indices: Mar–Jun re-measured from a Feb-25 base, June entries added** (v12 had none → June
   reverted to 1.0: Halloween ×1.16, Valentine's ×1.06, NYE ×0.85, Graduation ×1.16, Loungewear ×1.09,
   Resort-Festival ×0.81 from Feb). 4th of July Jun 1.3 → 1.10 (LY June lift ~1.14 vs segment, from both bases);
   Graduation Jun 0.75; Evening Mar 1.187 → 1.14 (between LY's Feb-base fall and TY's Feb hold).
4. **W14 0.4 → 0.5** (backtest §3).
5. Unchanged: WLONG 0.2, flash-day cap, ramps, AGE_FADE 0.93, retail mature-store method, K 16, CH_K 20,
   DRIFT_ADJ 1.0, AMP floor 0.7, SEAS_ANCHOR 0.75, POW 0, HAL_SOFT.

Forecast totals (1262 items; Feb actual for these items 569.4k = 20.3k/day; 11 items at zero):

| Month | Units | per day |
|---|---|---|
| Mar-26 | 593.6k | 19.1k |
| Apr-26 | 461.9k | 15.4k |
| May-26 | 447.9k | 14.4k |
| Jun-26 | 393.8k | 13.1k |
| 4-month total | 1.897M | |

By group Mar / Apr / May / Jun: WINTER 185 / 113 / 95 / 77k (Feb 253k), SUMMER 262 / 234 / 241 / 219k (Feb 192k),
CORE 146 / 115 / 112 / 98k (Feb 124k). Items ≈ 98% of the planned site units in Mar, 89% Apr, 78% May, 74% Jun
(retail taken at 2.5k/day for this ratio).

## 3. Backtests (r4 / r8: full rules, origins Feb-25..Jan-26 with the scoreboard item lists, site plan = actuals;
WAPE / bias / ±20 / ±50 = mean over the 9 origins with a complete 4-month window (Feb–Oct-25); h1 = mean h=1 WAPE
over 12 origins)

| variant | WAPE | bias | ±20% | ±50% | h1 |
|---|---|---|---|---|---|
| v12 (Feb actuals in the dicts) | 0.3136 | −7.2% | 0.357 | 0.735 | 0.2398 |
| v12 SEAS_ANCHOR 0.5 / 0.25 / 1.0 | 0.3153 / 0.3189 / 0.3138 | −5.9 / −4.6 / −8.2% | 0.353 / 0.349 / 0.363 | 0.728 / 0.720 / 0.739 | 0.2385 / 0.2397 / 0.2433 |
| v12 AMP floor 0.8 | 0.3148 | −6.2% | 0.357 | 0.734 | 0.2406 |
| v12 DRIFT_ADJ 0.98 | 0.3144 | −11.0% | 0.364 | 0.747 | 0.2399 |
| + new tag / tier indices (with June) | 0.3124 | −7.2% | 0.357 | 0.736 | 0.2371 |
| + Sweatpants trend 1.025^h / 1.04^h | 0.3122 / 0.3123 | −6.9% | 0.356 / 0.357 | 0.735 | 0.2370 |
| v12 June values for 4th of July / Graduation | 0.3120 | −6.9% | 0.357 | 0.736 | 0.2372 |
| **+ W14 0.5 (v13 final)** | **0.3114** | −7.1% | 0.356 | 0.735 | **0.2359** |
| W14 0.6 | 0.3109 | −7.0% | 0.358 | 0.738 | 0.2352 |
| v13 with SEAS_ANCHOR 0.5 / 0.25 / 0 | 0.3132 / 0.3169 / 0.3228 | −5.8 / −4.5 / −3.0% | 0.351 / 0.351 / 0.347 | 0.728 / 0.721 / 0.708 | 0.2346 / 0.2359 / 0.2397 |

The tag measurements overlap the Feb–May-25 backtest origins (partly in-sample); June entries are barely tested
(only the Feb–May-25 windows reach June). W14 0.5 / 0.6 improve 8 of 9 four-month origins, the spring bases most
(Apr-25 h1 0.1955 → 0.1890 at 0.6), but are worse on the Oct-25–Jan-26 bases (holiday last-14-days) → 0.5.
Anchoring: 0.75 best on 4-month WAPE (6 of 8 origins with an effect), but the Jan-26 origin's h=1 prefers 0
(0.2226 → 0.2078), as does Mar-25's h=1 → kept, flagged. Sweatpants trend rejected (no gain even in-sample; code
removed). Type special cases (Skirt) rejected: the sign flips by season.

## 4. Known weaknesses

- **The site plan is still the forecast.** Web YoY 1.30–1.38 is a judgement between the Feb monthly 1.28 and the
  late-Feb aligned 1.41. TY web has been flat at ~14k/day for six weeks while LY fell into March; if TY keeps that
  run-rate, March is ~10% above plan. Discount fell to 0.57 on Feb 27–28 — check whether that is a regime change.
- **Other = TikTok momentum bet**: the plan keeps ~4.0k/day; LY's pattern (TikTok halved Feb→Apr) would mean ~3.0k,
  continued TikTok growth ~4.5–5k. Items with a high other share carry this risk (not examined per item / account).
- **WINTER / CORE anchoring** failed twice at h=1 (WINTER 1.15 / 1.12, CORE 0.97 / 0.86). The rules still assume
  WINTER falls to 0.30 of Feb by June; a cold / late spring repeats the Jan–Feb miss. If WINTER is again > 1.1 in
  March, lower SEAS_ANCHOR (0.5) or switch it off when the TY move matched LY's in the last month.
- Item logic under site actuals is +3–5% high in the last three origins (−7% over 2025): the panel may be losing
  share faster than LY drift says (106 launches in Jan). DRIFT_ADJ 0.98 trades WAPE for hit rates; re-test later.
- Tag / tier indices rest on one LY season (n = 6–58); Evening is a compromise between conflicting reads; June
  4th of July 1.10 rests on 13 items. Retail factor rests on 3 LY mature stores; new stores (Nov-25) have no LY
  spring. Retail-heavy items (> 50% stores) 0.86 at h=1 in Feb.
- Future planners: update `SITE_WEB_PD` / `OTHER_PD` with Mar actuals and add Jul (back-to-school event: Jul-25 web
  13.9k vs Jun 8.8k — LY's July jump was once missed by 35%); add July values to tags / tiers from a Mar-25 base
  (4th of July Jul 1.2 is old); watch WINTER at h=1 and TikTok.

## Changelog

- **v13 (2026-02-28)**: site plan Mar–Jun-26 = 12.8 / 10.9 / 12.0 / 11.5k web/day (YoY 1.30–1.38; Feb actual 13.83k
  vs 14.5k planned) and other 4.2 / 3.95 / 3.95 / 3.8k (Feb actual 4.18k vs 3.3k: second +27% miss, TikTok rising
  against LY's spring decline); Feb actuals added. Occasion / tier indices Mar–Jun re-measured from a Feb-25 base and
  **June added for every tag and tier** (v12 read June as a reversal to 1.0; 4th of July Jun 1.3 → 1.10,
  Graduation Jun 1.0 → 0.75, Halloween / Valentine's / NYE / Party / Loungewear / Basics / Resort-Festival June
  values set; Evening Mar 1.187 → 1.14). W14 0.4 → 0.5 (oracle-site backtest 0.3124 → 0.3114, 8 of 9 origins).
  Tested and rejected: SEAS_ANCHOR 0 / 0.25 / 0.5 / 1.0, AMP floor 0.8, DRIFT_ADJ 0.98, Sweatpants type trend,
  W14 0.6, Skirt special case.
- v12 (2026-01-31): site plan Feb–May from January (web YoY 1.34–1.42, other 3.3 → 3.15k); AMP_CLIP floor 0.45 → 0.7,
  SEAS_ANCHOR 0.5 → 0.75 (backtest 0.3218 → 0.3148); occasion / tier Feb–May from a Jan-25 base with May added;
  Basics spring index re-added. Rejected: dropship / store-rollout uplifts, DRIFT_ADJ 1.02 / 1.04, amplitude 1.0,
  W14 0.3 / 0.5, WLONG 0.35, CH_K, SHRINK_K, AGE_FADE, RAMP0_NEW 2.5.
- v11 (2025-12-31): site plan Jan–Apr-26 from late-Dec aligned YoY 1.33; retail level from mature-store LFL ×
  retail panel drift (flagship over-stated Nov→Dec); AGE_FADE 0.93^(h−1) for 1–4-month-old items; RAMP0_NEW 2.0;
  occasion / tier indices from a Dec-24 base with April; Basics dropped, Loungewear added; dead code removed.
- v10 (2025-11-30): site plan re-based on Nov actual (aligned YoY 1.53), SPIKE_CAP 1.0, RAMP_NOV, occasion indices
  from a Nov-24 base (Valentine's, St. Patrick added), tier indices Evening / Basics / Resort-Festival.
- v9 (2025-10-31): site plan from Oct (Oct-14 flash day), DRIFT_ADJ 1.01 → 1.0, flash-day cap, Halloween post-peak,
  HAL_SOFT, Party-tier step.
- v8 (2025-09-30): adaptive seasonal amplitude (`ADAPT_AMP`), RET_POW 0.6 → 0.8, occasion indices from a Sep base.
- v7 (2025-08-31): SEAS_POW 0.8, ANCHOR_LONG, DRIFT 1.01, December occasion entries.
- v6 (2025-07-31): web / other = LY panel drift × explicit site plan (`SITE_WEB_PD`, `OTHER_PD`); NYE tag.
- v5 (2025-06-30): fully deflated web, pop-up / closed store rule, K 16, anchoring 0.5.
- v4 (2025-05-31): site web path by month, RET_POW 0.6, occasion tags as calendar index.
- v3 (2025-04-30): deflated web level, asymmetric pre-anchor, 4th of July multiplier.
- v2 (2025-03-31): July-24 discount-regime fix, retail two-anchor, new-store run-rate, graded ramp.
- v1 (2025-02-28): base blend, channel-weighted level, type × sleeve seasonal profile shrunk to group, young-item
  ramp, zero for items with no sales in 28 days.

## History (condensed from v1–v12 notes)

- **Site level drives most misses**: Apr-25 +16%, May −7% (Memorial Day), Jul −35% (back-to-school event), Aug −12%,
  Sep +25% (post-Labor-Day lull), Oct −5% (Oct-14 flash day), Nov −26% (BF; YoY held ~1.5), Dec +8% (retail
  flagship factor too high), Jan-26 −8.5% (web YoY 1.465 vs 1.32 planned, TikTok surge), Feb-26 +2.8% (web −4.6%,
  other +27%). Past months hold actuals in the plan dicts so oracle backtests isolate the item logic.
- **2024 regime**: late July 2024 edikted.com moved to permanent deep discounts (0.45 → 0.57; LFL web ×2.3); the
  deflated-web design (LY panel drift × site plan) removes it.
- **Seasonal amplitude**: spring 2025 needed full amplitude (Jan-25 test: 1.0 best), summer 2025 ran at ~0.5 of LY's
  log-moves (but a 0.7 floor backtests better), Sep→Jan 0.8–1.0 (measured 0.51 Sep, 0.67 Oct, 0.89 Nov, 0.98 Dec,
  0.78 Jan origins; Feb origin 0.7 at the floor).
- **Stores**: SoHo flagship (2024-01), LAGROVE (May-24), MNMOA (Jun-24), CAAMR / CAIRV (Nov-24), FSM (Mar-25),
  NYR (pop-up Jun-25, reopened Jul), BOCA (Aug-25), FLAVENTURA / PAKOP / VATYSONS (Nov-25). LY mature stores
  Dec→Jan 0.59, Feb 0.66, Mar 1.04, Apr 1.09.
- **Occasion tags**: Valentine's never shows a February lift (Nov-24, Dec-24, Jan-25 bases); NYE items held in Jan
  both years; Halloween slides post-peak through spring; Graduation +20–30% vs segment Feb–May-25 from Jan, falls
  in June; Homecoming fades after a January pop. Jan-25-base reads (v12, raw Feb / Mar / Apr / May): Graduation 1.30
  1.34 1.30 1.20, Homecoming 0.73 0.68 0.59 0.56, Halloween 0.99 0.89 0.87 0.86, NYE 0.99 0.89 0.86 0.82, Valentine's
  0.96 0.87 0.88 0.84, St. Patrick 1.30 1.67 1.19 0.94; tiers Party 0.99 0.98 0.91 0.83, Evening 1.07 1.12 0.97 0.89,
  Resort-Festival 1.14 1.04 0.82 0.83, Loungewear 0.88 → 0.79, Basics 1.04 1.03 1.04 1.30. v11 Dec-24-base (Jan /
  Feb / Mar / Apr): Halloween 0.98 / 0.98 / 0.95 / 0.93, NYE 1.0 / 0.98 / 0.95 / 0.92, Valentine's 1.0 / 0.97 /
  0.92 / 0.92, Homecoming 1.12 / 1.0 / 0.95 / 0.88.
- **Segment growth**: items with no store sales, high dropship share and small items grow faster than the panel on
  average (scoreboard), but every uplift rule tested (POW, DROP_GROW, RET_ROLL) hurt WAPE / hit shares.
- **Rejected repeatedly**: POW (uniform and by horizon; costs ±20/±50 hits), channel-share uplifts (v12), W14 0.3,
  ramp 1.8 / 1.5 from a Nov base, momentum anchoring, ANCHOR_WIN 1/3/6, SEAS_ANCHOR ≤ 0.25 (again in v13),
  DRIFT_ADJ > 1.0 (lower bias, worse WAPE), DRIFT_ADJ 0.98 (v13), product-type trend / special cases (v13).
