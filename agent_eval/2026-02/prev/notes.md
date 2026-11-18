# Forecast notes: Feb-26 – May-26 per style_color (snapshot 2026-01-31), rules v12

`python rules.py` reads `data/` and writes `predictions.csv` (1176 items × 4 months, a few seconds, 398 lines).
Research scripts in `research/`: `r1_site.py` (site / channel per month, Jan daily aligned YoY, LY Jan–Jun weekly,
channel accounts), `r2_review.py` (scoreboard by group / age / retail share / drop share / size / type / tag),
`r3_bt.py` (full-rules backtest, 11 origins Feb–Dec-25, site plan = actuals, variant dict), `r4_resid.py`
(residuals of that backtest by channel mix / size / age / group), `r5_season.py` (seasonal factors from a Jan base TY
and LY; Jan-25-origin backtest of the amplitude), `r6_perorigin.py` (per-origin WAPE of chosen variants),
`r7_tags.py` (tag / tier residuals vs segment oracle from a Jan-25 base), `r8_fc.py` (forecast summary).

Convention: **act/pred** = actual ÷ prediction, level-neutral (relative to the month's overall ratio) unless "raw".

## 0. Review of v11 (origin 2025-12) and older origins

- **Jan-26 at h=1: v11 566.3k vs actual 618.7k (1128 items), bias −8.5%, WAPE 0.259.** Older origins for Jan:
  v10 (h=2) +2.8%, v9 (h=3) −20%, v8 (h=4) −20%.
- **The miss is the site plan**: web 14.05k/day vs plan 12.65k (+11%, YoY 1.465 vs 1.32 assumed; TY ran a deeper
  discount, 0.611 vs LY 0.599, 0.62–0.63 on Jan 7–18); other 3.85k vs 3.0k (+28%: TikTok 2.55k/day, YoY 2.5, kept
  rising after Christmas while LY fell 0.63 Dec→Jan); retail 2.87k vs ~2.74k (+5%, mature-store method fine).
  With site actuals the item logic would have been ~2% HIGH (oracle backtest Dec origin raw act/pred 0.977; Nov
  origin 0.964) — the panel lost a bit more share to new items than LY drift says (opposite of early 2025).
- Segments Jan h=1: **WINTER 1.15 / SUMMER 0.82 / CORE 0.97** (Sweatpants 1.17, Hoodie 1.14, Sweater 1.12, Pants
  1.21; Corset 0.70, Skort 0.72, Short Jeans 0.81, Shorts 0.86, Tank Top 0.89). TY's Dec→Jan seasonal move was
  much milder than LY's (WINTER rel. to panel 0.977 vs 0.862, SUMMER 1.063 vs 1.254). Tags looked bad
  level-neutral (NYE 0.76, Halloween 0.85, 4th of July 0.83) but **vs a segment oracle all tags were 0.97–1.15** →
  the tag misses were the SUMMER/party segment, not the tag indices. Basics 1.18 (third month under: 1.13 Nov,
  1.36 Dec). Young items 0–2 months 1.07–1.08 at h=1; size: smaller quintiles 1.12–1.23 at h=1.
- **Scoreboard, all origins (systematic, n ≈ 1400–4600 per cell)**: items with no store sales in the base month
  act/pred 1.05 / 1.13 / 1.18 / 1.23 at h = 1..4, retail share > 25% 0.95 → 0.92; dropship share > 50% 1.05 /
  1.19 / 1.35 / 1.42; four smallest size quintiles 1.15–1.25 at h=4, top 0.92. These persist with site actuals
  (r4) and within size quintiles, but uplift rules for them fail on WAPE (§3) — the averages are driven by a few
  big growers, the median item does not grow.
- 4-month WAPE by origin (scoreboard) 0.30–0.40; oracle-site backtest of v11 0.3218 (mean of 8 full origins).

## 1. Key findings (new month)

**Site Jan** (per day): units 20.8k, web 14.05k (disc 0.611), retail 2.87k, other 3.85k. Web YoY by month Sep–Jan
1.42 / 1.60 / 1.65 / 1.37 / 1.47. Day-of-week-aligned YoY inside Jan: Jan 7–18 1.5–2.3 (TY 0.62–0.63 off vs LY
0.58–0.60); Jan 26–31 1.23 (TY 0.61 vs LY 0.63, LY's early-Feb promo). Late-Jan web run-rate 13.2k/day, other 3.37k.
LY Feb was LY's deepest promo month (0.611), LY Apr weak (7.9k; Easter Apr 20-25 vs Apr 5-26), LY May had a
Memorial-Day week at 10.8k.
**Other** by account Jan YoY: TikTok 2.48 (79k units, Sep→Jan per day 1.27 / 1.50 / 2.30 / 2.38 / 2.55k), wholesale
(Macy's, Nordstrom, Pacsun, Bloomingdale's, Tillys, Urban) 1.03 combined (Dec 1.16). LY ex-Flip per day vs Jan:
Feb 0.90, Mar 0.80, Apr 0.75, May 0.85 (LY TikTok halved Jan→Apr).
**Seasonal factors from the Jan-26 base**: measured amplitude 0.78; group anchoring (TY Nov→Jan vs LY) WINTER 0.89
/ SUMMER 1.12 / CORE 1.09 at SEAS_ANCHOR 0.75 — together ≈ LY's full Jan→May profile (WINTER 0.86 / 0.64 / 0.45 /
0.36, SUMMER 1.15 / 1.47 / 1.76 / 1.91 rel. to panel). Jan-25-origin backtest (r5): amplitude 1.0 0.300 vs 0.8
0.305 (WINTER/SUMMER act/pred 0.94 / 1.02 vs 0.85 / 1.10) — spring wants the full LY swing, as in 2025.
LY drift from Jan: 0.963 / 0.907 / 0.828 / 0.762; retail factor 1.02 / 1.35 / 1.30 / 1.05 (spring store season).
**Tag / tier residuals from a Jan-25 base, segment oracle (Feb / Mar / Apr / May-25)**: Graduation (13) 1.30 1.34
1.30 1.20; Homecoming (15) 0.73 0.68 0.59 0.56; Halloween (55) 0.99 0.89 0.87 0.86; NYE (22) 0.99 0.89 0.86 0.82;
Valentine's (50) 0.96 0.87 0.88 0.84; St. Patrick (6) 1.30 1.67 1.19 0.94; 4th of July ~1.0 (May 1.06). Tiers
(untagged): Party 0.99 0.98 0.91 0.83; Evening 1.07 1.12 0.97 0.89; Resort-Festival 1.14 1.04 0.82 0.83;
Loungewear 0.88 → 0.79; Basics 1.04 1.03 1.04 1.30; Social Daywear 0.91–0.97.

## 2. Rules (rules.py, v12)

`pred(i,m) = base_rate_i × days(m) × level_i(m) × season_seg(m) × ramp_i(h) × fade_i(h) × occasion_i(m)`
(unchanged structure; see the docstring).

1. **Site plan** (`SITE_WEB_PD`, web/day): Jan actual 14,050; **Feb 14.5k** (YoY 1.34), **Mar 13.15k** (1.36),
   **Apr 11.2k** (1.42, weak LY April / Easter shift), **May 12.4k** (1.36). Between the late-Jan aligned 1.23
   (discount-adjusted ~1.3) and the Sep–Jan monthly 1.37–1.65; v11's 1.32 was 10% low in Jan.
2. **Other plan** (`OTHER_PD`): Jan actual 3,849; Feb 3.3k, Mar 3.1k, Apr 2.95k, May 3.15k (LY ex-Flip seasonality
   from Jan, damped for TikTok momentum; late-Jan run-rate 3.37k).
3. **`AMP_CLIP` floor 0.45 → 0.7** and **`SEAS_ANCHOR` 0.5 → 0.75** (backtest §3). No effect of the floor this month
   (amp 0.78); anchoring lowers WINTER ~4% and raises SUMMER ~4% vs 0.5.
4. **Occasion / tier indices: Feb–May re-measured from a Jan-25 base**, shrunk halfway; **May values added for every
   tag and tier** (v11 had none → May read as a reversal: Halloween ×1.16, Valentine's ×1.04, Party ×1.07 from
   Jan; Graduation was ×1.6). Graduation now 1.15 / 1.17 / 1.15 / 1.10 from Jan; Homecoming 0.86 → 0.78;
   Halloween 1.0 → 0.92; NYE 1.0 → 0.91; Valentine's 0.98 → 0.92; 4th of July May 1.03; St. Patrick May 0.97.
   **Basics tier re-added for spring** (Feb–Apr ×1.03, May ×1.12; LY 1.03–1.30, TY under three months running).
5. Unchanged: base blend (W14 0.4, WLONG 0.2), flash-day cap, ramps, AGE_FADE 0.93, retail mature-store method,
   K 16, CH_K 20, DRIFT_ADJ 1.0, POW 0, HAL_SOFT.

Forecast totals (1176 items; Jan actual for these items 632.8k = 20.4k/day; 15 items at zero):

| Month | Units | per day |
|---|---|---|
| Feb-26 | 557.8k | 19.9k |
| Mar-26 | 556.8k | 18.0k |
| Apr-26 | 431.8k | 14.4k |
| May-26 | 415.9k | 13.4k |
| 4-month total | 1.962M | |

By group Feb / Mar / Apr / May: WINTER 226 / 180 / 107 / 88k (Jan 322k), SUMMER 192 / 237 / 216 / 222k (Jan 174k),
CORE 140 / 140 / 109 / 106k (Jan 137k). Items ≈ 96% of the planned site units in Feb, 89% Mar, 80% Apr, 72% May.

## 3. Backtests (r3 / r6: full rules, origins Feb–Dec-25 with the scoreboard item lists, site plan = actuals;
WAPE / bias / ±20 / ±50 = mean over the 8 origins with a complete 4-month window; h1 = mean h=1 WAPE, 11 origins)

| variant | WAPE | bias | ±20% | ±50% | h1 |
|---|---|---|---|---|---|
| v11 (base) | 0.3218 | −6.9% | 0.348 | 0.726 | 0.2394 |
| DROP_GROW 0.1 / 0.2 / 0.3 (× 1 + k·drop share·(h−1)) | 0.3225 / 0.3238 / 0.3256 | −5.7 / −4.4 / −3.2% | 0.347 / 0.343 / 0.340 | 0.720 / 0.710 / 0.700 | |
| RET_ROLL 0.03 / 0.06 / 0.1 (items under-represented in stores) | 0.3219 / 0.3234 / 0.3273 | −5.4 / −3.9 / −1.9% | 0.345 / 0.341 / 0.331 | 0.719 / 0.711 / 0.694 | |
| DRIFT_ADJ 1.02 / 1.04 | 0.3253 / 0.3330 | −2.9 / +1.3% | 0.342 / 0.329 | 0.710 / 0.687 | |
| amplitude fixed 1.0 | 0.3300 | −2.2% | 0.352 | 0.727 | 0.2437 |
| AMP_CLIP floor 0.6 / **0.7** / 0.8 / 0.9 | 0.3176 / **0.3161** / 0.3173 / 0.3214 | | 0.351 / 0.352 / 0.353 / 0.353 | 0.732 / 0.735 / 0.736 / 0.733 | |
| SEAS_ANCHOR 0 / 0.25 (floor 0.45) | 0.3339 / 0.3267 | | 0.333 / 0.341 | 0.700 / 0.714 | |
| floor 0.7 + SEAS_ANCHOR 0.75 | 0.3148 | −7.2% | 0.353 | 0.742 | 0.2402 |
| + W14 0.5 | 0.3137 | −7.1% | 0.354 | 0.742 | 0.2389 |
| W14 0.3 / 0.5 (floor 0.45) | 0.3235 / 0.3204 | | | | 0.2413 / 0.2380 |
| CH_K 10 / 40, SHRINK_K 8, AGE_FADE 0.90, fade range 1–6, RAMP0_NEW 2.5 | 0.3213–0.3218 | | | | |
| WLONG 0.35 | 0.3265 | | 0.344 | 0.714 | 0.2473 |
| **v12 final** (floor 0.7, anchor 0.75, new tag indices) | **0.3145** | −7.1% | **0.355** | **0.740** | 0.2415 |

Floor 0.7 gains come from the Jul–Sep-25 origins (amp read 0.45–0.6; 0.3202 → 0.3058, 0.3688 → 0.3462). Anchor
0.75 improves 7 of 11 origins and Dec→Jan-26 h=1 0.249 → 0.242 (WINTER act/pred 1.11 → 1.08). W14 0.5 not adopted
(+0.001 only; Dec-24 test last month had 0.6 worse). The new tag indices change the backtest by −0.0003.
Channel-growth uplifts: lower bias, worse WAPE and hit shares at every strength → rejected (code removed).
Under site actuals the item logic is −7% biased overall (raw act/pred by h 1.03 / 1.06 / 1.07 / 1.09) but ~+2–4%
in the Nov/Dec-25 origins; median-optimal WAPE prefers the negative aggregate bias (DRIFT_ADJ > 1 worse again).

## 4. Known weaknesses

- **The site plan is still the forecast.** Web YoY 1.34–1.42 is a judgement; TY's January was bought with a deeper
  discount (0.611). If TY returns to LY-like discounts the late-Jan 1.23 suggests ~8% lower; if TY keeps
  out-promoting LY (Nov, Jan) YoY ~1.5 means ~10% higher. Check Feb web/day vs 14.5k and the discount first.
- TikTok is now 2/3 of "other" and trending against LY's seasonality; the other plan (3.3 → 3.15k) assumes a
  partial LY-style spring decline. TikTok items may be concentrated — not examined per item.
- WINTER: Jan was much stronger than LY (cold January?). The rules assume LY's full spring transition (amp 0.78 ×
  anchoring) — WINTER falls to 0.27 of Jan by May. A late spring would make WINTER items under-forecast in
  Feb–Mar (as in Jan: 1.15); an early warm spring the reverse for SUMMER.
- Retail factor Mar/Apr 1.35 / 1.30 rests on the 3 LY mature stores (SoHo / LAGROVE / MNMOA); LY spring had
  FSM opening (Mar-25) excluded. New stores (Nov-25) have no LY spring.
- Items with no store sales and high-dropship items grow faster than the panel on average (scoreboard + oracle
  backtest), small items under-forecast at h≥3; every uplift rule tested (POW, DROP_GROW, RET_ROLL) hurts
  WAPE / hit shares. An item-level rule (e.g. recent trend of drop share) is untested.
- Tag / tier indices rest on one LY season (n = 6–68, segment-oracle residuals, shrunk halfway); Halloween / Party
  May values are new — Jun–Jul for Halloween still default 1.0 (re-measure before a May/Jun base).
- Future planners: update `SITE_WEB_PD` / `OTHER_PD` with Feb actuals and add Jun; check WINTER vs SUMMER in Feb at
  h=1 (anchoring 0.75 bets on the gap closing); Basics spring index; Graduation in Apr–May.

## Changelog

- **v12 (2026-01-31)**: site plan Feb–May-26 = 14.5 / 13.15 / 11.2 / 12.4k web/day (YoY 1.34–1.42; v11's Jan plan
  was 10% low, actual 14.05k) and other 3.3 / 3.1 / 2.95 / 3.15k (Jan actual 3.85k vs plan 3.0k; TikTok YoY 2.5);
  Jan actuals added. AMP_CLIP floor 0.45 → 0.7 and SEAS_ANCHOR 0.5 → 0.75 (oracle-site backtest, 11 origins:
  0.3218 → 0.3148, ±50% 0.726 → 0.742; Dec→Jan h=1 0.249 → 0.242). Occasion / tier indices for Feb–May re-measured
  from a Jan-25 base (segment controlled) with May added (v11 read May as a reversal; Graduation ×1.6 from Jan);
  Basics spring tier index re-added. Tested and rejected: dropship-share and store-rollout uplifts, DRIFT_ADJ
  1.02 / 1.04, fixed amplitude 1.0, W14 0.3 / 0.5, WLONG 0.35, CH_K, SHRINK_K, AGE_FADE variants, RAMP0_NEW 2.5.
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

## History (condensed from v1–v11 notes)

- **Site level drives most misses**: Apr-25 +16%, May −7% (Memorial Day), Jul −35% (back-to-school event), Aug −12%,
  Sep +25% (post-Labor-Day lull), Oct −5% (Oct-14 flash day), Nov −26% (BF; YoY held ~1.5), Dec +8% (retail
  flagship factor too high), Jan-26 −8.5% (web YoY 1.465 vs 1.32 planned, TikTok surge). Past months hold actuals
  in the plan dicts so oracle backtests isolate the item logic.
- **2024 regime**: late July 2024 edikted.com moved to permanent deep discounts (0.45 → 0.57; LFL web ×2.3); the
  deflated-web design (LY panel drift × site plan) removes it.
- **Seasonal amplitude**: spring 2025 needed full amplitude (Jan-25 test: 1.0 best), summer 2025 ran at ~0.5 of LY's
  log-moves (but a 0.7 floor backtests better), Sep→Jan 0.8–1.0 (measured 0.51 Sep, 0.67 Oct, 0.89 Nov, 0.98 Dec,
  0.78 Jan origins).
- **Stores**: SoHo flagship (2024-01), LAGROVE (May-24), MNMOA (Jun-24), CAAMR / CAIRV (Nov-24), FSM (Mar-25),
  NYR (pop-up Jun-25, reopened Jul), BOCA (Aug-25), FLAVENTURA / PAKOP / VATYSONS (Nov-25). LY mature stores
  Dec→Jan 0.59, Feb 0.66, Mar 1.04, Apr 1.09.
- **Occasion tags**: Valentine's never shows a February lift (Nov-24, Dec-24, Jan-25 bases); NYE items held in Jan
  both years; Halloween slides post-peak through spring; Graduation +20–30% vs segment Feb–May-25; Homecoming fades
  after a January pop. v11 Dec-24-base multipliers (Jan / Feb / Mar / Apr): Halloween 0.98 / 0.98 / 0.95 / 0.93,
  NYE 1.0 / 0.98 / 0.95 / 0.92, Valentine's 1.0 / 0.97 / 0.92 / 0.92, Homecoming 1.12 / 1.0 / 0.95 / 0.88.
- **Rejected repeatedly**: POW (uniform and by horizon; costs ±20/±50 hits), channel-share uplifts (v12), W14 0.3,
  ramp 1.8 / 1.5 from a Nov base, momentum anchoring, ANCHOR_WIN 1/3/6, SEAS_ANCHOR ≤ 0.25, DRIFT_ADJ > 1.0 (lower
  bias, worse WAPE).
