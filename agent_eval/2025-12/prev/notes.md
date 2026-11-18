# Forecast notes: Dec-25 – Mar-26 per style_color (snapshot 2025-11-30), rules v10

`python rules.py` reads `data/` and writes `predictions.csv` (1070 items × 4 months, ~2 s, 411 lines).
Research scripts in `research/`: `r1_site.py` (site / channel / store per month, daily Nov-25, weekly Nov-24–Mar-25),
`r2_review.py` (Nov-25 scoreboard by group / type / age / channel share / size / tag / tier; TY vs LY drift and
group moves; LY daily around BF / CM), `r3_bt.py` (same-season allocation backtest from the Nov-24 BF base month,
oracle group × month totals; age / size rel. across scoreboard origins), `r4_tags.py` (tag / tier profiles from a
Nov-24 base, same-segment controlled; ramp variants), `r5_fc.py` (factor dump, forecast summary, v9 vs v10
backtest Feb–Oct-25 origins with site actuals), `r6_drift.py` (DRIFT_ADJ variants), `rules_v9.py` (last month's rules).

## 0. Review of v9 (origin 2025-10 → Nov-25, h=1) and older origins

- **November was a big level miss: v9 850.7k vs actual 1,142.8k (1003 items), bias −25.6%, WAPE 0.364.** Older
  origins for Nov: v8 (h=2) −30%, v7 (h=3) −2%, v6 (h=4) −8%.
- **Almost all of it is site level.** Web 32.1k/day vs 26.5k planned (×1.21; YoY 1.64 vs 1.36 assumed); other
  6.15k vs 4.5k (×1.37; TikTok 69k vs LY 29k, Macy's 21k vs 8.7k, Urban new 8k); retail 4.15k/day vs ~2.8k
  (3 new stores FLAVENTURA / PAKOP / VATYSONS opened in Nov, plus BF). Panel drift Oct→Nov TY 0.862 vs LY 0.881
  (normal). v9's assumption that YoY keeps decaying (Jun 2.0 → Sep 1.42) was wrong: Oct 1.50 ex-spike, Nov 1.53
  calendar-aligned.
- **Segments were fine**: WINTER 0.99 / SUMMER 1.00 / CORE 1.02 rel. to total (amplitude 0.67 + anchoring worked;
  TY Oct→Nov WINTER 1.61 vs LY 1.78, SUMMER 0.62 vs 0.65). Types: Pants 1.58, Shorts 1.31 (small), Knitted Top 0.41
  (12 items), Shirt 0.53, T-Shirt 0.84, Skirt 0.83.
- Tags (rel.): Halloween 0.89 (second month in a row: the post-peak slide is steeper than ×0.375), **NYE 0.79
  (also 0.82 Jul, 0.83 Sep, 0.79 Oct origins — a persistent fade)**, Valentine's 1.06, 4th of July 1.13,
  Homecoming 1.36 (small), St. Patrick 0.61 (13 items, small). Tiers: Casual 1.05, Party 0.82, Evening 0.89, Social
  Daywear 0.79, Basics 0.885 — BF favoured casual items.
- Age 4–8 months 0.75 (Oct 0.89); not confirmed by the same-season Nov-24 test (1.03) → treated as noise.
  Size: smallest quintiles over-forecast this time (0.80–0.87) — the opposite of the usual skew. Web-only items 0.86.
- Scoreboard across all origins: h=1 WAPE 0.304 / bias −9.4%, h=2 0.414 / −10.8%, h=3 0.435 / −7.1%, h=4 0.490 /
  −11.5%. The rules under-forecast a growing business more often than not.

## 1. Key findings (new month)

**Site Nov** (per day): units 42.4k, web 32.1k (disc 0.584 vs LY 0.564), retail 4.15k, other 6.15k; cancelled
383/day. Daily: Nov 1–22 web 13.2k (LY Nov 1–23 9.0k, ×1.47); Sun Nov 23 – Sat Nov 29 86.2k/day (LY BF week
54.1k, ×1.59); BF Nov 28 271k web units; Sun Nov 30 69.6k (LY's equivalent Sunday was Dec 1: 49.8k).
**Calendar**: BF Nov 28 / CM Dec 1 (LY Nov 29 / Dec 2). TY November holds the post-BF Sunday that LY December
held → raw Nov YoY 1.64, aligned 1.53. LY aligned Dec (Dec-24 minus Dec 1 plus a Jan-1-type day) ≈ 16.5k/day.
LY weekly after CM: 16.1k, 13.0k, 12.8k (Christmas), Jan 12.2 → 8–9.6k, Feb ~11k, Mar 7–11.5k.
**Other**: LY ex-Flip Nov 3.85k → Dec 5.94k (Pacsun ×1.6, Macy's ×2.6, Nordstrom ×1.1, TikTok ×1.7), Jan 2.28k,
Feb 2.05k, Mar 1.82k.
**Retail**: 11 stores; SoHo 844/day (Oct 859 — the growth is new stores). LY SoHo Nov→Dec 1.26, Jan 0.73, Mar 1.21.
Rule factor (deflated, ^0.8) 1.22 / 0.75 / 0.73 / 1.04 × new-store run-rate 1.084.
**Drift** (LY, from Nov-24 cumulated): 0.976 / 0.919 / 0.885 / 0.833. **Amplitude** reads 0.89 now (0.67 last month).
**Season groups** LY rel. to panel from Nov-24 (Dec / Jan / Feb / Mar): WINTER 1.04 / 0.89 / 0.77 / 0.57, SUMMER 0.93 /
1.17 / 1.34 / 1.72, CORE 1.00 / 1.03 / 1.08 / 1.08. Anchoring WINTER 1.07 / SUMMER 0.98 / CORE 0.95.
**Same-season allocation test (Nov-24 base → Dec-24..Mar-25, oracle group totals; r3/r4)**: v9 allocation WAPE
0.2274; flash cap 1.0 0.2219 (none 0.2255); smaller ramp 1.3/1.2/1.1 0.2178 (young items 0.84 of oracle with the
standard 1.6/1.4 ramp: BF lifts new arrivals into the base); W14 0.2 / 0.6 ±0.001 (kept 0.4); WLONG 0 0.2429
(blend matters); POW 0.05 worse 0.2383 although the smallest quintiles came in 1.2–1.4× (rejected again).
**Tags / tiers from a Nov-24 base, same-segment controlled (Dec / Jan / Feb / Mar)**: Halloween (51) 0.96 1.02 1.08
0.88 (both tier halves alike); NYE (22) 1.18 1.23 1.25 1.08; Valentine's (47) 0.92 0.92 0.86 0.74 (no Feb lift);
Homecoming (15) 0.84 1.26 0.82 0.77; Graduation (13) 1.08 0.95 1.28 1.38; 4th of July (13) 1.14 1.13 1.18 1.16;
St. Patrick (4) 1.07 1.48 1.69 1.63. Untagged tiers: Party 1.0 0.95 0.98 0.96 (flat); **Evening (23) 1.29 1.19
0.99 0.81** (holiday parties); **Basics (23) 0.76 0.79 0.84 0.90**; **Resort-Festival (10) 1.21 1.94 2.49 1.40**;
Casual / Social Daywear ~1.0; Loungewear (8) 1.29 → 0.68 (not used, n=8).

## 2. Rules (rules.py, v10)

`pred(i,m) = base_rate_i × days(m) × level_i(m) × season_seg(m) × ramp_i(h) × occasion_i(m)` — structure as v9.

1. **Site plan** (`SITE_WEB_PD`, web/day): Nov actual 32,130; **Dec 24.5k** (aligned LY 16.5k × YoY 1.49),
   **Jan-26 13.8k** (1.44), **Feb 15.0k** (1.39), **Mar 13.4k** (1.39). Slow growth decay from Nov's aligned 1.53.
   Web factor (incl. drift) 0.744 / 0.395 / 0.413 / 0.347.
2. **Other plan** (`OTHER_PD`): Nov actual 6,146; Dec 8.0k, Jan 3.2k, Feb 2.8k, Mar 2.45k (YoY 1.35–1.40 on LY
   ex-Flip) → factor 1.27 / 0.48 / 0.40 / 0.33.
3. **Flash-day cap `SPIKE_CAP` 2.0 → 1.0** (fires on Nov 23 and Nov 26–30 this month).
4. **`RAMP_NOV` (1.35, 1.25, 1.1)** replaces RAMP0/1/2 when the base month is November (other months unchanged —
   the 2025 scoreboard says items launched inside the base month are usually UNDER-forecast at h=1).
5. **Occasion indices re-measured from a Nov base, March added** (ratios from Nov, Dec / Jan / Feb / Mar):
   Halloween 0.98 / 1.0 / 1.04 / 0.97 (v9 index gave 1.02 / 1.11 / 1.28 / 1.11); NYE 1.05 / 1.0 / 1.0 / 0.95 (LY
   ~1.2 vs TY persistent 0.8 fade); Homecoming 0.89 / 1.14 / 1.0 / 0.86; Graduation 1.04 / 1.04 / 1.22 / 1.32;
   4th of July 1.07; **new: Valentine's 0.96 / 0.96 / 0.93 / 0.87; St. Patrick Feb 1.1 / Mar 1.2.**
6. **Tier indices (`TIER_IDX`)**: Party Sep/Oct step kept (flat from a Nov base); **new** Evening (not Halloween /
   NYE tagged) Dec ×1.12, Jan ×1.06; Basics 0.88 / 0.88 / 0.9 / 0.93; Resort-Festival 1.05 / 1.25 / 1.35 / 1.15.
   All roughly half the LY log-effect (n = 10–23, one season).
7. Unchanged: base blend (W14 0.4, WLONG 0.2), K, channel shrink, adaptive amplitude + anchoring, RET_POW 0.8,
   new / closed store rules, DRIFT_ADJ 1.0, POW 0, HAL_SOFT.

Forecast totals (1070 items; Nov actual for these items 1,237k = 41.2k/day):

| Month | Units | per day |
|---|---|---|
| Dec-25 | 1,151.9k | 37.2k |
| Jan-26 | 596.3k | 19.2k |
| Feb-26 | 538.7k | 19.2k |
| Mar-26 | 557.5k | 18.0k |
| 4-month total | 2.844M | |

By group Dec / Jan / Feb / Mar: WINTER 574 / 255 / 202 / 160k (Nov 582k), SUMMER 309 / 199 / 202 / 262k (Nov 340k),
CORE 269 / 142 / 135 / 136k (Nov 315k). Halloween (79 items, Nov 76k) 74 / 45 / 47 / 54k; NYE (54) 32 / 20 / 19 /
23k; Valentine's (93) 75 / 48 / 44 / 52k. v9 logic with the v10 site plan would be +2–4% per month.

## 3. Backtests

- Same-season allocation test (Nov-24 base, r3/r4): see §1; the v10 cap + Nov ramp choices come from it.
- Full rules, origins Feb–Oct-25 with the scoreboard item lists, complete months ≤ Nov-25, site at actuals (r5):
  v9 mean WAPE 0.3163 / bias −5.1% / ±20% 0.347 / ±50% 0.721; **v10 0.3167 / −4.6% / 0.348 / 0.717** (neutral —
  the v10 changes mostly act from a November base, which this window cannot test; Oct origin 0.313 → 0.315 from
  the tighter cap on the Oct-14 flash day).
- DRIFT_ADJ (r6): 1.0 0.3167 / −4.6%; 1.01 0.3183 / −2.9%; 1.02 0.3208 / −1.1% — lower bias but worse WAPE and
  ±20 / ±50 → kept 1.0.

## 4. Known weaknesses

- **The site plan is still the forecast.** Dec is 40% of the 4-month total. Dec 24.5k assumes Cyber Monday +
  December at aligned YoY 1.49; if YoY holds Nov's 1.53 Dec is ~+3%, if the BF-week YoY (1.59) repeats in CM week
  ~+6%. Jan–Mar YoY 1.39–1.44 is a judgement (LY Jan–Mar was itself a growth period in the same discount regime);
  the 2025 scoreboard bias is negative at every horizon (−7 to −12%), so the risk is skewed to under-forecasting.
- Other Dec 8.0k depends on wholesale holiday orders (Pacsun / Macy's / Nordstrom) and TikTok repeating LY's Dec/Nov.
- Retail: 3 stores opened in November; their run-rate is from the BF-heavy last 14 days (median day). The
  flagship-based seasonal factor (SoHo Nov→Dec 1.26 LY) may not fit the new mall stores.
- Tag / tier indices rest on one LY season and n = 4–51 items; NYE is a compromise between LY (+20%) and TY's
  persistent fade (−20%/month at h=1). Indices are only defined Dec–Mar for Valentine's / Basics / Resort:
  from a Mar base, April gets a reversal (e.g. Valentine's ×1.15) — revisit before the March origin.
- Size skew flips sign month to month (Nov-24 analogue: small items 1.2–1.4× under; Nov-25 h=1: 0.8–0.87 over);
  POW still rejected.
- Future planners: check Dec web/day vs 24.5k and the CM week; Halloween / NYE rel. in Dec; Evening-tier lift
  (assumed ×1.12); WINTER rel. (anchoring 1.07 expects WINTER catch-up); add Dec actuals to `SITE_WEB_PD` /
  `OTHER_PD`; extend plans and indices to April (Easter Apr 5 2026 vs Apr 20 2025; spring break / Festival).

## Changelog

- **v10 (2025-11-30)**: site plan re-based on Nov actual (32.1k web/day vs 26.5k planned; aligned YoY 1.53, decay
  stopped) and extended to Mar-26: Dec 24.5 / Jan 13.8 / Feb 15.0 / Mar 13.4k; other 8.0 / 3.2 / 2.8 / 2.45k.
  SPIKE_CAP 2.0 → 1.0 and RAMP_NOV (1.35 / 1.25 / 1.1) from the Nov-24 same-season allocation test (0.2274 →
  ~0.218). Occasion indices re-measured from a Nov-24 base with March added; Valentine's and St. Patrick tags added;
  TIER_IDX generalised (Party kept; Evening, Basics, Resort-Festival added). 2025-origin backtest neutral (0.3163 →
  0.3167). DRIFT_ADJ 1.01 / 1.02 tested and rejected.
- v9 (2025-10-31): site plan from Oct (Oct-14 flash day), DRIFT_ADJ 1.01 → 1.0, flash-day cap, occasion indices
  from an Oct base (Halloween post-peak, HAL_SOFT), Party-tier step.
- v8 (2025-09-30): adaptive seasonal amplitude (`ADAPT_AMP`), RET_POW 0.6 → 0.8, occasion indices from a Sep base.
- v7 (2025-08-31): SEAS_POW 0.8, ANCHOR_LONG, DRIFT 1.01, December occasion entries.
- v6 (2025-07-31): web / other = LY panel drift × explicit site plan (`SITE_WEB_PD`, `OTHER_PD`); NYE tag.
- v5 (2025-06-30): fully deflated web, pop-up / closed store rule, K 16, anchoring 0.5.
- v4 (2025-05-31): site web path by month, RET_POW 0.6, occasion tags as calendar index.
- v3 (2025-04-30): deflated web level, asymmetric pre-anchor, 4th of July multiplier.
- v2 (2025-03-31): July-24 discount-regime fix, retail two-anchor, new-store run-rate, graded ramp.
- v1 (2025-02-28): base blend, channel-weighted level, type × sleeve seasonal profile shrunk to group, young-item
  ramp, zero for items with no sales in 28 days.

## History (condensed from v1–v9 notes)

- **Site level drives most misses**: Apr-25 +16%, May −7% (Memorial Day), Jul −35% (back-to-school event), Aug −12%,
  Sep +25% (post-Labor-Day lull; retail LFL 0.45), Oct −5% (Oct-14 flash day: 45k web units, 145k cancelled),
  Nov −26% (BF; growth decay assumed but YoY held ~1.5). Planned site levels are explicit per month; past months
  hold actuals so oracle backtests isolate the item logic.
- **2024 regime**: late July 2024 edikted.com moved to permanent deep discounts (0.45 → 0.57; LFL web ×2.3); the
  Jun→Jul-24 link is replaced by the discount-matched Jul 8–21 lift and the web level is deflated by LY's site growth.
- **Seasonal amplitude**: spring 2025 needed full amplitude, summer 2025 ran at ~0.5 of LY's log-moves, Sep→Nov back
  to ~0.8–1.0 → measured each month (0.51 Sep, 0.67 Oct, 0.89 Nov origins).
- **Stores**: SoHo flagship (2024-01), LAGROVE (May-24), MNMOA (Jun-24), CAAMR / CAIRV (Nov-24), FSM (Mar-25),
  NYR (pop-up Jun-25, reopened Jul), BOCA (Aug-25), FLAVENTURA / PAKOP / VATYSONS (Nov-25).
- **Rejected repeatedly**: POW (costs ±50% hits; again worse on the Nov-24 test), W14 0.3, ramp 1.8 / 1.5,
  momentum anchoring, ANCHOR_WIN 1/3/6, SEAS_ANCHOR 0.25, DRIFT_ADJ > 1.0 (lower bias, worse WAPE).
