# Forecast notes: Jan-26 – Apr-26 per style_color (snapshot 2025-12-31), rules v11

`python rules.py` reads `data/` and writes `predictions.csv` (1128 items × 4 months, a few seconds, 396 lines).
Research scripts in `research/`: `r1_site.py` (site / channel / store per month, Dec daily), `r1b_yoy.py`
(day-of-week-aligned YoY inside December, LY Jan–Apr weekly), `r2_review.py` (scoreboard by h / origin, Dec-25 h=1
by group / age / channel / size / type / tier / tag), `r3_drift.py` (panel-vs-site drift by channel, TY vs LY),
`r4_bt.py` (Dec-24 base → Jan–Apr-25 allocation test, group oracle; parameter variants), `r5_resid.py`
(segment-oracle tag / tier residuals from Dec-24; scoreboard by size quintile and age × h), `r6_bt.py` / `r6b_bt.py`
(full-rules backtest on the 2025 origins, variants), `r7_fc.py` (factor dump, forecast summary), `r8_decomp.py`
(ramp / tag contributions; share of next-month units from base-month sellers), `patch_refactor.py` (dead-code removal,
output checked identical).

Convention below: **act/pred** = actual ÷ prediction relative to the month's overall ratio (>1 = under-forecast).

## 0. Review of v10 (origin 2025-11) and older origins

- **Dec-25 at h=1: v10 1,151.9k vs actual 1,065.5k (1070 items), bias +8.1%, WAPE 0.218** — the best h=1 month
  so far. Older origins for Dec: v9 (h=2) −10.7%, v8 (h=3) −17%, v7 (h=4) +15%.
- **Where the +8% came from**: web site level on plan (24.2k/day vs 24.5k, −1%), web panel drift 0.962 vs LY
  0.976 (−1.5%); other 7.37k vs 8.0k (−8%, ~17% of units); **retail: v10 factor ~1.32 (flagship SoHo LY Nov→Dec
  1.30 × new-store run-rate 1.084) vs the panel's actual retail 0.91** (site retail +3%, retail panel drift 0.88) →
  ~+3 points of the bias. The flagship over-states the season for the mall stores: LY mature stores Nov→Dec 1.12,
  TY mature stores 1.10 (SoHo 1.27, others 0.9–1.2).
- Segments (act/pred): WINTER 1.02 / SUMMER 0.97 / CORE 0.99 — fine. **Basics 1.36** (v10 Dec ×0.88 Basics index
  was wrong; Nov was 1.13 too), Pants 1.29, Strapless 1.29, Sweatpants 1.14; Dress 0.86, Social Daywear 0.88,
  Knitted Top 0.85, Shirt 0.87. Tags: **NYE 1.25** (Dec peak bigger than ×1.05), Graduation 0.79, St. Patrick 1.18,
  Halloween 0.94, Valentine's 0.98. Items launched in Oct (1–2 months old) 1.29; web-only items 1.23; size flat.
- **Scoreboard, all origins**: h=1 WAPE 0.289 / bias −6.3%, h=2 0.398 / −10.8%, h=3 0.438 / −9.1%, h=4 0.499 /
  −6.1%. 4-month WAPE by origin 0.30–0.40. Two systematic patterns across 2025 origins (r5):
  - **Age**: items 1–4 months old at the base month: act/pred 0.99 / 0.96 / 0.87 / 0.77 at h = 1..4 (n ≈ 1300 per
    h) — they fade faster than the panel at long horizons. Items launched inside the base month 1.44 / 1.72 / 1.70 /
    1.65 (n ≈ 25).
  - **Size**: smallest four quintiles act/pred 1.0–1.06 at h=1 but 1.16–1.31 at h=4; top quintile 0.91 at h=4.

## 1. Key findings (new month)

**Site Dec** (per day): units 35.9k, web 24.2k (disc 0.556 vs LY 0.569), retail 4.28k, other 7.37k. Raw web YoY
1.37, calendar-aligned 1.47 (TY Nov held the post-BF Sunday). **Day-of-week-aligned YoY inside Dec: Dec 4–10 1.57,
11–17 1.67 (TY event Dec 13–16), 18–24 1.34, 25–31 1.33.** TY after-Christmas promo Dec 25–30 (Christmas Day 30.8k);
Dec 31 9.6k vs LY Dec 31 10.4k. When TY cut the discount to 0.42–0.47 (Dec 4–6) web fell to 11–14k/day. LY Jan–Feb-25
were deep-promo months (disc 0.60); LY Apr-25 weak (7.9k, Easter Apr 20; 2026 Easter Apr 5).
**Other** by account, Dec YoY: Macy's 1.87, TikTok 1.45, Bloomingdale's 1.39, Pacsun 0.92, Nordstrom 0.95, Tillys
0.87, Urban new (8.6k/month). Other ex-Flip YoY Nov 1.60 → Dec 1.24.
**Retail**: 11 stores, no openings in Dec. LY mature stores (SoHo, LAGROVE, MNMOA) Dec→Jan 0.59, Feb 0.66, Mar 1.04,
Apr 1.09; flagship alone 0.58 / 0.58 / 0.96 / 1.14.
**Panel**: LY share of next-month units coming from base-month sellers: Dec-24→Jan 98.5%, →Feb 96.1%, →Apr 80.9%.
LY drift (web) from Dec-24: 0.941 / 0.907 / 0.853 / 0.779. Amplitude now reads 0.98 (0.89 last month); anchoring
WINTER 1.05 / SUMMER 0.96 / CORE 0.98. LY group rel. Jan–Apr: WINTER 0.86 / 0.74 / 0.55 / 0.39, SUMMER 1.25 / 1.44 /
1.84 / 2.21, CORE 1.02–1.08.
**Dec-24 base allocation test (r4, group × month oracle, item logic only; Dec-23 does not exist so the full rules
cannot run from Dec-24)**: v10 logic 0.2902; seg oracle 0.2751; SPIKE_CAP 0.8 / 1.5 / off ±0.001 (kept 1.0); ramp
2.0 / 1.6 / 1.25 0.2864 (launched-in-Dec items 1.51 with 1.6); W14 0.2 / 0.6 worse; **WLONG 0.35 0.2841** (not
adopted: one season, 2025 backtests chose 0.2); POW 0.05 0.2679 (but rejected again in the full backtest, below).
**Tag / tier residuals from a Dec-24 base, segment × month oracle (Jan / Feb / Mar / Apr)**: Halloween (55) 1.00
1.03 0.94 0.91; NYE (22) 1.02 1.00 0.92 0.88 (no post-NYE drop); Valentine's (50) 1.00 0.95 0.85 0.85 (no Feb lift,
again); Homecoming (15) 1.31 0.95 0.90 0.77; Graduation (13) 0.94 1.22 1.29 1.24; 4th of July ~1.0; St. Patrick (6)
1.06 1.35 1.80 1.31. Untagged tiers: Party (67) 0.98 0.99 0.94 0.86; Evening (32) 0.98 1.06 1.12 0.97; Basics (28)
0.97–1.0; Resort-Festival (11) 1.23 1.50 1.47 1.06; Social Daywear (35) 1.11–1.19; Loungewear (9) 0.95 → 0.76.

## 2. Rules (rules.py, v11)

`pred(i,m) = base_rate_i × days(m) × level_i(m) × season_seg(m) × ramp_i(h) × fade_i(h) × occasion_i(m)`.

1. **Site plan** (`SITE_WEB_PD`, web/day): Dec actual 24,203; **Jan-26 12.65k** (YoY 1.32), **Feb 14.1k** (1.30),
   **Mar 12.75k** (1.32), **Apr 10.7k** (1.35). Anchored on the late-Dec aligned YoY 1.33 rather than the
   whole-December 1.47 (mid-Dec event) and on TY's lower discount vs LY's deep-promo Jan–Feb. v10 had 13.8 / 15.0 /
   13.4k. Web factor incl. drift 0.492 / 0.528 / 0.450 / 0.344.
2. **Other plan** (`OTHER_PD`): Dec actual 7,367; Jan 3.0k, Feb 2.7k, Mar 2.35k, Apr 2.15k (per-account LY × recent
   YoY, damped; ≈1.3 on LY ex-Flip) → factor 0.383 / 0.332 / 0.272 / 0.227.
3. **Retail level from mature stores** (open > 60 days before LY's base month, LFL per day) × LY retail panel drift
   (panel retail LFL ÷ all-store retail), instead of the flagship deflated by web growth. Applied to Nov→Dec this would
   have given ~1.03 (× new-store adj.) instead of 1.32 (actual 0.91). Factor now 0.64 / 0.67 / 0.88 / 0.85 (flagship
   method: 0.61 / 0.60 / 0.86 / 0.91). RET_POW 0.8, pre-anchor, new / closed store rules unchanged.
4. **`AGE_FADE` 0.93**: items 1–4 months old at the base month × 0.93^(h−1) (from the scoreboard age pattern).
5. **`RAMP0_NEW` 2.0** (was 1.6) for items launched inside a non-November base month (scoreboard 1.44–1.72; Dec-24
   test 1.51). Affects 1 item this month. RAMP_NOV unchanged.
6. **Occasion / tier indices for Jan–Apr re-measured from a Dec-24 base** (segment-controlled, shrunk ~halfway):
   multipliers from Dec — Halloween ×0.98 / 0.98 / 0.95 / 0.93; NYE 1.0 / 0.98 / 0.95 / 0.92; Valentine's 1.0 / 0.97 /
   0.92 / 0.92; Homecoming 1.12 / 1.0 / 0.95 / 0.88; Graduation 1.0 / 1.08 / 1.12 / 1.15; 4th of July flat;
   St. Patrick 1.0 / 1.1 / 1.25 / 1.1. Tiers (untagged): Party 1.0 / 1.0 / 0.97 / 0.93; Evening 1.0 / 1.03 / 1.06 /
   1.0; Resort-Festival 1.1 / 1.22 / 1.2 / 1.03; **new Loungewear** 0.98 / 0.94 / 0.91 / 0.9; **Basics index dropped**
   (TY act/pred 1.13 Nov, 1.36 Dec; LY flat from Dec). Social Daywear not used (LY up, TY over-forecast twice).
   Tier indices now skip items with a Halloween / NYE / Valentine's tag (they keep the tag index). April values were
   added for every tag (v10 had none → Apr read as a reversal, e.g. Graduation ×1.47).
7. **Code clean-up** (output identical): removed the retired web LFL anchors (W_DEFL had been 1.0), the July-24
   regime fix (only fed those anchors), the OTHER_PATH switch and the flagship retail path.
8. Unchanged: base blend (W14 0.4, WLONG 0.2), flash-day cap 1.0, K 16, channel shrink, adaptive amplitude +
   anchoring, DRIFT_ADJ 1.0, POW 0, HAL_SOFT, RAMP1/2.

Forecast totals (1128 items; Dec actual for these items 1,106.9k = 35.7k/day; 11 items at zero):

| Month | Units | per day |
|---|---|---|
| Jan-26 | 566.3k | 18.3k |
| Feb-26 | 521.6k | 18.6k |
| Mar-26 | 526.8k | 17.0k |
| Apr-26 | 413.3k | 13.8k |
| 4-month total | 2.028M | |

By group Jan / Feb / Mar / Apr: WINTER 253 / 201 / 148 / 81k, SUMMER 188 / 198 / 257 / 239k, CORE 126 / 123 / 122 /
93k. Items ≈ 98.6% of the planned site units in Jan (LY 98.5%), 93% in Feb (LY 96%), 84% in Apr (LY 81%).

## 3. Backtests

- Full rules, 2025 origins with the scoreboard item lists, site plan dicts = actuals for past months (r6 / r6b),
  mean over the Feb–Aug-25 origins (complete 4-month windows):

  | variant | WAPE | bias | ±20% | ±50% |
  |---|---|---|---|---|
  | v10-like (flagship retail, RAMP0 1.6, no fade) | 0.3210 | −3.6% | 0.347 | 0.721 |
  | **v11 (final)** | **0.3151** | −6.7% | **0.354** | **0.736** |
  | v11 + fade neutral (rescaled) | 0.3162 | −5.0% | 0.352 | 0.729 |
  | v11 + fade range 1–6 months | 0.3154 | −7.5% | 0.354 | 0.740 |
  | *with POW_H 0 / .03 / .06 / .08 on:* v11 | 0.3152 | −6.4% | 0.336 | 0.695 |
  | … POW_H × 1.5 | 0.3174 | −6.3% | 0.323 | 0.674 |
  | … without AGE_FADE | 0.3186 | −4.8% | 0.330 | 0.687 |
  | … AGE_FADE 0.90 | 0.3147 | −7.0% | 0.337 | 0.698 |
  | … flagship retail (v10 method) | 0.3167 | −5.0% | 0.336 | 0.692 |

  The POW_H runs were the first round; component effects (fade 0.3186 → 0.3152, retail 0.3167 → 0.3152) are
  measured there.  Horizon-dependent POW:  WAPE flat and ±20 / ±50 clearly worse → rejected, like POW before. Nov-25 origin (h=1 only): 0.2031 → 0.1905
  (retail fix). With site actuals the item logic under-forecasts by 4–7% — the panel loses more to new items than LY
  drift says, not corrected (DRIFT_ADJ > 1 was rejected last month on WAPE).

## 4. Known weaknesses

- **The site plan is still the forecast.** Jan–Apr YoY 1.30–1.35 is a judgement between the late-Dec aligned 1.33
  and the Sep–Dec 1.42–1.53. If YoY returns to ~1.45 the forecast is ~10% low; 2025's bias was negative at every
  horizon in the scoreboard (but +8% at Dec h=1). Discount policy (0.556 TY Dec vs LY Jan–Feb 0.60) drives this.
- April: LY Apr-25 was weak (Easter Apr 20); TY Easter Apr 5 might pull spring demand into late March / early April.
  WINTER items fall to 0.13 of Dec by April (LY profile at amplitude 0.98) — a warm/cold spring shifts this.
- Retail: mature-store set from Dec-24 is only SoHo / LAGROVE / MNMOA (3 of 11 stores today); malls vs tourist
  stores may differ in spring. RET_POW 0.8 damping kept without fresh evidence.
- Tag / tier indices rest on one LY season, n = 6–67, segment-oracle residuals; Graduation and St. Patrick are small.
  The Graduation April value (0.8625) breaks the old Apr→May shape (1.2/0.8625) — re-measure before the Mar/Apr
  origins. Tier list strings are free text (e.g. "Party | Evening" gets both).
- AGE_FADE is level-reducing (≈ −1.5% at h=4 in total); the backtest bias with site actuals is −6.7%.
- Size skew at long horizons (small items under) is real in the scoreboard but every POW variant hurts the ±20/±50
  shares; an alternative (e.g. a floor for small sellers at h≥3) is untested.
- Future planners: check Jan web/day vs 12.65k and the January discount; Loungewear / Resort-Festival indices; NYE
  items in Jan (LY says no drop); retail Jan vs 0.64 factor (~2.7k/day site); update `SITE_WEB_PD` / `OTHER_PD`
  with Jan actuals and add May.

## Changelog

- **v11 (2025-12-31)**: site plan Jan–Apr-26 = 12.65 / 14.1 / 12.75 / 10.7k web/day (YoY 1.30–1.35 from the late-Dec
  aligned 1.33; v10 13.8 / 15.0 / 13.4k) and other 3.0 / 2.7 / 2.35 / 2.15k; Dec actuals added. Retail level from
  mature-store LFL × retail panel drift (flagship over-stated Nov→Dec: v10 1.32 vs actual 0.91; backtest 0.3167 →
  0.3152). AGE_FADE 0.93^(h−1) for 1–4-month-old items (scoreboard 0.99 → 0.77 over h;
  backtest 0.3186 → 0.3152, ±50% 0.687 → 0.695); v11 overall vs v10-like 0.3210 → 0.3151, ±50% 0.721 → 0.736. RAMP0_NEW 2.0 for non-November bases. Occasion / tier indices
  re-measured from a Dec-24 base with April added; Basics index dropped, Loungewear added, tier indices skip tagged
  items. POW by horizon tested and rejected. Dead code removed (output identical), 459 → 396 lines.
- v10 (2025-11-30): site plan re-based on Nov actual (aligned YoY 1.53), SPIKE_CAP 1.0, RAMP_NOV, occasion indices
  from a Nov-24 base (Valentine's, St. Patrick added), tier indices Evening / Basics / Resort-Festival.
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

## History (condensed from v1–v10 notes)

- **Site level drives most misses**: Apr-25 +16%, May −7% (Memorial Day), Jul −35% (back-to-school event), Aug −12%,
  Sep +25% (post-Labor-Day lull), Oct −5% (Oct-14 flash day), Nov −26% (BF; growth decay assumed but YoY held ~1.5),
  Dec +8% (site on plan; retail flagship factor too high). Planned site levels are explicit per month; past months
  hold actuals so oracle backtests isolate the item logic.
- **2024 regime**: late July 2024 edikted.com moved to permanent deep discounts (0.45 → 0.57; LFL web ×2.3). The
  deflated-web design (LY panel drift × site plan) removes it; the old Jun→Jul-24 regime fix was retired in v11.
- **Seasonal amplitude**: spring 2025 needed full amplitude, summer 2025 ran at ~0.5 of LY's log-moves, Sep→Dec back
  to ~0.8–1.0 → measured each month (0.51 Sep, 0.67 Oct, 0.89 Nov, 0.98 Dec origins).
- **Stores**: SoHo flagship (2024-01), LAGROVE (May-24), MNMOA (Jun-24), CAAMR / CAIRV (Nov-24), FSM (Mar-25),
  NYR (pop-up Jun-25, reopened Jul), BOCA (Aug-25), FLAVENTURA / PAKOP / VATYSONS (Nov-25).
- **Occasion tags**: Valentine's never shows a February lift (Nov-24 and Dec-24 bases); NYE items fade at h=1 in
  Aug–Nov origins but peaked in Dec-25 (act/pred 1.25); Halloween slides post-peak through spring.
- **Rejected repeatedly**: POW (uniform and by horizon; costs ±20/±50 hits), W14 0.3, ramp 1.8 / 1.5 from a Nov base,
  momentum anchoring, ANCHOR_WIN 1/3/6, SEAS_ANCHOR 0.25, DRIFT_ADJ > 1.0 (lower bias, worse WAPE).
