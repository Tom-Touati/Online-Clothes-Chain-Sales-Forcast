# Forecast notes: Jul-26 – Oct-26 per style_color (snapshot 2026-06-30), rules v14

`python rules.py` reads `data/` and writes `predictions.csv` (1424 items × 4 months, ~2 s, 430 lines).
Research scripts in `research/`: `r1_site.py` (site / channel / store per day by month), `r2_review.py` (scoreboard
review of the Feb-26 origin by age / tier / channel share / size / type / tag), `r3_daily.py` (LY May–Oct and TY
Mar–Jun weekly, LY Jul / TY Jun daily), `bt.py` (oracle-site backtest harness: 12 origins Mar-25..Feb-26 with the
scoreboard item lists, `run(name, overrides)`), `bt_v1.py` (parameter variants), `r4_summer.py` (residuals of the
May / Jun / Jul-25 origins by group / age / tier / size / type / tag), `r5_launchseason.py` (residuals by launch half
and tier by month), `bt_v2.py` / `bt_v3.py` (cohort rule and tag variants), `r6_fc.py` (forecast factors / totals).

Convention: **act/pred** = actual ÷ prediction, level-neutral (relative to the window's overall ratio) unless "raw".

## 0. Review of v13 (origin 2026-02, Mar–Jun-26) and older origins

- **v13 live: 4-month WAPE 0.335, bias −3.5% (pred 1.897M vs 1.965M), ±20% 0.36, ±50% 0.74** (1262 items).
  Monthly raw bias Mar −0.2%, Apr −3.3%, May −3.8%, Jun −7.8%. Recent origins' 4-month WAPE: Oct-25 0.322
  (−19%), Nov 0.274 (+8%), Dec 0.297 (−2%), Jan-26 0.316 (+2.5%), Feb-26 0.335 (−3.5%). Jun-25 origin (last
  summer, the analogue of this forecast) 0.401 / −32%: the plan missed LY's July back-to-school jump.
- **Site plan vs actual**: web 12.8 / 10.9 / 12.0 / 11.5k planned vs 13.1 / 11.06 / 10.68 / 10.61k actual
  (May / June +12% / +8% too high: YoY fell from 1.40 in April to 1.17 / 1.20). Other 4.2 / 3.95 / 3.95 / 3.8k vs
  3.89 / 3.32 / 3.61 / 3.44k (TikTok fell 3.05k → 1.76k Feb→May; the "TikTok keeps rising" bet failed). Retail
  2.5k → 3.1 / 3.5 / 4.1 / **6.1k/day** (summer + 6 new stores: CHOAKBROOK / HOUGALLERIA Apr, NYFIFTH late May
  ~750/day in June, FLLINCOLN / STONEBRIAR June, BARBIE pop-up May). The over-planned web and the un-planned store
  rollout roughly cancelled; under site actuals the item logic was **−7% (May −12%, Jun −14%)** for this origin.
- **Segments (systematic, Mar–Jun every month)**: WINTER over-forecast (Hoodie 0.75, Sweatpants 0.73, Sweater 0.79,
  Pants 0.58 — 4 of 4 months), SUMMER under (Short Jeans 1.51, Swimwear Bottom 1.47, Crop Top 1.38, Dress 1.31,
  Skirt 1.31 — 4 of 4 months). Spring 2026 swung harder than the 0.7–1.0 amplitude + WINTER anchoring assumed
  (the reverse of the Jan / Feb h=1 misses). Amplitude floor 0.85 / 1.0 and anchoring 0.5 re-tested: worse overall.
- Age: 1–2 months old at base (the Jan-26 launch wave) 1.39, 4–8 months 0.90, 8–14 months 1.15. Size: items
  < 300 units/month 1.30–1.39, 1k+ 0.89 (regression to the mean; POW re-tested, still worse on ±20/±50). Tags: 4th of
  July 1.19, Graduation 3.2 (21 items; TY graduation season far above LY), St. Patrick 1.54; Valentine's 0.94. Tiers:
  Resort-Festival 1.58, Party 1.16, Basics 0.90, Loungewear 0.73. Items without store sales 1.37 (small / young).
- **Launch-season effect (new, r5)**: items 3–15 months old launched Aug–Jan (FW) vs Feb–Jul (SS), level-neutral
  residual over all 12 oracle origins: FW 1.03 / 1.03 / 1.06 / **1.12 / 1.18 / 1.11 / 1.10** / 1.01 / 0.97 / 1.01 /
  **0.90 / 0.84** (Mar-25 .. Feb-26 origins), SS 0.99 / 0.98 / 0.97 / 1.00 / 0.96 / 0.90 / **0.79** / 0.86 / 1.02 /
  1.05 / **1.18 / 1.11**. Systematic: an item keeps the seasonality of the season it was launched for, beyond what
  its product type says. This explains the age-band residuals above (4–8 vs 8–14 months).

## 1. Key findings (new months)

**Site per day** (web / other / retail): Mar 13.10 / 3.89 / 3.12k, Apr 11.06 / 3.32 / 3.51k, May 10.68 / 3.61 /
4.06k, Jun 10.61 / 3.44 / 6.12k. Web discount 0.577 / 0.567 / 0.556 / 0.543 (LY 0.569 / 0.557 / 0.534 / 0.514: TY
~3 pts deeper). June weekly web YoY 1.18 / 1.11 / 1.23 / 1.17; June 14–16 sale 18.6–24.7k/day (0.63 disc), not a
flash day (< 2.5 × median). June 28–30 web 12–15k at 0.58–0.59 (a July 4th sale starting; LY's ran Jul 3–8).
**LY Jul–Oct shape**: web Jun 8.83k → Jul 13.9k (Jul 4–8 sale 13–26k/day; **back-to-school sale Jul 20 – Aug 16
at 0.57, 17–21k/day sustained**), Aug 15.6k, Sep 12.2k (Labor Day week 20k, then ~10k), Oct 12.75k (Oct 5–18
events 15–20k, end-Oct 6.5k). 2024 had the same "Jul–Oct ≥ 1.4 × June" shape (plus the July-24 discount regime).
**Other**: TikTok Feb→Jun 3.05 / 2.27 / 1.95 / 1.76 / 2.26k; drop/wholesale 1.13 / 1.62 / 1.38 / 1.85 / 1.18k.
LY Jun→Jul other ×1.71 (Nordstrom anniversary 245 → 798/day, Pacsun back-to-school, TikTok growth).
**Stores**: mature-store LY Jun → Jul / Aug / Sep / Oct ≈ 1.15 / 1.2 / 0.62 / 0.70 (school-out summer peak, then
September drop). TY May→Jun mature ×1.35 (LY same). New stores fade in their 2nd–3rd month relative to mature
(NYR 0.85, Nov-25 stores 0.85–1.0, CHOAKBROOK / HOUGALLERIA 0.64 / 0.50 in June) — not modelled.
**Rules' factors (June base)**: amplitude 1.0 (TY matched LY's spring moves), anchoring WINTER 1.08 / SUMMER 0.95 /
CORE 0.97; LY group rel. Jul–Oct WINTER 1.15 / 1.30 / 1.58 / 1.95, SUMMER 0.92 / 0.83 / 0.82 / 0.72; LY drift
0.95 / 0.90 / 0.81 / 0.76; retail factor 1.08 / 1.05 / 0.55 / 0.57 (×1.055 for the two June-opened stores).
Cohort factors (LY, before ^0.5): FW 0.98 / 1.02 / 1.08 / 1.13, SS 1.01 / 1.01 / 0.95 / 0.94.

## 2. Rules (rules.py, v14)

`pred = base_rate × days × level × season_seg × ramp × fade × occasion × cohort` (docstring).

1. **Site plan**: Mar–Jun actuals in the dicts. Web Jul **16.0k**, Aug **17.8k**, Sep **13.8k**, Oct **14.3k**/day
   (LY × YoY 1.15 / 1.14 / 1.13 / 1.12: May–Jun YoY 1.17–1.20 and decelerating, ~3 pts of it bought with discount).
   Other Jul **4.4k**, Aug **4.35k**, Sep **4.05k**, Oct **4.45k** (TikTok ~2.4k flat, drop ≈ LY × 1.1).
2. **Occasion / tier Jul–Oct entries from the June-25 origin** (full-rules residuals under site actuals, applied
   about halfway): 4th of July Jun 1.10 → 1.17; Graduation Jul–Oct 0.73 / 0.78 / 0.80 / 0.66; NYE Jul / Aug 0.97 /
   0.92 added, Sep 1.04, Oct 1.25 → 1.05; Valentine's Jul–Oct 0.89 / 0.89 / 0.92 / 0.86 and St. Patrick 0.94 / 0.86 /
   0.89 / 0.84 added (v13 had no entries → silent jump back to 1.0); Basics Jul–Oct 1.18 / 1.22 / 1.2 / 1.22;
   Resort-Festival 0.95 / 0.73 / 0.73 / 0.67. Halloween, Homecoming, Party, Evening unchanged.
3. **Launch-season cohort** (`cohort_factors`, COH_POW 0.5, COH_AGE 3–15 months, clip 0.5–2): items 3–15 months
   old at the base are FW (launched Aug–Jan) or SS; factor = LY cohort LFL move ÷ panel move ÷ the move the
   type profiles predicted for that cohort, applied ^0.5.
4. Unchanged: W14 0.5, WLONG 0.2, ramps, AGE_FADE 0.93, retail method, K 16, CH_K 20, DRIFT_ADJ 1.0, AMP floor 0.7,
   SEAS_ANCHOR 0.75, POW 0, HAL_SOFT, flash-day cap.

Forecast totals (1424 items; June actual for these items 601.4k = 20.0k/day, 30% retail; 11 items at zero):

| Month | Units | per day |
|---|---|---|
| Jul-26 | 824.3k | 26.6k |
| Aug-26 | 840.4k | 27.1k |
| Sep-26 | 572.2k | 19.1k |
| Oct-26 | 593.5k | 19.1k |
| 4-month total | 2.830M | |

By group Jul / Aug / Sep / Oct: WINTER 120 / 144 / 116 / 155k (Jun 66k), SUMMER 450 / 414 / 293 / 271k (Jun 368k),
CORE 254 / 282 / 163 / 167k (Jun 168k). Items ≈ 97% of planned site units in Jul, 88% Sep, 85% Oct.

## 3. Backtests (bt.py: full rules, 12 origins Mar-25..Feb-26 with the scoreboard item lists, site dicts = actuals;
means over origins; "summer" = mean 4-month WAPE of the May–Aug-25 origins)

| variant | WAPE | bias | ±20% | ±50% | h1 | summer |
|---|---|---|---|---|---|---|
| v13 (Mar–Jun actuals in dicts) | 0.3060 | −3.0% | 0.364 | 0.743 | 0.2385 | 0.3124 |
| AMP floor 0.85 / 1.0 | 0.3071 / 0.3162 | −1.6 / +0.3% | 0.368 / 0.367 | 0.742 / 0.738 | 0.2398 / 0.2436 | 0.3202 / 0.3399 |
| SEAS_ANCHOR 0.5 | 0.3087 | −2.2% | 0.359 | 0.737 | 0.2375 | 0.3144 |
| POW 0.05 / 0.1 | 0.3064 / 0.3140 | −2.6 / −2.1% | 0.352 / 0.319 | 0.704 / 0.648 | 0.2426 / 0.2546 | |
| AGE_FADE 0.97 | 0.3081 | −2.0% | 0.362 | 0.738 | 0.2385 | 0.3156 |
| RAMP1 1.6 / RAMP2 1.25 | 0.3104 | −1.5% | 0.362 | 0.737 | 0.2416 | 0.3215 |
| DRIFT_ADJ 1.02 | 0.3126 | +1.0% | 0.357 | 0.728 | 0.2397 | 0.3198 |
| WLONG 0.1 / CH_K 40 | 0.3059 / 0.3058 | −2.8 / −3.0% | 0.363 / 0.366 | 0.742 / 0.743 | 0.2370 / 0.2384 | |
| + cohort ^0.5 / ^0.75 / ^1.0 | 0.3054 / 0.3060 / 0.3072 | −3.2% | 0.366 | 0.743 | 0.2385 | 0.3113 |
| tags v14 (with Nov–Jan entries), cohort off / ^0.5 | 0.3046 / 0.3043 | −3.2 / −3.5% | 0.367 / 0.369 | 0.744 | 0.2381 | 0.3066 / 0.3057 |
| **v14 final (no Nov+ entries, cohort ^0.5)** | **0.3043** | −3.5% | **0.369** | 0.744 | 0.2380 | 0.3060 |

The tag / tier entries are measured on the Jun-25 origin they are tested on (in-sample; the gain is concentrated in
the summer origins, 0.3124 → 0.3060). Nov–Jan entries I first added for continuity made the Oct–Dec origins worse →
removed. The cohort rule gains little in this backtest (8 of 12 origins better, ±0.004) although the residual it
targets is large: the LY-measured cohort effect is weaker than the realised one (FW Oct factor 1.13 vs residual
1.12–1.18 over the whole window). Kept at ^0.5 as a principled, LY-measured rule; a stronger version is untested.

## 4. Known weaknesses

- **The site plan is still the forecast**, and this window contains LY's biggest non-holiday swing (Jun→Aug web
  ×1.77). If TY repeats the back-to-school event, Jul–Aug may beat the plan (LY ratio at June's YoY 1.20 → 16.7k /
  18.8k); if the event is smaller or YoY keeps falling (May–Jun 1.17–1.20 with deeper discounts), July ~14–15k.
  Check the first two weeks of July (LY Jul 1–19 averaged ~12k/day; the big leg starts Jul 20).
- **Retail is now 30% of units** and rests on LY mature-store seasonality (Sep / Oct ~0.55 of June) plus June's
  base. Not modelled: further store openings (8 in 8 months), new-store fade after the opening months (NYFIFTH first
  full month in the base), store-specific June peaks. Rollout uplift was rejected in v12 but Mar–Jun-26 under-ran
  by 7% under site actuals, partly for this reason.
- **Seasonal amplitude** is read as 1.0 now; spring 2026 swung harder than the rules (WINTER 0.73–0.79, SUMMER
  1.3–1.5 for Mar–Jun) and WINTER anchoring is +8% from June. A cold early fall or a warm September moves WINTER /
  SUMMER by ±20%.
- **Launch-season residual** only partly addressed (see §3). Young items: the Jan-26 launch wave ran 1.39 while
  LY's June launches ran 0.80–0.86 at h=1–4 — ramps are a compromise.
- Small items under / big items over (1.3–1.4 vs 0.89 in Mar–Jun) — POW still rejected on hit shares.
- Graduation items ran 3.2× in spring; LY says they fall after June (Jul–Oct 0.73–0.66 index), 16–21 items.
- Tag / tier Jul–Oct entries rest on one LY season (n = 11–68), in-sample; NYE Oct cut from 1.25 to 1.05.
- Future planners: put Jul actuals into `SITE_WEB_PD` / `OTHER_PD`; re-plan Nov (BF: LY Nov web 32.1k, the Oct-25
  origin missed −26%) and add Nov entries for Valentine's / St. Patrick / Basics / Resort-Festival measured from a
  Jul / Aug base (now silent reversals to 1.0 after Oct); watch retail rollout and TikTok.

## Changelog

- **v14 (2026-06-30)**: Mar–Jun-26 actuals in the site dicts; site plan Jul–Oct-26 web 16.0 / 17.8 / 13.8 / 14.3k
  (LY × YoY 1.15→1.12, keeping LY's back-to-school jump; May–Jun YoY 1.17–1.20) and other 4.4 / 4.35 / 4.05 / 4.45k
  (v13 overplanned May/Jun web by 8–12% and other by 9–19%). Jul–Oct occasion / tier entries from June-25-origin
  residuals: 4th of July Jun 1.17 (peak in June), Graduation / NYE lowered (NYE Oct 1.25 → 1.05), Valentine's /
  St. Patrick / Basics / Resort-Festival Jul–Oct entries added (v13 had none → silent reversal to 1.0 after June).
  New launch-season cohort index (FW / SS launched items 3–15 months old, LY-measured, ^0.5): systematic residual
  FW 1.10–1.18 / SS 0.79–1.0 in the Jun–Sep origins, reversed in Jan–Feb. Oracle backtest 0.3060 → 0.3043, summer
  origins 0.3124 → 0.3060, ±20% 0.364 → 0.369. Tested and rejected: AMP floor 0.85 / 1.0, SEAS_ANCHOR 0.5, POW 0.05 /
  0.1, AGE_FADE 0.97, RAMP1 1.6 / RAMP2 1.25, DRIFT_ADJ 1.02, WLONG 0.1, CH_K 40, cohort ^0.75 / ^1.0, cohort age
  2–12, Nov–Jan tag / tier entries.
- v13 (2026-02-28): site plan Mar–Jun-26 = 12.8 / 10.9 / 12.0 / 11.5k web/day and other 4.2 / 3.95 / 3.95 / 3.8k;
  occasion / tier indices Mar–Jun re-measured from a Feb-25 base, June added for every tag and tier; W14 0.4 → 0.5.
  Rejected: SEAS_ANCHOR 0 / 0.25 / 0.5 / 1.0, AMP floor 0.8, DRIFT_ADJ 0.98, Sweatpants trend, W14 0.6, Skirt case.
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

## History (condensed from v1–v13 notes)

- **Site level drives most misses**: Apr-25 +16%, May −7% (Memorial Day), Jul −35% (back-to-school event), Aug −12%,
  Sep +25% (post-Labor-Day lull), Oct −5% (Oct-14 flash day), Nov −26% (BF; YoY held ~1.5), Dec +8% (retail
  flagship factor too high), Jan-26 −8.5% (web YoY 1.465 vs 1.32 planned, TikTok surge), Feb-26 +2.8% (web −4.6%,
  other +27%), Mar–Jun-26 −0.2 / −3.3 / −3.8 / −7.8% (web over-planned May–Jun, store rollout un-planned). Past
  months hold actuals in the plan dicts so oracle backtests isolate the item logic.
- **2024 regime**: late July 2024 edikted.com moved to permanent deep discounts (0.45 → 0.57; LFL web ×2.3); the
  deflated-web design (LY panel drift × site plan) removes it.
- **Seasonal amplitude**: spring 2025 needed full amplitude (Jan-25 test: 1.0 best), summer 2025 ran at ~0.5 of LY's
  log-moves (but a 0.7 floor backtests better), Sep→Jan 0.8–1.0, spring 2026 swung harder than the rules (WINTER
  0.73–0.79, SUMMER 1.3–1.5 at act/pred) while Jan / Feb-26 h=1 had WINTER under (1.15 / 1.12).
- **Stores**: SoHo flagship (2024-01), LAGROVE (May-24), MNMOA (Jun-24), CAAMR / CAIRV (Nov-24), FSM (Mar-25),
  NYR (pop-up Jun-25, reopened Jul), BOCA (Aug-25), FLAVENTURA / PAKOP / VATYSONS (Nov-25), CHOAKBROOK / HOUGALLERIA
  (Apr-26), BARBIE (pop-up May-26), NYFIFTH (late May-26), FLLINCOLN / STONEBRIAR (Jun-26). LY mature stores Dec→Jan
  0.59, Feb 0.66, Mar 1.04, Apr 1.09; Jun → Jul / Aug / Sep / Oct ≈ 1.15 / 1.2 / 0.62 / 0.70.
- **Occasion tags**: Valentine's never shows a February lift; NYE items held in Jan both years; Halloween slides
  post-peak through spring; Graduation +20–30% vs segment Feb–May-25, falls in June (TY 2026 spring 3.2× v13);
  Homecoming fades after a January pop. v13 Feb-25-base raw Mar / Apr / May / Jun: 4th of July 0.99 0.95 1.03 1.14,
  Graduation 1.03 1.00 0.92 0.74, Homecoming 0.93 0.80 0.76 0.51, Halloween 0.90 0.88 0.87 0.89, NYE 0.90 0.87 0.83
  0.78, Valentine's 0.90 0.92 0.87 0.88, St. Patrick 1.30 0.94 0.74 0.78; tiers Party 0.98 0.92 0.83 0.78,
  Resort-Festival 0.99 0.83 0.84 0.96, Loungewear 0.94 0.92 0.91 0.82, Basics 1.00 1.02 1.26 1.30, Evening 0.86 0.79
  0.76 0.60 (set between this and TY's Feb hold). Jan-25-base (v12) and Dec-24-base (v11) reads: see v12 notes
  (similar shapes).
- **Segment growth**: items with no store sales, high dropship share and small items grow faster than the panel on
  average (scoreboard), but every uplift rule tested (POW, DROP_GROW, RET_ROLL) hurt WAPE / hit shares.
- **Rejected repeatedly**: POW (uniform and by horizon; v14 again), channel-share uplifts (v12), W14 0.3, ramp 1.8 /
  1.5 from a Nov base, momentum anchoring, ANCHOR_WIN 1/3/6, SEAS_ANCHOR ≤ 0.5 (v13, v14), DRIFT_ADJ > 1.0 (lower
  bias, worse WAPE; v14 again), DRIFT_ADJ 0.98 (v13), product-type trend / special cases (v13), AMP floor ≥ 0.8.
