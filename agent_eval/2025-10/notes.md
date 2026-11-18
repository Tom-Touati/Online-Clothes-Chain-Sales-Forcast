# Forecast notes: Nov-25 – Feb-26 per style_color (snapshot 2025-10-31), rules v9

`python rules.py` reads `data/` and writes `predictions.csv` (1003 items × 4 months, ~2 s).
Research scripts in `research/`: `r1_site.py` (site / channel / store levels per month, weekly Sep–Feb 2024-25
and Sep–Oct 2025, YoY), `r2_review.py` (Oct-25 scoreboard by group / age / type / channel share / size / tag),
`r3_factors.py` (Oct-14 spike concentration, TY vs LY group moves, factor dump, tag profiles from an Oct-24
base), `r4_drift.py` (monthly panel-vs-site drift TY vs LY, new-item share), `r5_attr.py` / `r6_seg.py`
(attribute / tier / tag profiles vs same-group and same-segment untagged items, scoreboard rel. by attribute),
`r7_fc.py` (forecast summary and variants at this origin), `bt.py` + `bt_var*.py` (backtest harness, origins
Feb–Sep-25 with the scoreboard item lists, per-item totals of complete months ≤ Oct-25), `rules_v8.py` (copy of
last month's rules).

## 0. Review of v8 (origin 2025-09 → Oct-25, h=1) and older origins

- **October was close: v8 523.0k vs actual 552.1k (941 items), bias −5.3%, WAPE 0.268** (best h=1 month of the
  year with Mar 0.256 / May 0.213 / Jun 0.255). Older origins for Oct: v7 (h=2) +18%, v6 (h=3) +12%, v5 (h=4) −26%.
- The site came in above plan (web 12.75k/day vs 11.2k, +14%; other 3.39k vs 2.6k, +30%; retail 2.79k/day =
  ×1.13 on Sep vs ×1.01 assumed), but the panel did not follow fully: with the site held at actuals (oracle
  backtest, Sep origin) v8 is **+8%** on October. Part of the site beat is a one-day flash event on **Oct 14**
  (45.0k web units kept, **145k units cancelled**; ex-spike Oct web ≈ 11.9k/day) — concentrated in a few items
  (S12749_BLACK 13× its normal day, S14947 14×, S21376 9×; items >5× took 22% of that day).
- **WINTER under / SUMMER over at h=1** (rel. to total): WINTER 1.12 (Sweater 1.25, Sweatpants 1.22, Hoodie 1.21,
  Jacket 1.42), SUMMER 0.93 (Shorts 0.87, Skort 0.88, Short Jeans 0.75). The direct check: TY Sep→Oct group moves
  rel. to panel WINTER 1.24 vs LY 1.27, SUMMER 0.88 vs 0.90 — **the first month of 2025 at LY's full seasonal
  amplitude** (Jun–Sep ran at ~0.5). v8's measured amplitude 0.51 was therefore too small for October; the
  adaptive rule now reads 0.67 (3-month window) and rises automatically. h=2/h=3 origins (Aug/Jul, SEAS_POW-era
  settings) had WINTER 0.96 / 0.92 — over-forecast — so this is a flip, not a stable bias; kept the measured rule.
- Tags (rel., h=1): Halloween 0.89 (77 items, 119k pred: the Oct peak ×1.83 on Sep vs LY 2.09), NYE 0.81 (also
  0.80 in Aug, 1.11 in Sep), Homecoming 1.12 (small), Valentine's 0.98, Graduation 0.86, 4th of July 0.98.
- Size: bottom four pred-quintiles 1.05–1.09 rel., top 0.97 — the same persistent skew (POW still rejected).
  Age 4–8 months 0.89 (one month, not seen before — noise for now). Web-only items 1.09, retail 25–50% 0.91.
- Scoreboard WAPE by horizon across all origins: h=1 0.21–0.38, h=2 0.33–0.44, h=3 0.40–0.60, h=4 0.46–0.59 —
  the long-horizon misses are dominated by the site level (Apr–Jun origins missed the July regime / events).

## 1. Key findings (new month)

**Site Oct** (per day): units 18.9k, web 12.75k (disc 0.546), retail 2.79k, other 3.39k; cancelled 5.1k/day (all
Oct 14–15). Weekly web 15.1 / 19.6 / 11.4 / 7.05k; Oct 30–31 4.3k / 2.3k (LY's last October week 4.1k/day:
the pre-November lull is seasonal). **YoY web/day Jun 2.0, Jul 1.80, Aug 1.65, Sep 1.42, Oct 1.60 (1.50
ex-spike)**; sale weeks 1.43–1.75× LY, quiet weeks 1.47–1.72×. LY from Oct-24: Nov 19.5k (BF week 54k/day),
Dec 17.7k (Cyber Monday week 30k/day), Jan-25 9.6k, Feb-25 10.8k (disc 0.60–0.61).
**Other** 3.39k/day: TikTok 46.6k (LY 33.8k), Pacsun 29.8k (19.6k), Nordstrom 10.2k, Macy's 10.2k (5.4k), Urban
3.5k. LY ex-Flip from Oct: Nov ×1.58, Dec ×2.44 (Pacsun 69.6k, Macy's 22.4k, Nordstrom 28.6k), Jan ×0.94, Feb ×0.84.
**Retail** 2.79k/day; SoHo 859/day (Sep 662), all 8 stores trading Oct 31. LY same-store Oct→Nov / Dec / Jan:
SoHo 1.15 / 1.45 / 0.83, LAGROVE 1.16 / 1.10 / 0.59. Rule factor 1.01 / 1.19 / 0.73 / Feb 0.71.
**Panel drift** (LFL per-day move ÷ site per-day move): TY Jul–Oct 0.950 / 0.948 / 0.899 / 0.944 vs LY 0.929 /
0.906 / 0.899 / 0.900 (TY milder); LY Nov / Dec / Jan / Feb 0.881 / 0.976 / 0.941 / 0.963. Rule drift from
Oct-24 cumulated: 0.889 / 0.877 / 0.834 / 0.811.
**Season groups** (LY LFL rel. to panel from Oct-24, Nov / Dec / Jan / Feb): WINTER 1.78 / 1.85 / 1.60 / 1.36,
SUMMER 0.65 / 0.61 / 0.76 / 0.88, CORE 0.88 / 0.88 / 0.90 / 0.95. Amplitude 0.67, anchoring WINTER 1.06 /
SUMMER 0.95 / CORE 1.02.
**Tags from an Oct-24 base** (tagged ÷ untagged, same group | same segment, Nov / Dec / Jan / Feb):
Halloween (n=38) 0.35 0.36 0.39 0.45 | 0.39 0.43 0.47 0.49; split by tier: Party/Evening-tier Halloween items
(n=18) 0.37 0.41 0.45 0.41, other tiers (n=20) 0.73 0.78 0.78 0.88. Halloween items' last-14-day rate is 0.94 of
their October rate vs 0.75 for the rest (the base blend lifts them ~8%). Homecoming (12) 0.65 0.58 1.35 0.96 |
0.86 0.80 1.26 0.85. NYE w/o Homecoming (7) 0.54 0.70 0.87 1.19 | 0.79 1.13 0.94 1.31 (no December lift).
Valentine's (26) 1.07 0.97 1.05 0.90 | 1.08 1.02 1.02 0.90 (no pre-Valentine lift; Feb is lower). Graduation
0.78 0.79 1.10 1.33; 4th of July 1.21 1.24 1.13 1.25.
**Occasion tier / attributes (same segment):** Party tier without Halloween tag (n=77) 0.78 / 0.76 / 0.82 /
0.82; Evening tier without Halloween ~1.0–1.2; sheer 0.44 (mostly Halloween overlap); lounge 1.08 / 1.25
(captured by segments). No Christmas / holiday-party tags exist.

## 2. Rules (rules.py, v9)

`pred(i,m) = base_rate_i × days(m) × level_i(m) × season_seg(m) × ramp_i(h) × occasion_i(m)` — structure as v8.

1. **Site plan** (`SITE_WEB_PD`, web/day): Oct actual 12,751; **Nov 26.5k** (YoY 1.36; BF Nov 28), **Dec 23.3k**
   (1.32), **Jan-26 11.5k** (1.20), **Feb-26 12.0k** (1.11). YoY decays from Oct's 1.50 ex-spike as LY's growth
   and LY's deep-discount Jan–Feb annualise. Web factor 1.82 / 1.55 / 0.73 / 0.73 (incl. drift).
2. **Other plan** (`OTHER_PD`): Oct actual 3,390; Nov 4.5k, Dec 6.3k, Jan 2.7k, Feb 2.4k (LY ex-Flip shape, YoY
   1.17 / 1.06 / 1.18 / 1.17) → factor 1.18 / 1.62 / 0.65 / 0.57.
3. **DRIFT_ADJ 1.01 → 1.0** (backtest below; the direct TY drift is milder than LY, but the oracle bias at the
   recent origins is positive, so no extra lift).
4. **Flash-day cap** (`SPIKE_X` 2.5, `SPIKE_CAP` 2.0): on a base-month day where the site sold > 2.5× its median
   day, an item (live ≥ 14 days) may sell at most 2 × site ratio × its own normal day; the base is then rescaled
   to the pre-cap level (the site level of the base month still contains the flash day). Oct 14 only; moves
   ~5k units from S12749_BLACK (−2.2k), S21376_BLACK, S14947_DARK-BLUE, S14833_BROWN to the rest.
5. **Occasion index from an October base, February added**: Halloween Oct 2.6 → 2.4, Nov 0.9 / Dec 0.92 / Jan 1.0 /
   Feb 1.15 (×0.375 / 0.383 / 0.417 / 0.48 from Oct); **`HAL_SOFT` 0.5**: Halloween items outside Party/Evening tier
   get the index swing ^0.5 (×0.61 in Nov). Homecoming ×0.80 / 0.75 / 1.0 / 0.9; NYE ×0.90 / 1.0 / 0.95 / 1.0
   (v8 had a December lift ×1.16); Graduation Feb ×1.125; 4th of July ×1.06.
6. **Party-tier step** (`TIER_IDX` {Sep 1.1, Oct 1.1}): Party-tier items without the Halloween tag ×0.91 from an
   October base in Nov–Feb (LY 0.78–0.82, shrunk about halfway); Sep→Oct neutral.
7. Unchanged: base blend, ramp, K, channel shrink, adaptive amplitude (now 0.67) + anchoring, RET_POW 0.8,
   new / closed store rules, POW 0.

Forecast totals (1003 items; Oct actual for these items 578.5k = 18.7k/day, Sep 528.8k):

| Month | Units | per day |
|---|---|---|
| Nov-25 | 850.7k | 28.4k |
| Dec-25 | 852.9k | 27.5k |
| Jan-26 | 404.8k | 13.1k |
| Feb-26 | 359.2k | 12.8k |
| 4-month total | 2.468M | |

By group Nov / Dec / Jan / Feb: WINTER 391 / 407 / 171 / 135k (Oct 167k; share 46 / 48 / 42 / 38% vs LY Nov–Jan
47 / 49 / 42%), SUMMER 238 / 227 / 128 / 126k (Oct 252k), CORE 222 / 220 / 106 / 97k (Oct 160k). Halloween (78
items, Oct 112k) → 63 / 62 / 33 / 36k; NYE (44) 25 / 27 / 14 / 14k; Valentine's (89) 57 / 55 / 30 / 29k.
Fixed amplitude 0.8 would add +17k WINTER in Nov; DRIFT_ADJ 1.01 would add +1–4%.

## 3. Backtests (research/bt.py) — origins Feb–Sep-25, per-item totals of complete months ≤ Oct-25

Oracle site levels (`SITE_WEB_PD` / `OTHER_PD` hold actuals for past months): tests drift, segments, amplitude,
retail, tags — not the site assumptions. Means over 8 origins.

| Variant | mean WAPE | bias | ±20% | ±50% | h=1 WAPE | per-origin WAPE Feb…Sep |
|---|---|---|---|---|---|---|
| v8 | 0.3021 | +0.8% | 0.358 | 0.745 | 0.2383 | .322 .303 .297 .273 .328 .266 .345 .282 |
| v8 + DRIFT 1.0 / 0.99 | 0.2997 / 0.2982 | −0.9 / −2.5% | 0.365 / 0.366 | 0.753 / 0.758 | 0.238 | |
| v8 + AMP_WIN 2 / 1 | 0.3031 / 0.3035 | | 0.361 | 0.743 | 0.238 / 0.237 | |
| v8 + amp floor 0.6 | 0.3024 | | 0.356 | 0.746 | | |
| v8 + RET_POW 0.6 / 1.0 | 0.3035 / 0.3017 | | | | | |
| v8 + SEAS_ANCHOR 0.25 | 0.3088 | | | | | |
| v8 + W14 0.3 / WLONG 0.3 | 0.3035 / 0.3036 | | | | | |
| v8 + POW 0.03 | 0.3004 | | 0.355 | **0.719** | | |
| **v9** | **0.2965** | −2.5% | 0.365 | 0.755 | 0.2359 | .329 .301 .296 .272 .331 .260 .327 .257 |
| v9 without HAL_SOFT | 0.2991 | −1.4% | 0.364 | 0.754 | | Sep .269 |
| v9 without Party-tier step | 0.2959 | | 0.367 | 0.757 | | Aug .325 |
| v9 DRIFT 1.01 | 0.2982 | −0.9% | 0.358 | 0.747 | | |
| v9 without flash cap / cap 3 | 0.2965 | | | | | (Labor Day Sep 1 barely trimmed) |

- DRIFT 0.99 is marginally better on WAPE but pushes bias to −2.5% before v9's other changes; 1.0 chosen.
- HAL_SOFT and the Halloween October re-base help mostly at the Sep origin (in-sample for October; the split
  itself comes from the LY Nov–Feb profiles, which no backtest can test).
- The Party-tier step costs 0.0006 at the Aug origin (Aug→Sep ×1.1); kept for its Nov–Feb evidence (LY only).
- Nov–Feb cannot be tested inside 2025 data; the site plan, amplitude and post-October tag choices rest on the
  2024-25 profiles and Sep–Oct 2025.

## 4. Known weaknesses

- **The site plan is still the forecast.** Nov + Dec are 69% of the 4-month total. Nov 26.5k web assumes BF at
  YoY 1.36 (event weeks in 2025 ran 1.43–1.75× LY; if BF week repeats LY's 54k/day ×1.5 Nov web is ~29k, +9%).
  If YoY keeps falling at the Jun–Sep pace (−0.15/month), Dec/Jan are 10–15% over. Jan/Feb have no 2025 analogue
  in the deep-discount regime other than Jan–Feb-25 themselves.
- **Amplitude flip risk**: 0.51 → 0.67 in one month. If Nov's WINTER step is at LY's full amplitude (as Sep→Oct
  was), WINTER Nov is ~+20k under; if the summer-like 0.5 returns, ~−25k over. Re-measure after November.
- Other Dec 6.3k/day depends on holiday wholesale orders (Pacsun / Macy's / Nordstrom) repeating at LY's ×2.4.
- Halloween post-peak (×0.375; 63k in Nov) and the HAL_SOFT split rest on n=18 / 20 LY items; the Party-tier
  step on n=77 LY items. NYE / Homecoming on n=7 / 12.
- Flash days: the Oct-14 event (145k cancelled) suggests promo-code / pricing incidents; the cap only fixes item
  distribution, not the site level (which carries the flash day in `web_B`, consistent with the plan).
- Persistent size skew (smallest items under-forecast ~1.05–1.3 at h=1; POW trades away ±50% hits).
- `rules.py` 414 lines (slightly over ~400, mostly comments / evidence).
- Future planners: check Nov web/day vs 26.5k and the BF week; WINTER rel. at h=1 (Nov assumed 1.78^0.67 × 1.06
  ≈ 1.55 vs panel); Halloween items in Nov (×0.375 / ×0.61 assumed); add each month's actuals to `SITE_WEB_PD` /
  `OTHER_PD`; extend the plan and `OCCASION` to March (St. Patrick's Day tag exists, n=4 LY).

## Changelog

- **v9 (2025-10-31)**: site plan re-based on Oct actual (12.75k web/day vs 11.2k planned; Oct-14 flash day),
  Nov–Feb 26.5 / 23.3 / 11.5 / 12.0k; other 4.5 / 6.3 / 2.7 / 2.4k. DRIFT_ADJ 1.01 → 1.0 (backtest 0.3021 →
  0.2997). Flash-day cap on item base rates (S12749 etc. 9–14× their normal day on Oct 14). Occasion indices
  re-based from an Oct-24 base with Feb added: Halloween Oct 2.6 → 2.4 (Oct rel 0.89) and post-peak ×0.375–0.48,
  softer (^0.5) outside Party/Evening tier; NYE December lift removed (Oct rel 0.81, LY no Dec lift); Homecoming,
  Graduation, 4th of July adjusted. Party-tier (non-Halloween) ×0.91 after October. Backtest mean 0.3021 → 0.2965,
  h=1 0.2383 → 0.2359, ±50% 0.745 → 0.755. Adaptive amplitude kept (reads 0.67 now; v8 0.51 under-forecast
  WINTER in Oct, rel 1.12).
- v8 (2025-09-30): site plan from Sep; adaptive seasonal amplitude (`ADAPT_AMP`, 0.51); RET_POW 0.6 → 0.8;
  occasion indices from a Sep base with January added.
- v7 (2025-08-31): site plan from Aug, SEAS_POW 0.8, ANCHOR_LONG, DRIFT 1.01, December occasion entries.
- v6 (2025-07-31): web / other = LY panel drift × explicit site plan in units/day (`SITE_WEB_PD`, `OTHER_PD`);
  NYE tag; `DRIFT_ADJ`, `ANCHOR_WIN` switches.
- v5 (2025-06-30): fully deflated web (W_DEFL 1.0), pop-up / closed store rule, K 16, SEAS_POW 0.9, anchoring 0.5,
  other ×0.90/month.
- v4 (2025-05-31): site web path by calendar month, RET_POW 0.6, other ×0.93, occasion tags as calendar index.
- v3 (2025-04-30): deflated web level (LY LFL ÷ LY site web), asymmetric pre-anchor, other ×0.9/month, RET_POW
  0.7, 4th of July multiplier.
- v2 (2025-03-31): July-24 discount-regime fix for the web chain, retail two-anchor, new-store run-rate, graded
  ramp, parameter dict.
- v1 (2025-02-28): base blend, channel-weighted level, product-type × sleeve seasonal profile shrunk to season
  group, young-item ramp, zero for items with no sales in 28 days.

## History (condensed from v1–v8 notes)

- **Site level drives most misses**: Apr-25 +16% (web LFL 0.75 vs 0.94), May −7% (Memorial Day), Jul −35%
  (back-to-school event, web ×1.57), Aug −12% (event ran to Aug 20), Sep +25% (post-Labor-Day weeks 9–10k/day;
  retail LFL 0.45 vs 0.69), Oct −5%. Planned site levels are now explicit per month; past months hold actuals.
- **2024 regime**: late July 2024 edikted.com moved to permanent deep discounts (0.45 → 0.57; LFL web ×2.3); the
  Jun→Jul-24 link is replaced by the discount-matched Jul 8–21 lift and the web level is deflated by LY's site
  growth. Feb-24 was a retail promo month (low pre-anchor weight).
- **Seasonal amplitude**: spring 2025 needed full/greater amplitude (WINTER under in May/Jun ×1.09–1.61); summer
  2025 ran at ~0.5 of LY's log-moves (WINTER over Jul–Sep); Sep→Oct back to ~1.0 → measured each month.
- **Stores**: SoHo flagship (2024-01), LAGROVE (May-24), MNMOA (Jun-24), CAAMR / CAIRV (Nov-24), FSM (Mar-25),
  NYR (pop-up Jun-25, reopened Jul), BOCA (Aug-25). New stores in the base month count at their run-rate;
  stores with no sale in the last 5 days are treated as closed.
- **Rejected repeatedly**: POW (regression to mean; costs ±50% hits), W14 0.3, ramp 1.8 / 1.5, momentum
  anchoring, ANCHOR_WIN 1/3/6, SEAS_ANCHOR 0.25.
