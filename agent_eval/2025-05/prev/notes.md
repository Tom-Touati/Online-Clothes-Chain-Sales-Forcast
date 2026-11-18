# Forecast notes: May–Aug 2025 per style_color (snapshot 2025-04-30)

`python rules.py` reads `data/` and writes `predictions.csv` (686 items × 4 months, ~1.5 s).
Research scripts in `research/`: `r1_review.py` (scoreboard review by age / type / size),
`r2_site.py` (site, channel, store levels), `r3_levels.py` (LFL channel moves this year vs last, factors used),
`r4_season_retail.py` (season-group misses, occasion tags, retail day-of-week), `r5_occasion.py` (tag profiles),
`bt1.py` / `bt2.py` (honest backtests of variants).

## 0. Review of v2 (scoreboard: origin 2025-03 → Apr-25 h=1; origin 2025-02 (v1) → Mar h=1, Apr h=2)

- **Overall level is the systematic miss.** Apr-25: v2 pred 392k vs actual 338k (**bias +15.7%**, WAPE 0.276);
  v1 at h=2 +11%. Old items (12+ months) actual/pred 0.86 in both — i.e. a level miss, not a segment miss.
- **By channel (LFL per day, Mar→Apr-25):** web **0.75** (v2 assumed 0.94), retail 0.97 (assumed 0.91,
  slightly under-forecast), other 0.85 (assumed 1.0). Web is the whole miss, other adds a little.
- Why web missed: v2's web factor was last year's LFL chain. Last year the site was growing fast
  (web/day 2.8k Jan-24 → 4.4k Jun-24, then the July discount regime), so LY LFL moves carry that growth.
  This year the site web/day is falling (10.8k Feb → 9.7k Mar → 7.9k Apr). The pre-anchor ("rebound to
  March") also pushes the wrong way when the base month is a persistent step down.
- **Young items: no systematic miss now.** Origin Mar: Jan launches 0.98, Feb launches 0.74, Dec 0.79,
  Mar 1.59 (n = 4) vs old 0.86. Young items overall 0.90 vs old 0.86 → ramp about right. Kept.
- **Season groups: noise.** Relative to the total miss, Apr: SUMMER 1.00 / 1.06, WINTER 1.04 / 0.88, CORE
  0.97 / 0.99 (origin Mar / Feb). March had the opposite sign (winter over). No change.
- Product types (relative to 0.86): Jeans 0.88, Shorts 0.92, Top 0.91, Tank 1.07, Short Jeans 1.29, Hoodie
  1.04 — mixed and small-n; no change.
- Size quintiles flat at h=1 (0.81–0.95); at h=2 top quintile 0.84, bottom 1.46 (as before; not acted on).
- Occasion: 4th of July tag (n = 15) 1.08 / 1.17 relative in April, Graduation 1.04 / 1.09.

## 1. Key findings (numbers from the data)

**The panel.** 686 items; 8 launched in Apr-25, 27 in Mar, 26 in Feb, 52 in Jan. They sold 355k in April
(web 229k, retail 73k, other 52k), 11.8k/day; last 14 days 11.2k/day.

**Site in April** (per day): units 12.1k (Mar 13.9k), web 7.9k (Mar 9.7k) at discount 0.557 (0.569),
retail 2.46k (2.33k), other 1.77k (1.96k). The web drop happened in the last week of March (7.2k/day) and
April was then flat by week (8.0 / 8.5 / 7.3 / 7.8 / 9.0k) — a step, not a continuing slide.
Web peaks on Sundays (10–11.6k) independent of discount; discount explains less than in winter.
Other: TikTok 28k → 20k → 16k (Feb–Apr), Flip 4.4k → 1.5k; Nordstrom / Pacsun up; Urban new (341).

**Panel drift relative to the site is stable across both years.** LFL web/day ÷ site web/day:
2025 Jan→Feb 0.96, Feb→Mar 0.94, Mar→Apr 0.91; 2024 Feb→Mar 0.93, Mar→Apr 0.86, Apr→May 0.94,
May→Jun 0.90, Jun→Jul 0.93, Jul→Aug 0.91. The *site* move is what differs between years (growth +
regime in 2024). Hence the deflated factor: LY (LFL/site) chain × assumed site trend. For this origin:
0.94 / 0.84 / 0.78 / 0.71 (May–Aug), and it removes the July-24 regime jump without a special fix.

**Retail.** Spring break week (Apr 14–20) 3.3k/day; the last three days (Mon–Wed Apr 28–30) 1.0–1.1k/day,
~30% below earlier April Mon–Wed. SoHo 918/day in April (already above Aug-24's 794). 2024 summer
retail is strongly seasonal (LAGROVE Jun–Aug ~1.5× Sep–Oct, MNMOA ~2.2×), but SoHo 2024 was still ramping
(Sep–Oct 2024 1.9× Mar–Apr 2024), so the flagship-based factor overstates seasonality.

**Seasonal mix this spring runs like last year, slightly earlier.** Group LFL relative to panel:
2025 Feb→Mar SUMMER 1.28 / WINTER 0.75, Mar→Apr 1.20 / 0.70; 2024 1.19 / 0.78 and 1.15 / 0.67.
Profile for this origin (Apr-24 → m): SUMMER 1.08 / 1.13 / 0.97 / 0.81, WINTER 0.58 / 0.48 / 0.65 / 0.87,
CORE 1.01 / 0.98 / 1.13 / 1.30 (back-to-school shift in Jul/Aug, as noted in v2).

**Occasion tags in the panel:** Halloween 62, Valentine's 58, NYE 24, 4th of July 18, Homecoming 15,
Graduation 13. Last year 4th-of-July items ran ×1.32 / 1.63 / 1.47 / 1.13 vs same-group items in May–Aug
(n = 3); Homecoming ×0.40–0.75 in summer then 2.6 in Sep (n = 8, not used); Graduation fell after April
(n = 8, not used).

## 2. Rules (rules.py, v3)

`pred(i,m) = base_rate_i × days(m) × level_i(m) × season_seg(m) × ramp_i(h) × occasion_i(m)`

1. **Base rate** — unchanged (0.6 last month + 0.4 last 14 days, level-neutral, 80/20 with last 91 days;
   zero if nothing sold in 28 days).
2. **Channel level** (item's base-month mix shrunk to panel with 20 pseudo-units):
   - **web (new)**: 0.75 × deflated factor (LY LFL ÷ LY site web, site held flat) + 0.25 × v2 LFL chain
     (July regime fix kept). The pre-anchor term may only lower the factor (`PRE_ASYM`).
     Result: 0.95 / 0.87 / 0.85 / 0.83 (v2 logic would give 1.13 / 1.05 / 1.18 / 1.32).
   - **retail**: deflated flagship as v2, pre-anchor asymmetric, **damped f^0.7 (new)**:
     1.09 / 1.48 / 1.59 / 1.61 (undamped 1.13 / 1.75 / 1.94 / 1.98).
   - **other (new)**: ×0.9 per month, compounded (0.90 / 0.81 / 0.73 / 0.66).
3. **Seasonal profile** — unchanged (product type × sleeve, shrunk to group with K = 8, cap [0.15, 4]).
4. **Ramp** — unchanged (1.6 / 1.4 / 1.15, ×1.1 from h = 2).
5. **Occasion (new)**: 4th of July tag ×1.25 in June and July.

Forecast totals (686 items):

| Month | Units | per day |
|---|---|---|
| May-25 | 349.9k | 11.3k |
| Jun-25 | 360.0k | 12.0k |
| Jul-25 | 374.0k | 12.1k |
| Aug-25 | 362.7k | 11.7k |
| 4-month total | 1.447M | |

April-25 actual for these items: 355k (11.8k/day). By group May / Jun / Jul / Aug: SUMMER 209 / 233 / 214 /
173k (Apr 191k), WINTER 40 / 33 / 47 / 62k (Apr 65k), CORE 100 / 94 / 113 / 128k (Apr 98k).

## 3. Backtests (research/bt1.py, bt2.py)

Honest out-of-time: origins Jan-25, Feb-25, Mar-25, data up to each origin only, scored on per-item totals
over the target months that are complete (≤ Apr-25: 3 / 2 / 1 months). Items: scoreboard items (Feb, Mar),
items selling in the last 60 days (Jan). Small sample, overlapping months — read the direction, not decimals.

| Variant | mean WAPE | Jan-25 WAPE / bias | Feb-25 | Mar-25 |
|---|---|---|---|---|
| v2 | 0.326 | 0.396 / +27% | 0.305 / +4% | 0.276 / +16% |
| v2 + asymmetric pre-anchor | 0.318 | 0.396 / +27% | 0.282 / −7% | 0.276 / +16% |
| pre-anchor off | 0.329 | | | 0.307 / +21% |
| deflated web 0.5 | 0.287 | 0.315 / +13% | 0.290 / −2% | 0.257 / +13% |
| deflated web 1.0 | 0.271 | 0.268 / −1% | 0.306 / +4% | 0.240 / +9% |
| deflated 1.0 + other 0.9 | 0.263 | 0.259 / −4% | 0.298 / +2% | 0.233 / +8% |
| deflated 0.75 + other 0.9 | 0.267 | 0.273 / +3% | 0.290 / −1% | 0.240 / +9% |
| deflated 1.0 + other 0.85 | 0.260 | | | |
| + site web 0.95 (trend extrapolation) | 0.255 | −8% | −2% | +4% |
| **v3 chosen (defl 0.75, other 0.9, retail^0.7, 4th July)** | **0.268** | 0.268 / +1% | 0.290 / −1% | 0.244 / +10% |

- Deflated web is the big win (−0.06 WAPE, bias from +27/+4/+16% to ~0/−1/+10%).
- 0.75 instead of 1.0: costs ~0.004 in the backtest but keeps a quarter of last year's summer web seasonality
  (July-4 sale, back-to-school), which the Feb–Apr backtest months cannot test.
- Other 0.85 scored slightly better than 0.9; 0.9 chosen as the less extreme of two similar results.
- Site-trend extrapolation (0.95/month) helps only because Feb–Apr was a decline; not adopted.
- Ramp (1.3/1.25/1.1 and 1.8/1.5/1.2), K 4/16, W14 0.6, WLONG 0.1/0.3, CH_K 5, POW 0.05: all within
  ±0.005 of the base; POW costs 5 points of ±50% share again. No change.
- Retail damping and the 4th-of-July multiplier cannot be tested (no summer in the honest window).

## 4. Known weaknesses

- **Web site level is still the main risk.** v3 assumes edikted.com stays at its April per-day level (~7.9k).
  It fell 27% Feb→Apr; a further slide would leave v3 over again, a summer sale / back-to-school lift
  (2024: Jul ×1.75, Aug ×2.15 site vs Jun, regime included) would leave it under. Every 0.1 on web ≈ 6–7% units.
- The deflated factor implies the panel's web sales fall ~8%/month (to 0.71 by August before the 25% LFL
  blend). This is consistent across 2024 and 2025 but means old items decline; if new launches stop, survivors
  may hold up better.
- **Retail summer** ×1.5–1.6 (Jun–Aug) still rests on 2024 SoHo/LAGROVE/MNMOA; damping exponent 0.7 is a
  judgment. Late-April retail was ~30% below early-April weekdays (post spring-break lull) — May could be soft.
- Other ×0.9/month assumes TikTok/Flip keep declining; new marketplaces (Urban started in April) are upside.
- 4th-of-July ×1.25 rests on 3 items last year plus calendar reasoning.
- Small items under / top quintile over at h ≥ 2 (persistent, not fixed). Seasonal profiles: one year,
  small segments. Homecoming/Graduation/Halloween tags not used (peaks outside the horizon or thin evidence).
- Future planners: when Sep-25 becomes a target, the LY Aug→Sep-24 site move (×0.91 web/day, Sep 2–8
  sale week at 2.3×) and Homecoming (Sep ×2.6 last year) need a look.

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
- **v3 (2025-04-30 snapshot)**:
  1. *Deflated web level* (`W_DEFL` = 0.75, `SITE_WEB` = 1.0): web factor = 0.75 × (LY LFL ÷ LY site web
     per day, site held flat) + 0.25 × v2 LFL chain. Evidence: Apr-25 bias +15.7%, entirely web (LFL 0.75 vs
     0.94 assumed); panel/site drift is stable (0.86–0.96 per month in both years) while LY's site growth was
     not repeated; honest backtest mean WAPE 0.326 → 0.27, bias ~0 at Jan/Feb origins.
  2. *Asymmetric pre-anchor* (`PRE_ASYM`): the "month before base" anchor may only lower web/retail factors.
     Evidence: the April step down persisted all month (weekly 7.3–9.0k); v2's anchor would have read it as a dip
     and raised May web to ×1.31. Backtest 0.326 → 0.318.
  3. *Other channel ×0.9 per month* (`OTHER_MONTHLY`): other LFL/day 0.86 / 0.84 / 0.85 for Feb / Mar / Apr-25
     (TikTok −44% Feb→Apr); backtest −0.008 WAPE.
  4. *Retail damping* (`RET_POW` = 0.7): 2024 flagship factors include the store's own ramp-up; Jun 1.75 → 1.48.
     Mar-origin April retail was under-forecast (0.91 vs 0.97), which damping also reduces.
  5. *4th of July occasion multiplier* ×1.25 Jun/Jul (`OCCASION` dict).
  Unchanged after testing: base blend, ramp, shrinkage K, channel shrink, POW off.
