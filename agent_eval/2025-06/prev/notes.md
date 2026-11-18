# Forecast notes: Jun–Sep 2025 per style_color (snapshot 2025-05-31)

`python rules.py` reads `data/` and writes `predictions.csv` (730 items × 4 months, ~2 s).
New research scripts in `research/`: `r1_review.py` (scoreboard by age / type / tag / size / channel mix),
`r2_site.py` (site, store, channel levels), `r3_factors.py` (factor dump), `r4_daily.py` (late-May daily vs 2024),
`r5_persist.py` (segment misses across all origins), `r6_season.py` (month-over-month group / tag LFL, 2024–25),
`r7_occasion.py` (LY tag profiles vs own segment), `bt.py` / `bt_final.py` (honest backtests, origins Jan–Apr-25),
`r8_final_diag.py` (forecast by group / cohort / tag), `write_notes.py` (builds this file).

## 0. Review of v3 (origin 2025-04 → May-25, h=1) and older origins

- **Overall: v3 May-25 pred 350k vs actual 377k, bias −7.2%, WAPE 0.213** (best h=1 so far; v2 0.276, v1 0.256).
  h=1 bias by origin: −4.5% / +15.7% / −7.2% — no systematic level miss left. ±20% share 0.45, ±50% 0.84.
- **The −7% is channel-specific, not segment-specific.** Items with retail share >40% (April mix) actual/pred
  0.92, web-heavy (<10% retail) 1.20. Site per day Apr→May: web +15% (7.9k → 9.1k), retail −11% (2.46k →
  2.19k), other +12%. v3 held the site web flat; May contains the Memorial-Day sale (May 22–27, peak 26.8k on
  May 26; 2024 had the same sale May 22–27). Panel LFL web/day Apr→May 1.06 = site 1.15 × drift 0.92, i.e. the
  drift assumption held, the site move did not. Retail: v3 assumed about 1.09, panel LFL retail/day was 0.78.
- **Young items: fine.** Mar launches (RAMP1) 0.97, Feb 0.99, Jan 1.05, Apr launches (RAMP0, n=8) 1.65.
  RAMP0 = 2.0 tested worse in the backtest; only 1 item launched in May, so RAMP0 hardly matters now.
- **Product types relative to the month total, persistent across origins (Feb/Mar/Apr → May):** Sweater
  1.28 / 1.60 / 1.43, Dress 1.33 / 1.43 / 1.31, Sweatpants / Hoodie / Jacket 1.1–1.3, Pants 0.69–0.75,
  Crop Top 0.5–0.9. Winter types were under because 2025's May winter dip was much milder than 2024's (group
  LFL rel Apr→May: WINTER 0.79 vs 0.58 LY; SUMMER 1.08 vs 1.08). Dress = prom / graduation season.
- **Occasion tags relative to total (May, origins Feb / Mar / Apr):** 4th of July 1.15 / 1.09 / 1.20,
  Graduation 1.20 / 1.28 / 1.20, Homecoming 1.00 / 1.14 / 1.20 (all dresses), St Patrick's 0.55–0.75 after
  March. Halloween / Valentine's / NYE about 1.0.
- Size quintiles (h=1): bottom 1.51, 2nd 1.20, 3rd 1.24, top two 1.06 — small items under again (persistent).

## 1. Key findings (new month)

**Panel:** 730 items; 51 launched in Apr-25, 26 Mar, 26 Feb, 53 Jan, 1 in May. They sold 403k in May
(online 297k, retail 68k, drop 39k).

**Site in May** (per day): units 13.3k, web 9.1k at discount 0.534; weekly web 7.5 / 7.7 / 7.7k, then 10.8k
(Memorial week) and 12.2k (May 26–31, mostly May 26–27); last three days 5.1–5.6k (normal Thu–Sat trough).
The weekly sale cycle (Sun–Wed ~0.54 discount, Thu–Sat ~0.45–0.49) is the same as 2024. Ex-Memorial May is about
7.8k/day, i.e. April's level. 2024 post-regime months: late Jul 11.0k, Aug 9.5k, Sep 8.6k, Oct 8.0k per day →
a back-to-school lift and a fade after Labor Day, independent of 2024's growth. June 2024 had a 3-day sale
(Jun 16–18), like Memorial Day.

**Retail May:** 2.19k/day (Apr 2.46k); all stores down vs April (SoHo 918 → 761, FSM 413 → 341); weekly
1.9 / 1.6 / 2.2 / 2.4 / 2.4k — rising into late May (schools ending). One-year store evidence for summer:
LAGROVE Jun–Aug-24 about 1.5× Sep–Oct-24, MNMOA about 1.9×; SoHo May-25 already 1.44× its fall-24 level.

**Other channels:** TikTok 15.8k → 21.3k (Apr → May), Urban 341 → 2.2k (new), Pacsun up, Flip flat at 1.4k.
Other/day +12% after three months at about −15%.

**Seasonal groups from the May base (2024 profile, relative to panel, Jun–Sep):** SUMMER 1.04 / 0.90 / 0.75 /
0.74, WINTER 0.83 / 1.13 / 1.50 / 2.10, CORE 0.97 / 1.12 / 1.29 / 1.20. 2025 runs like 2024 for summer, with a
milder winter trough (WINTER cumulative Mar–May 0.55 vs 0.39 LY).

**Occasion tags vs own segment, last year (May-24 base, research/r7):** 4th of July (n=4) Jun 1.28, Jul 1.13,
Aug 1.18, Sep 0.95; Graduation (n=8) 0.77 / 0.80 / 0.67 / 0.64; Homecoming (n=8) 0.61 / 0.87 / 1.21 / 2.68;
Halloween (n=22) 0.94 / 0.95 / 1.02 / 1.37 (Oct 2.0); Valentine's (n=12) Sep 1.53 and NYE (n=10) Sep 2.18 are
not used (no calendar reason). Panel now: 4th of July 20, Graduation 16, Homecoming 17 (no overlap), Halloween 65.

## 2. Rules (rules.py, v4)

`pred(i,m) = base_rate_i × days(m) × level_i(m) × season_seg(m) × ramp_i(h) × occasion_i(m)`

1. Base rate, seasonal profile (type × sleeve, K = 8), ramp (1.6 / 1.4 / 1.15, ×1.1 from h=2): unchanged.
2. Channel level:
   - web: as v3 (0.75 deflated + 0.25 LFL chain), but the deflated part now uses a **site path by calendar
     month** `SITE_WEB_IDX` (May 1.0, Jun 1.0, Jul 1.0, Aug 1.03, Sep 0.93) instead of a flat site.
     Result 0.90 / 0.88 / 0.87 / 0.71 (Jun–Sep).
   - retail: deflated flagship as v3, **damped f^0.6** (was 0.7): 1.30 / 1.38 / 1.40 / 0.99.
   - other: **×0.93 per month** (was 0.9): 0.93 / 0.86 / 0.80 / 0.75.
3. **Occasion as a calendar index** (`OCCASION`): multiplier = idx[target month] / idx[base month], so a peak
   already in the base month is not counted twice. From May: 4th of July 1.13 / 1.04 / 1.0 / 0.96, Graduation
   0.83 / 0.83 / 0.79 / 0.79, Homecoming 0.79 / 0.88 / 1.08 / 1.58, Halloween Sep 1.15.
4. `SEAS_ANCHOR` (season-group level anchoring) implemented but **off** (0.0) — see backtests.

Forecast totals (730 items):

| Month | Units | per day |
|---|---|---|
| Jun-25 | 406.9k | 13.6k |
| Jul-25 | 424.6k | 13.7k |
| Aug-25 | 426.8k | 13.8k |
| Sep-25 | 361.6k | 12.1k |
| 4-month total | 1.620M | |

May-25 actual for these items: 403k (13.0k/day). By group Jun / Jul / Aug / Sep: SUMMER 254 / 228 / 190 /
147k (May 235k), WINTER 51 / 74 / 99 / 111k (May 59k), CORE 102 / 122 / 138 / 103k (May 109k). Apr-25 launch
cohort 34k in May → 47 / 52 / 52 / 43k. 4th of July items 21.6k in May → 27.7 / 25.6 / 21.8 / 17.0k.

## 3. Backtests (research/bt.py) — origins Jan / Feb / Mar / Apr-25, months up to May-25

| Variant | mean WAPE | Jan / Feb / Mar / Apr (WAPE / bias) | ±20% | ±50% |
|---|---|---|---|---|
| v3 | 0.2746 | 0.299/−0% 0.313/−3% 0.274/+7% 0.213/−7% | 0.414 | 0.790 |
| new occasion index | 0.2750 | Apr 0.210 | 0.411 | 0.790 |
| SEAS_ANCHOR 0.5 | 0.2719 | Mar 0.256 / Apr 0.220 (−11%) | 0.411 | 0.799 |
| SEAS_ANCHOR 1.0 | 0.2724 | Mar 0.245 / Apr 0.232 (−15%) | 0.411 | 0.805 |
| other 0.95 / 1.0 | 0.2779 / 0.2820 | Apr 0.211 / 0.210 | | |
| RET_POW 0.5 | 0.2747 | about v3 | | |
| W_DEFL 0.5 / 1.0 | 0.2814 / 0.2719 | Apr 0.211 / 0.215 | | |
| POW 0.05 / 0.1 | 0.2718 / 0.2772 | | 0.388 / 0.339 | 0.751 / 0.678 |
| W14 0.6 | 0.2727 | Apr 0.207 | | |
| RAMP0 2.0 | 0.2762 | | | |
| **v4 chosen** | **0.2771** | 0.302/+1% 0.317/−2% 0.280/+8% 0.209/−6% | 0.407 | 0.788 |
| v4 with other 0.9 | 0.2750 | | 0.411 | 0.790 |

- None of the backtest months is summer, so the main v4 changes (site path Aug/Sep, retail damping, occasion
  profiles for Jun–Sep) cannot be tested honestly; they rest on 2024 data and the calendar. The backtest only
  shows they do no harm on Feb–May (within ±0.003).
- SEAS_ANCHOR: helps the Mar origin, hurts the Apr origin (bias to −11%) — mixed; for this origin it would cut
  WINTER by 16% while the scoreboard shows winter under-forecast at three origins. Not used.
- Other 0.93 costs 0.002 in the backtest, which is dominated by the Feb–Apr TikTok decline; May reversed it
  (+12%/day, Urban ramping). Chosen as a partial move, not the full 1.0.
- POW again lowers hit shares; W14 0.6 is a small gain but the last 14 days are Memorial-sale weighted: not used.

## 4. Known weaknesses

- **Web site level is still the main risk.** v4 assumes edikted.com holds its May per-day level (9.1k, which
  includes Memorial Day) through July, +3% in August, −7% in September. A July-4 sale weaker than Memorial Day,
  or a slide back toward April's 7.9k, leaves v4 over; a strong back-to-school leaves it under.
- Panel drift (~0.9/month relative to the site) makes old items decline steadily; Sep web factor 0.71.
- Retail summer ×1.30–1.40 is damped 2024 flagship evidence; May retail was 22% below what v3 assumed.
- Winter types / dresses were under-forecast in May at every origin; the 2024 winter rebound (WINTER ×2.1 by
  Sep relative to the panel) is used as is. If 2025's milder trough means a milder rebound, WINTER is over in
  Aug–Sep (SEAS_ANCHOR is the ready switch).
- Occasion profiles rest on 4–22 items from one year; Homecoming ×1.58 in Sep is the biggest single bet
  (17 items, ~3–5k units/month).
- Small items still under-forecast (bottom quintile ×1.5); not fixed (POW hurts hit shares).
- Future planners: Sep-25 is now a target — check the Labor Day (Sep 1) sale week and back-to-school against the
  Aug/Sep `SITE_WEB_IDX`; for Oct, Halloween (index 1.6) and Homecoming (1.9) are already in `OCCASION`.

---
# History: v3 notes (snapshot 2025-04-30), kept for reference

Research scripts in `research/`: `r1_review.py` (scoreboard review by age / type / size),
`r2_site.py` (site, channel, store levels), `r3_levels.py` (LFL channel moves this year vs last, factors used),
`r4_season_retail.py` (season-group misses, occasion tags, retail day-of-week), `r5_occasion.py` (tag profiles),
`bt1.py` / `bt2.py` (honest backtests of variants).

### v3-0. Review of v2 (scoreboard: origin 2025-03 → Apr-25 h=1; origin 2025-02 (v1) → Mar h=1, Apr h=2)

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

### v3-1. Key findings (numbers from the data)

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

### v3-2. Rules (rules.py, v3)

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

### v3-3. Backtests (research/bt1.py, bt2.py)

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

### v3-4. Known weaknesses

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
- **v4 (2025-05-31 snapshot)**:
  1. *Site web path by calendar month* (`SITE_WEB_IDX`: Jun 1.0, Jul 1.0, Aug 1.03, Sep 0.93 relative to May)
     in the deflated web factor instead of a flat site. Evidence: v3's May miss (−7%) was the site web move
     (+15% Apr→May, Memorial Day) while the panel/site drift held (0.92); 2024 post-regime Aug/Sep/Oct web/day
     9.5k / 8.6k / 8.0k show a back-to-school lift and a post-Labor-Day fade that a flat site ignores.
  2. *Retail damping 0.7 → 0.6* (Jun 1.36 → 1.30, Aug 1.48 → 1.40). Evidence: May retail LFL/day 0.78 vs about
     1.09 assumed; retail-heavy items 0.92 actual/pred vs web-heavy 1.20.
  3. *Other ×0.9 → ×0.93 per month*. Evidence: other/day +12% in May (TikTok +35%, Urban 0.3k → 2.2k) after
     three months at −15%; the backtest (Feb–Apr decline months) prefers 0.9 by 0.002, so a partial move.
  4. *Occasion tags as a calendar index* (target ÷ base month) with Graduation (post-season ×0.8), Homecoming
     (Sep ×1.58) and Halloween (Sep ×1.15) added and 4th of July re-based to May (Jun ×1.13). Evidence: LY
     tag-vs-segment profiles (research/r7) shrunk about halfway; scoreboard shows Graduation / Homecoming /
     4th of July all ×1.2 under in May (their peak is in the base month and would otherwise be carried on).
  5. `SEAS_ANCHOR` implemented, off (mixed backtest). Unchanged after testing: base blend, ramp, W_DEFL, POW off.
