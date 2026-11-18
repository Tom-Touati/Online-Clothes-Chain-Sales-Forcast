# Forecast notes: Jul–Oct 2025 per style_color (snapshot 2025-06-30)

`python rules.py` reads `data/` and writes `predictions.csv` (768 items × 4 months, ~2 s).
Research scripts in `research/`: `r1_review.py` (scoreboard by group / age / channel mix / size / type / tag),
`r2_site.py` (site, channel, store levels; 2024 Jun–Oct weekly), `r3_factors.py` (factor dump, new-store check),
`r4_occasion.py` (tag profiles from a June-24 base), `r5_diag.py` (party-tag check, forecast by group / tag /
cohort), `bt.py` (honest backtest, origins Jan–May-25, scored on per-item totals of complete months ≤ Jun-25).

## 0. Review of v4 (origin 2025-05 → Jun-25, h=1) and older origins

- **Overall: v4 June pred 407k vs actual 383k, bias +6.1%, WAPE 0.255** (h=1 by origin: −4.5 / +15.7 / −7.2 /
  +6.1% — alternating sign, no systematic level miss at h=1). First complete 4-month total (origin Feb-25, v1):
  WAPE 0.370, bias +7%, ±20% 0.32. Longer horizons of v1/v2 were over (+19% / +34% in June); v3 at h=2 +4%.
- **By channel (panel, May → Jun per day):** online −9%, retail **+38%**, drop −19%. v4 assumed web 0.90, retail
  1.30, other 0.93 → web slightly over, other over, retail about right. Site web/day 9.10k → 8.83k (×0.97, v4
  assumed 1.0; June had a 3-day sale Jun 15–17 at 20–23k/day, like 2024).
- **Seasonal amplitude is the systematic miss.** June by group relative to the total, origins Feb / Mar / Apr /
  May: WINTER 1.30 / 1.61 / 1.45 / 1.09 (under at every origin), SUMMER 0.94 / 0.90 / 0.93 / 0.95 (over), CORE
  ~1.03. Same in May (Sweater ×1.3–1.6). Types: Shorts 0.67–0.81 and Strapless 0.64–0.80 over at all four origins;
  Skirt 1.16–1.40, Short Jeans 1.14–1.59, Swimwear 1.4–2.9 (small) under. 2025's swings are milder than 2024's
  (2024 was also a growth year for summer types), so last year's profile over-amplifies.
- Young items fine (age 1–2 months 1.15, 2–3 months 0.95). Channel-mix buckets 0.95–1.07: noise.
- Size: bottom pred-quintile under (×1.14 at h=1, ×1.7–2.8 at h=2–4), top quintile 0.90–0.96 (persistent).
- Tags (June, rel. to total): 4th of July 0.97 at h=1 (fixed by the v4 index), Graduation 0.85 (still over),
  Homecoming 1.10, Halloween / Valentine's 1.0, NYE 0.88.

## 1. Key findings (new month)

**Panel:** 768 items, June-25 actual 407k (13.6k/day); launches: 51 Apr, 36 May, 3 Jun.
**Site June** (per day): units 13.8k, web 8.83k (disc 0.514), retail 3.14k (May 2.19k), other 1.81k (drop 1.06k).
Weekly web 7.9 / 7.7 / 11.5 (sale) / 8.3k — the same Sun–Wed / Thu–Sat cycle as before.
**Retail summer is real:** every store up May → Jun per day: SoHo 761 → 1032, LAGROVE 334 → 501, MNMOA 242 → 432,
CAIRV 345 → 436, CAAMR 171 → 238, FSM 341 → 421. **NYR is a 4-day pop-up (Jun 19–22, 2.25k units)** — v4's
new-store run-rate rule would have counted it as a permanent 690/day store (retail ×1.197 instead of ×0.976).
2024: retail per day Aug 1.82k → Sep 1.07k (×0.58, back to school); the flagship-based factor now gives
Jul 1.04 / Aug 1.05 / Sep 0.74 / Oct 0.75 from the June base.
**Other:** TikTok flat (21.3k → 21.7k), Pacsun −24%, Nordstrom −28%, Flip −40%; Urban flat 2.2k.
**Web from June:** deflated factor Jul 0.93 / Aug 0.87 / Sep 0.70 / Oct 0.59 (panel drift ×~0.92/month plus site
path Aug 1.03, Sep 0.93, Oct 0.87). The 25% LY-LFL part of v4 was inflated for Aug–Oct (a_feb 1.25 for Aug):
the Jul→Aug-24 link still contains the late-July-24 regime jump (only Jun→Jul is fixed).
**Occasion tags, June-24 base vs own segment (Jul / Aug / Sep / Oct):** 4th of July (n=4) 0.85 / 0.83 / 0.61 /
0.55; Graduation (10) 1.04 / 0.83 / 0.75 / 0.71; Homecoming (11) 1.19 / 1.74 / 4.71 / 3.40; Halloween (26) 1.06 /
1.33 / 1.94 / 4.45. The May→Jun-25 tag moves repeated LY's (4th July 1.20 vs 1.34, Graduation 0.77 vs 0.67,
Homecoming 0.63 vs 0.58, Halloween 1.05 vs 0.95). Valentine's (14) and NYE (11) also jumped in Sep-24 (2.4 / 4.2)
but are mostly dresses co-tagged Homecoming (NYE 6 of 11) — homecoming-dress effect, not used separately.
**Season-group anchoring** (Apr→Jun this year vs last): WINTER 0.81, SUMMER 1.01, CORE 0.99 — winter fell less
this spring, so its LY rebound (WINTER rel. Oct ×3.2) is over-stated from the higher base.

## 2. Rules (rules.py, v5)

`pred(i,m) = base_rate_i × days(m) × level_i(m) × season_seg(m) × ramp_i(h) × occasion_i(m)` — structure as v4.

1. Base rate, ramp (1.6 / 1.4 / 1.15, ×1.1 from h=2), channel-mix shrink: unchanged.
2. Web: **fully deflated** (`W_DEFL` 1.0): LY LFL ÷ LY site web × `SITE_WEB_IDX` (Oct 0.87 added).
3. Retail: flagship deflated, damped ^0.6 as v4; **stores with no sale in the last 5 days of the base month are
   treated as closed** (`STORE_CLOSED_DAYS`), their base-month units removed (NYR pop-up: ×0.976).
4. Other: **×0.90 per month** (was 0.93).
5. Seasonal profile: **K 16** (was 8), **damped rel^0.9**, **season-group anchoring 0.5** (`SEAS_ANCHOR`, 2-month
   window, clip 0.7–1.4).
6. Occasion index re-checked from a June base: 4th of July Jul 0.92 / Aug 0.85 / Sep 0.75 / Oct 0.69;
   Graduation 1.0 / 0.9 / 0.85 / 0.85; Homecoming 1.11 / 1.37 / 2.21 / 2.0; Halloween 1.0 / 1.05 / 1.3 / 1.9.

Forecast totals (768 items):

| Month | Units | per day |
|---|---|---|
| Jul-25 | 404.4k | 13.0k |
| Aug-25 | 390.4k | 12.6k |
| Sep-25 | 323.1k | 10.8k |
| Oct-25 | 321.6k | 10.4k |
| 4-month total | 1.440M | |

June-25 actual for these items 407k (13.6k/day). By group Jul / Aug / Sep / Oct: SUMMER 222 / 183 / 143 / 129k
(Jun 245k), WINTER 60 / 75 / 79 / 90k (Jun 54k), CORE 122 / 133 / 100 / 102k (Jun 108k). Halloween items 48k in
June → 46 / 42 / 40 / 55k; 4th of July 27k → 23 / 18 / 13 / 11k. v4's parameters at this origin would give
1.725M (Jul 457k): +20%, about a quarter of it from the pop-up store.

## 3. Backtests (research/bt.py) — origins Jan–May-25, per-item totals of complete months ≤ Jun-25

| Variant | mean WAPE | Jan / Feb / Mar / Apr / May (WAPE / bias) | ±20% | ±50% |
|---|---|---|---|---|
| v4 | 0.2998 | .302/+1% .343/−1% .338/+11% .261/−2% .255/+6% | 0.380 | 0.770 |
| W_DEFL 1.0 | 0.2959 | | 0.382 | 0.777 |
| SEAS_POW 0.8 / 0.6 | 0.2957 / 0.3032 | | 0.382 / 0.379 | |
| SEAS_ANCHOR 0.5 | 0.2939 | Mar .316, Apr −6% | 0.383 | 0.779 |
| SHRINK_K 16 | 0.2958 | | 0.379 | 0.771 |
| OTHER 0.9 | 0.2972 | | 0.387 | |
| POW 0.05 | 0.2960 | | 0.363 | 0.718 |
| W_DEFL 1 + anchor 0.5 + pow 0.8 + K16 + other 0.9 | 0.2841 | .296/−8% .349/−1% .290/+2% .245/−8% .240/0% | 0.398 | 0.790 |
| **v5 chosen (as above, SEAS_POW 0.9)** | **0.2833** | .293/−8% .343/−1% .292/+2% .246/−8% .243/+1% | 0.394 | 0.796 |
| 0.8-combo + POW 0.03 | 0.2803 | | 0.387 | 0.770 |
| 0.8-combo + RAMP 1.8 / 1.5 ; W14 0.3 | 0.2849 ; 0.2859 | | | |

- Each change helps alone and they add up (−0.016 WAPE, ±20% +1.4 pts). Target months are Feb–Jun only, so the
  seasonal fixes are tested on the spring decline; for the autumn rebound they rest on the same logic (higher
  trough → smaller rebound), not on direct evidence. SEAS_POW 0.9 vs 0.8 is a tie; 0.9 is the milder.
- POW again buys WAPE with hit shares (±50% −2.6 pts): not used. Pop-up rule and occasion / site-path updates
  cannot be backtested (no pop-up before, no Jul–Oct targets).
- Backtest bias of v5 averages −3% (Jan and Apr origins −8%): mild downside lean.

## 4. Known weaknesses

- **Level falls ~20% from June to Sep–Oct** (web drift ×0.92/month compounding, site Sep 0.93 / Oct 0.87,
  retail back-to-school ×0.74, other ×0.9/month). If launches keep the panel fresher or the Labor-Day (Sep 1)
  and mid-October sales are stronger than 2024's, Sep–Oct is under. The backtest lean is already −3%.
- July 2025: a July-4 sale is likely; `SITE_WEB_IDX` Jul = Jun (which had its own 3-day sale). Unknown size.
- Winter rebound: damping + anchoring cut WINTER Oct relative from 3.2 to ~2.2. If 2025's autumn winter demand
  matches 2024's absolute level, winter types are under in Sep–Oct (90k in Oct forecast).
- Halloween Oct ×1.9 on 68 items / 48k June units (the biggest occasion bet, ~+25k units in Oct); Homecoming
  Sep ×2.2 (17 items, small). Valentine's / NYE dresses' Sep jump (homecoming dresses) is not modelled.
- Retail Jul–Aug ×1.04 assumes June's summer level holds; 2024 stores rose Jun→Aug (+15–35%), damped away.
- Small items still under-forecast (bottom quintile ×1.1–2.8); POW fixes WAPE but hurts hit shares.
- Shorts / Strapless over, Skirt / Short Jeans / Swimwear under at all origins; K16 + damping only partly fixes.
- Future planners: check July's site web vs June (8.83k/day) and whether the pop-up rule fired correctly
  (any store with a gap of 5+ days at month end is dropped); re-check the Halloween index after Sep-25.

---
# History: v4 notes (snapshot 2025-05-31), kept for reference

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
- **v5 (2025-06-30 snapshot)**:
  1. *Web fully deflated* (`W_DEFL` 0.75 → 1.0). Evidence: the 25% LY-LFL part carries the late-July-24 regime
     jump into Aug–Oct (Aug factor 1.25 vs deflated 0.87); backtest 0.2998 → 0.2959 alone.
  2. *Pop-up / closed stores* (`STORE_CLOSED_DAYS` 5): NYR sold only Jun 19–22; v4's new-store run-rate would
     have added ~20% to retail for all four months.
  3. *Seasonal amplitude*: `SEAS_ANCHOR` 0 → 0.5, `SEAS_POW` 0.9 (new), `SHRINK_K` 8 → 16. Evidence: WINTER
     under / SUMMER over in May and June at every origin (WINTER ×1.09–1.61, Shorts 0.67–0.81); each change helps
     in the backtest, combined 0.2998 → 0.2833 with ±20% 0.380 → 0.394.
  4. *Other ×0.93 → ×0.90/month*: drop −19%/day in June (Pacsun / Nordstrom / Flip down); backtest −0.003.
  5. *Site web path*: Oct 0.87 added (2024 Oct/Aug 0.84); June confirmed at May's level (×0.97).
  6. *Occasion index re-checked from a June-24 base*: 4th of July fades faster (Oct 0.69), Graduation lower,
     Homecoming Sep 2.21, Halloween Aug 1.05 / Sep 1.3 / Oct 1.9 (was 1.15 / 1.6).
  Unchanged after testing: base blend (W14 0.3 worse), ramp (1.8 / 1.5 worse), POW off (hit shares).
