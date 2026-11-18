# Forecast notes: Sep–Dec 2025 per style_color (snapshot 2025-08-31)

`python rules.py` reads `data/` and writes `predictions.csv` (867 items × 4 months, ~3 s).
Research scripts in `research/`: `r1_site.py` (site / channel / store levels, 2024 Jul–Dec weekly, Aug-25 daily),
`r2_review.py` (scoreboard Aug-25 by group / type / age / retail share / size / tag), `r3_occ.py` (tag profiles from
an Aug-24 base, TY vs LY Jul→Aug tag and group moves, BOCA opening), `r4_factors.py` (factor dump at this origin),
`r5_anc.py` (anchoring window sensitivity), `r6_fc.py` (forecast by group under variants), `bt.py` (backtest
harness, origins Feb–Jul-25 with the scoreboard item lists, per-item totals of complete months ≤ Aug-25),
`rules_v6.py` (copy of last month's rules), `write_notes.py` (assembles this file).

## 0. Review of v6 (origin 2025-07 → Aug-25, h=1) and older origins

- **August was again a level miss, and it was the site plan.** v6 Aug 615.0k vs actual 697.8k (816 items):
  **bias −11.9%, WAPE 0.227** (the best h=1 WAPE so far). Older origins for August: v5 h=2 −40%, v4 h=3 −29%,
  v3 h=4 −35% (the summer events were not in their plans). Decomposition of v6's miss: edikted.com web/day
  15.64k vs plan 13.5k (**+16%**; the back-to-school event ran to Aug 20 at 14–20k/day, the last 10 days were
  11.5–12.2k/day weeks, i.e. 1.6–1.75× the same weeks of 2024, not 2024's fall to 0.65×); other 3.13k vs 2.9k
  (+8%); retail 4.92k vs ~4.26k (+15%, mostly the **new BOCA store from Aug 7**, 1.2k/day in its first week,
  ~0.5k/day by the end). Weighted by channel share these explain ≈ −12%: conditional on the site, v6 had no level
  bias (oracle-site backtest at the Jul origin: −2%).
- **Seasonal amplitude, rebound side: WINTER over, SUMMER under, at all three origins.** Aug relative to the total,
  origins Jul / Jun / May: WINTER 0.85 / 0.87 / 0.73, SUMMER 1.10 / 1.09 / 1.12, CORE ~1.0. Types: Swimwear
  0.71–0.75, Shirt 0.76, Dress 0.90, Sweater / Sweatpants 0.91; Skort 1.19, Tank Top 1.15, Shorts 1.11 (Bodysuit
  1.53, n=11). Direct check: group LFL moves relative to the panel, TY vs LY: WINTER Jun→Jul 1.15 vs 1.36, Jul→Aug
  1.13 vs 1.34; SUMMER 0.92 vs 0.86, 0.90 vs 0.835 — **2025's monthly group moves are about LY's^0.45–0.57**.
  In spring (v5 review) winter fell less than LY; now it rises less: a milder seasonal cycle on a higher winter
  base (winter share of panel units Aug-25 18% vs Aug-24 9%), not a lagging rebound that will catch up.
- v6's 2-month season-group anchoring now points the wrong way: Jun→Aug winter rose less than LY, so it reads a
  gap (LY/TY 1.40) and adds WINTER ×1.18 for Sep–Dec. Over the window since January the gap is the other way
  (0.90: TY winter is already relatively higher than LY at the same point).
- Young items: Jul launches (n=22) 1.32 relative, other ages 0.88–1.19; RAMP1 1.6 / RAMP0 1.8 tested worse
  (0.2784 vs 0.2754): not changed. Retail-share buckets 0.97–1.03, size quintiles 0.90–1.005 (small items no
  longer under): no signal.
- Tags (rel. at h=1 / h=2): Halloween 0.95 / 1.20 (v6's Aug index 1.18 a bit high), NYE 0.82 / 0.91 (over),
  Homecoming 1.04 / 1.05, 4th of July 1.08 / 1.04, Graduation 1.04 / 1.07, Valentine's 1.01 / 1.02.

## 1. Key findings (new month)

**Site Aug** (per day): units 23.7k, web 15.64k (disc 0.52), retail 4.92k, other 3.13k (drop 1.83k). Weekly web
17.7 / 19.5 / 18.1 / 12.2 / 11.5k; **Aug 31 (Sunday before Labor Day) 20.5k at discount 0.58**: the Labor-Day
sale is under way as in 2024 (Sep 1–7-24 14.0k/day, 2× the weeks after). YoY web/day: Jun 2.0, Jul 1.80, Aug 1.65
(first fully regime-matched month), late-Aug weeks 1.58–1.76. **2024 from August:** web/day Sep 0.91, Oct 0.84,
Nov 2.06, Dec 1.87 × Aug (Black-Friday week 54k/day, Cyber-Monday week 30k/day; BF Nov 29-24 / Nov 28-25, Cyber
Monday in December both years).
**Other:** Flip ended (0 since July); Pacsun 27.0k (Aug-24 23.0k), TikTok 40.1k (43.7k), Nordstrom 12.0k after the
July anniversary 24.7k, Macy's 8.2k, Tillys 4.2k. 2024 ex-Flip other/day from Aug: Sep 1.08, Oct 0.93, Nov 1.48,
**Dec 2.28** (Pacsun 69.6k, Macy's 22.4k, Nordstrom 28.6k in Dec-24).
**Retail:** 8 stores open at month end (BOCA new, NYR open all month at 739/day). Same-store 2024 Aug→Sep/Oct/Nov/Dec:
SoHo 0.62 / 0.70 / 0.80 / 1.01, LAGROVE 0.65 / 0.61 / 0.71 / 0.67, MNMOA 0.41 / 0.40 / 0.52 / 0.46. Rule factor
(flagship deflated ^0.6; new-store run-rate ×0.979 because BOCA's opening spike is replaced by its run-rate):
0.69 / 0.69 / 0.70 / 0.79.
**Panel drift** (LY LFL ÷ LY site web, cumulative from Aug-24): 0.899 / 0.809 / 0.712 / 0.695 — Dec holds Nov.
**Season groups, LY profile from Aug-24** (rel. to panel, Sep–Dec): WINTER 1.40 / 1.77 / 3.15 / 3.27, SUMMER
0.99 / 0.89 / 0.58 / 0.54, CORE 0.94 / 0.98 / 0.87 / 0.87. LY winter share Nov-24 47%, Dec-24 49%.
**Occasion tags, Aug-24 base vs same-segment untagged items (Sep / Oct / Nov / Dec):** Halloween (n=30) 1.37 /
2.59 / 0.83 / 0.84; Homecoming (12) 2.94 / 2.00 / 1.72 / 1.59; NYE without Homecoming (5) 1.39 / 2.02 / 2.79 /
4.57; 4th of July (8) 0.72 / 0.62 / 0.79 / 0.81; Graduation (11) 0.92 / 0.78 / 0.79 / 0.83; Valentine's (20)
1.38 / 1.02 / 1.33 / 1.27 (no calendar reason, not used). TY Jul→Aug tag moves vs LY: Halloween 1.00 vs 1.05,
Homecoming 1.41 vs 1.46, NYE 0.94 vs 1.45, 4th of July 0.88 vs 0.88.

## 2. Rules (rules.py, v7)

`pred(i,m) = base_rate_i × days(m) × level_i(m) × season_seg(m) × ramp_i(h) × occasion_i(m)` — structure as v6.

1. **Site plan** (`SITE_WEB_PD`, web/day): Aug actual 15.64k; **Sep 14.0k** (2024 Sep/Aug 0.91; late-Aug 11.85k ×
   2024's Sep / late-Aug 1.21 = 14.3k), **Oct 12.6k** (×0.9), **Nov 28.5k** (Black Friday, 2.26× Oct vs 2.45× LY),
   **Dec 25.0k** (0.88× Nov vs 0.905 LY). Implied YoY 1.63 / 1.58 / 1.46 / 1.41 (Aug 1.65, decaying as 2024's
   growth annualises). Web factor 0.81 / 0.67 / 1.34 / 1.16.
2. **Other plan** (`OTHER_PD`): Aug 3.13k actual; Sep 3.1k, Oct 2.75k, Nov 4.2k, Dec 5.5k (LY ex-Flip shape shaded
   for 2024's onboarding growth) → factor 0.90 / 0.73 / 0.99 / 1.27.
3. **Seasonal amplitude:** `SEAS_POW` 0.9 → **0.8**; season-group anchoring keeps a=0.5 but the gap is the
   geometric mean of the 2-month gap and the **gap since January** (`ANCHOR_LONG`). Anchors here: WINTER 1.06 (v6
   1.18), SUMMER 0.90 (0.93), CORE 1.01 (1.04). WINTER rel. Nov 3.15 → 3.15^0.8 × 1.06 = 2.8 (v6 3.3).
4. `DRIFT_ADJ` 1.0 → **1.01** (panel holds up slightly better than LY drift; see backtests).
5. Occasion index re-based from August, December added: Halloween Aug 1.18 → 1.1, from Aug ×1.27 / 2.05 / 0.91
   / 0.86; Homecoming ×2.0 / 1.62 / 1.42 / 1.35; NYE (not on Homecoming items) 1.1 / 1.25 / 1.45 / 1.8; 4th of
   July and Graduation Dec index 0.9 / 0.75.
6. Unchanged: base blend, ramp, K, channel shrink, retail rule (the new-store run-rate handled BOCA), pop-up rule.

Forecast totals (867 items; Aug actual for these items 723.4k, 23.3k/day):

| Month | Units | per day |
|---|---|---|
| Sep-25 | 606.8k | 20.2k |
| Oct-25 | 597.1k | 19.3k |
| Nov-25 | 899.1k | 30.0k |
| Dec-25 | 882.9k | 28.5k |
| 4-month total | 2.986M | |

By group Sep / Oct / Nov / Dec: SUMMER 279 / 257 / 239 / 225k (Aug 364k), WINTER 153 / 167 / 433 / 440k (Aug
131k; share 25 / 28 / 48 / 50%), CORE 175 / 174 / 227 / 218k (Aug 228k). Halloween items (72) 87k in Aug → 85 /
124 / 63 / 58k; NYE (38) 27k → 30 / 37 / 44 / 54k; Homecoming (23) 8k → 18 / 12 / 9 / 8k. v6's settings with the
same site plan give 3.20M (Nov 987k, WINTER Nov 532k = 54% share).

## 3. Backtests (research/bt.py) — origins Feb–Jul-25, per-item totals of complete months ≤ Aug-25

Site plans hold actual past levels (oracle web / other level): these test drift, segments, anchoring, ramp —
not the site assumptions. Last column = actual/pred of WINTER relative to the total, per origin.

| Variant | mean WAPE | Feb / Mar / Apr / May / Jun / Jul (WAPE / bias %) | ±20% | ±50% | WINTER rel by origin |
|---|---|---|---|---|---|
| v6 settings | 0.2754 | .325/−16 .294/−7 .288/−3 .251/−6 .287/−6 .208/−2 | 0.399 | 0.777 | .90 1.13 1.09 1.02 .93 .87 |
| anchoring off | 0.2813 | | 0.389 | 0.764 | .90 1.18 1.14 .89 .79 .90 |
| momentum (a −0.5) | 0.2948 | | 0.370 | 0.740 | |
| ANCHOR_WIN 1 / 3 / 6 | 0.2793 / 0.2758 / 0.2829 | | | | w3: … .94 .99 |
| level-neutral anchoring | 0.2773 | | 0.388 | 0.766 | |
| ANCHOR_LONG (pow .9) | 0.2762 | Jul .208/−6 | 0.400 | 0.785 | … 1.00 .92 .91 |
| SEAS_POW .8 / .7 / .6 | 0.2751 / 0.2765 / 0.2796 | | | | .7: … 1.06 1.02 .92 |
| POW .8 + LONG | 0.2759 | | 0.398 | 0.784 | .84 1.04 1.05 1.01 .96 .93 |
| POW .7 + LONG | 0.2774 | | 0.400 | 0.781 | .79 .96 1.01 1.03 1.00 .96 |
| RAMP1 1.6 (± RAMP0 1.8) ; W14 0.3 ; POW 0.03 | 0.2784 ; 0.2770 ; 0.2797 | | | | |
| v7 with DRIFT_ADJ 1.0 / 1.02 | 0.2774 / 0.2775 | bias avg −9% / −6% | 0.399 / 0.394 | 0.786 / 0.774 | |
| **v7 (POW .8, LONG, DRIFT 1.01, new occasion)** | **0.2762** | .330/−14 .292/−5 .288/−2 .253/−7 .289/−8 .205/−6 | 0.398 | 0.778 | .85 1.04 1.04 1.01 .95 .92 |

- All variants are within ±0.002 of each other except anchoring off / momentum / long windows alone (worse). The
  choice rests on the rebound-phase evidence (sections 0–1) rather than these decimals: the backtest months are
  mostly the spring decline, and Sep–Dec cannot be tested.
- v7 keeps v6's WAPE (0.2762 vs 0.2754) and hit shares, and balances WINTER at recent origins (Jun / Jul 0.95 /
  0.92 vs 0.93 / 0.87). Oracle-site bias averages −7% (v6 −7%); DRIFT 1.02 removes 3 more points at no WAPE cost
  but −0.5 point of ±50%; 1.01 is the middle.
- The Feb origin is hurt by lower SEAS_POW (March needs full amplitude): amplitude may differ by season phase.

## 4. Known weaknesses

- **The site plan is still the forecast.** Nov + Dec are 60% of the 4-month total and rest on Black Friday
  (28.5k/day web) and December holding 0.88× Nov. A BF like 2024's (2.45× Oct) is +8% on Nov; YoY falling to
  1.3 by Nov–Dec is −10%. Sep 14.0k assumes the Labor-Day week (started Aug 31 at 20.5k) runs like 2024's.
- **Winter Nov–Dec** is the biggest segment bet (433 / 440k, 48–50% share). The milder 2025 cycle argues for
  less (Jul/Aug moves ~LY^0.5 would give ~2.0× instead of 2.8× by Nov); a return to Nov-24's mix argues for
  2.6–3.0×. Summer anchoring 0.90 may be too low if summer keeps declining slower than LY (under in August).
- Oracle-site backtest bias ≈ −7% (panel decays less than LY drift; WAPE-optimal but biased low).
- Other Dec ×1.27 (5.5k/day) relies on holiday wholesale orders (Pacsun / Macy's / Nordstrom) repeating; these
  are lumpy. Retail Dec 0.79 has no explicit holiday store lift (2024 SoHo Dec 1.01× Aug; mall stores lower);
  BOCA (Florida) may run stronger in winter than the summer-tourist stores.
- Halloween Oct ×2.05 (72 items, 87k Aug units → 124k in Oct) and NYE Dec ×1.8 (n=5 last year) are the tag bets.
- Future planners: check September web/day vs 14.0k (Labor-Day week size), WINTER rel. at h=1 (v7 assumes Sep
  1.40^0.8 × 1.06 ≈ 1.39 relative to the panel), Halloween in Sep (×1.27 assumed); add each month's actuals to
  `SITE_WEB_PD` / `OTHER_PD`. Consider making seasonal amplitude phase-dependent (decline vs rebound).

---
