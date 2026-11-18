"""Builds notes.md from prev/notes.md: new v4 sections on top, v3 sections kept as history, changelog appended."""
prev = open("prev/notes.md", encoding="utf-8").read()
body, chlog = prev.split("## Changelog\n")
old_sections = body.split("\n", 3)[3]  # drop title + run line

new = """# Forecast notes: Jun–Sep 2025 per style_color (snapshot 2025-05-31)

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

"""
for k in "01234":
    old_sections = old_sections.replace(f"## {k}.", f"### v3-{k}.")
new += old_sections + "## Changelog\n" + chlog.rstrip("\n") + """
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
"""
open("notes.md", "w", encoding="utf-8").write(new)
print(len(new.splitlines()), "lines")
