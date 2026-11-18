# Forecast notes: Aug–Nov 2025 per style_color (snapshot 2025-07-31)

`python rules.py` reads `data/` and writes `predictions.csv` (816 items × 4 months, ~2 s).
Research scripts in `research/`: `r1_site.py` (site / channel / store levels, 2024 Jun–Dec weekly),
`r2_review.py` (scoreboard July by group / age / size / retail share / type / tag), `r3_factors.py` (channel
history, factors at this origin, panel drift by month), `r4_occasion.py` / `r5_overlap.py` (tag profiles from a
Jul-24 base, season-group magnitudes), `r6_fc.py` (forecast breakdown), `bt.py` (backtest harness, origins
Jan–Jun-25, per-item totals of complete months ≤ Jul-25), `bt2.py` / `bt3.py` (variants), `bt_diag.py`
(assumed vs actual channel moves per origin), `write_notes.py` (assembles this file).

## 0. Review of v5 (origin 2025-06 → Jul-25, h=1) and older origins

- **July was a level miss, not a segment miss.** v5 July 404k vs actual 622k (**bias −35%**, WAPE 0.383); v4
  at h=2 −26%, v3 at h=3 −28%. Scaled to the right total, v5's July WAPE is 0.241 (the usual h=1 skill).
  Season groups rel. to total 0.96 / 1.01 / 1.02 (CORE / SUMMER / WINTER), age buckets 0.89–1.03, size
  quintiles 0.86–1.13: no segment signal.
- **Cause: the site.** edikted.com web/day 8.83k (Jun) → 13.9k (Jul, ×1.57): 4th-of-July sale (Jul 3–8, peak
  25.9k on Jul 8, discount 0.61) + a **back-to-school event from Jul 20** (16–18.5k/day for 12 days, no fade,
  discount 0.57). 2024 had the same event on the same dates (Jul 21 – Aug 17, discount 0.56–0.57), which v2–v5
  treated as a one-off "discount regime" and deflated away. Retail 3.14k → 4.30k/day (NYR, removed by v5's
  pop-up rule, **reopened Jul 2 and sold 937/day**; same-store +11%); other 1.81k → 3.10k (Nordstrom
  Anniversary 7.4k → 24.7k units, TikTok +75%, Pacsun +51%).
- Earlier origins at h=2–4 for July were also under (−26 to −28%): the summer site lift was never in the plan.
- Panel drift vs site held: Jun→Jul LFL web ÷ site web 0.95 (2025 Jan–Jul 0.91–0.96, 2024 0.86–0.94).
- Retail-heavy items (>40% retail) rel 0.88, web-only 1.07 (web surged more than retail). Types: Swimwear
  0.54–0.58 (season over), Bodysuit / Crop Top / Sweater / Dress 1.16–1.27 (small n, one month: not acted on).
  Tags rel: Halloween **1.15** (rising early), others 0.97–1.04.

## 1. Key findings (new month)

**Site** (per day): Jul units 21.3k, web 13.9k, retail 4.30k, other 3.10k (drop 1.88k). Last week of July web
17.3k/day at discount 0.57. **2024 path after the same event:** web/day Aug 9.47k (0.88 × late-Jul-24),
Sep 8.61k (Labor-Day week 14k, then ~6.6k), Oct 7.96k (mid-Oct sale weeks 10.5–11.2k), **Nov 19.5k (×2.45 Oct;
Black-Friday week 54k/day)**, Dec 17.7k. YoY web: Jun 2.0, early Jul 1.95, late Jul 1.59 (2024's step is
annualising). **Other 2024:** TikTok / Pacsun rose Jul → Sep, marketplaces ×1.5–2.7 in Nov.
**Retail 2024 same-store from July:** Aug ×1.0–1.3, Sep ×0.5–0.7 (back to school), Oct ×0.5–0.8, Nov ×0.7–0.9.
**Panel drift** (LY LFL ÷ LY site, per day, cumulative from Jul-24): Aug 0.906, Sep 0.814, Oct 0.732,
Nov 0.645 — the Oct→Nov-24 link (0.88) is normal, i.e. panel items took their share of Black Friday.
**Season groups:** Jul-24 → Nov-24 WINTER rel ×3.66 (direct LFL), SUMMER ×0.54; Nov-24 → Jul-25 WINTER 0.335
(a return to Nov-24's mix would be ×2.99). WINTER's share of panel units has been 2–2.5× last year's all
spring/summer (Jul 15.4% vs 6.2%): 2025 carries more winter weight, not only a milder trough.
**Occasion tags, Jul-24 base vs untagged same-segment items (Aug / Sep / Oct / Nov):** Halloween (n=29) 1.31 /
1.83 / 3.73 / 1.33 (Halloween-only n=24 the same); Homecoming (11 dresses) 1.38 / 4.52 / 2.88 / 2.25; New
Year's Eve without Homecoming (n=5) 1.03 / 1.68 / 3.18 / 2.59; 4th of July 0.93 / 0.75 / 0.61 / 0.84;
Graduation 0.80 / 0.73 / 0.69 / 0.66; Valentine's-only (12, corsets / tanks) 1.14 / 1.68 / 1.60 / 1.31 — no
calendar reason, not used. Jun→Jul tag moves TY vs LY: Halloween 1.20 vs 1.05, 4th of July 0.87 vs 0.84.

## 2. Rules (rules.py, v6)

`pred(i,m) = base_rate_i × days(m) × level_i(m) × season_seg(m) × ramp_i(h) × occasion_i(m)` — structure as v5.

1. Base rate, ramp, channel-mix shrink, seasonal profile (K 16, rel^0.9, anchoring 0.5 over 2 months), retail
   (flagship deflated ^0.6, pop-up rule): unchanged.
2. **Web = LY panel drift × site plan.** `SITE_WEB_PD` holds absolute edikted.com units/day by month (actuals up
   to Jul-25, used by the backtest; assumptions after): **Aug 13.5k, Sep 12.3k, Oct 11.2k, Nov 25.5k** (Jul
   13.9k). Web factor from July: 0.88 / 0.72 / 0.59 / 1.18.
3. **Other = same drift × other-channel plan** `OTHER_PD`: Aug 2.9k, Sep 2.7k, Oct 2.4k, Nov 3.6k/day
   (Jul 3.10k incl. Nordstrom Anniversary) → 0.85 / 0.71 / 0.57 / 0.75 (×0.9^h would be 0.90 … 0.66).
4. Occasion index from the July base: Halloween Aug 1.18 / Sep 1.45 / Oct 2.3 / Nov 1.15 (v5: 1.05 / 1.3 /
   1.9 / 1.0); Homecoming Nov 1.43; **New Year's Eve (new, not on Homecoming items)** 1.0 / 1.15 / 1.35 / 1.5;
   4th of July Nov 0.75; Graduation Nov 0.8. Retail factor (computed) 0.99 / 0.70 / 0.705 / 0.709.

Forecast totals (816 items; July actual for these items 650.8k, 21.0k/day):

| Month | Units | per day |
|---|---|---|
| Aug-25 | 615.0k | 19.8k |
| Sep-25 | 524.9k | 17.5k |
| Oct-25 | 522.1k | 16.8k |
| Nov-25 | 771.8k | 25.7k |
| 4-month total | 2.434M | |

By group Aug / Sep / Oct / Nov: SUMMER 294 / 237 / 217 / 194k (Jul 365k), WINTER 126 / 137 / 152 / 386k (Jul
100k), CORE 195 / 151 / 153 / 192k (Jul 186k). Halloween items 82k in July → 80 / 79 / 116 / 63k; NYE 25k →
27 / 30 / 39 / 58k; 4th of July 37k → 28 / 19 / 16 / 22k. v5's code at this origin gives 2.19M with Nov only
475k (no Black Friday in its site index).

## 3. Backtests (research/bt.py) — origins Jan–Jun-25, per-item totals of complete months ≤ Jul-25

The site plans hold actual site levels for past months, so the web / other LEVEL is oracle in these runs: they
test drift, segments, ramp, base blend — not the site assumptions (those are judgment, section 1).

| Variant | mean WAPE | Jan / Feb / Mar / Apr / May / Jun (WAPE / bias) | ±20% | ±50% |
|---|---|---|---|---|
| **v6** | **0.2728** | .290/−11% .325/−16% .294/−7% .265/−1% .224/−5% .241/−5% | 0.404 | 0.796 |
| other ×0.9^h (v5 rule) | 0.2745 | Jun .250/−12% | 0.402 | 0.805 |
| DRIFT_ADJ 1.02 / 1.03 / 1.05 | 0.2750 / 0.2771 / 0.2835 | bias → −4 … +2% | 0.400 / 0.397 / 0.395 | 0.782 / 0.775 / 0.760 |
| SEAS_POW 0.8 / 1.0 | 0.2735 / 0.2742 | | 0.401 / 0.398 | |
| SEAS_ANCHOR 0 / 1.0 | 0.2778 / 0.2743 | | | |
| ANCHOR_WIN 4 / 6 | 0.2777 / 0.2778 | | | |
| SHRINK_K 32 ; CH_K 10 | 0.2730 ; 0.2729 | | | |
| POW 0.03 | 0.2723 | | 0.398 | 0.773 |
| W14 0.5 ; WLONG 0.1 / 0.3 | 0.2718 ; 0.2730 / 0.2748 | | | |
| RET_POW 0.8 / 0.4 | 0.2744 / 0.2723 | | | |
| RAMP 1.4 / 1.25 / 1.1 | 0.2724 | | 0.406 | 0.802 |

- Even with oracle site levels the panel's web ran ~6% above the drift-based factor at every origin (actual /
  assumed 1.00–1.12; 2025 drift 0.934/month vs 2024's 0.906). Correcting it (`DRIFT_ADJ`) removes the bias but
  **worsens WAPE and hit shares** (item-level skew: decliners outnumber holders) — kept at 1.0 as a switch.
- Everything else is within ±0.001–0.002 of v6 (noise): unchanged. POW again costs ±50% hits.

## 4. Known weaknesses

- **The site plan is the forecast.** Every 10% on web ≈ 6–7% of units. Biggest bets: (a) Aug 13.5k assumes the
  back-to-school event runs to mid-August as in 2024; if it ended Jul 31, Aug is ~10–15% over. (b) **Nov 25.5k
  = Black Friday ×2.28 on Oct** — Nov is 32% of the 4-month total; a BF like 2024's (×2.45) is +7% on Nov, a
  weaker one (×1.8) −20%. (c) Sep / Oct rest on 2024's post-Labor-Day fade, measured in a growth year.
- Drift: the backtest says the panel holds up ~3%/month better than LY drift; v6 deliberately keeps LY drift
  (WAPE-optimal). Expect aggregate bias around −5% even if the site plan is right.
- WINTER Nov ×3.8 rel. to panel (50% of Nov units vs 47% in Nov-24); the symmetric "return to Nov-24 mix"
  estimate is ×3.0. If 2025's extra winter weight does not persist, winter types are over in Nov.
- Halloween Oct ×2.3 on 70 items (82k July units) is the biggest tag bet; NYE rests on n=5 last year.
- Retail Nov ×0.71 has no explicit Black-Friday store lift beyond the deflated flagship path; NYR is assumed
  permanent (it closed once in June).
- Future planners: check August web/day against 13.5k (did the event run past mid-August?), the Labor-Day week,
  and re-check Halloween after Sep; add each new month's actuals to `SITE_WEB_PD` / `OTHER_PD`.
