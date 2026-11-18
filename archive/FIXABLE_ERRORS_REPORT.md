# Clearing the fixable errors: answers and results (18 Nov 2026)

This covers decisions D252–D260 in log.md. The scripts are `eda/180*.py`, `eda/175k`, `eda/175l` and the re-fits of `eda/179x2` / `eda/179b`.

Scores are WAPE ÷ R3 on the nine recent backtest origins unless stated. The "system" is the five-model final ensemble, p2c32a: rules, the D198 LightGBM, the NN, conformal TabPFN and TabPFN dyn.

## Result

| System | Recent | All 13 | Black Friday origins | Bias recent | Buying loss |
|---|---|---|---|---|---|
| p2c32a (final system before) | 0.3977 | 0.4542 | 0.5122 | −4.6% | 10.74% |
| p2c32a_geo (D248) | 0.3902 | 0.4482 | 0.5079 | −4.6% | 10.78% |
| E:ec_ns: new LightGBM in the system (D253) | 0.3961 | 0.4522 | 0.5094 | −5.0% | **10.71%** |
| E:ec_ns_geo: + corrections re-fitted (D257) | 0.3888 | 0.4463 | 0.5049 | −5.0% | 10.74% |
| **E:ec_ns_x_geo: + falling-item signals (D259)** | **0.3871** | **0.4449** | | −5.0% | 10.72% |
| u65ec: newsvendor ladder with the new LightGBM (D260) | | | | | **10.36%** (u65 10.38%) |

The new system improves accuracy by 2.7% against p2c32a and 0.8% against p2c32a_geo (bootstrap −0.0031 to −0.0011). Buying loss is level: it is driven by bias and the peak months, which these corrections deliberately leave unchanged.

## Your questions

**Do we have days since launch as a feature, and a mod-365 version?**
- The feature table had `days_live` and `age_m`, but **the lean set (D134/D135) dropped them, so the tree member had no age feature at all**. Age entered only through which lags exist.
- I added the AGE block:
  - true days live;
  - days live mod 365 (sin and cos);
  - launch day of year;
  - distance from the target window to the launch anniversary.
- Alone it is level: 0.4313 against 0.4326. The launch-curve block below carries the age information better.

**More holidays and events.**
- 16 event families existed. I added 15 more: Super Bowl, St Patrick's, Father's Day, Pride, October Prime days, Singles Day, the Christmas shipping rush, after-Christmas sales, New Year sales, MLK and Presidents' Day weekends, tax-refund season, back-to-college, New York fashion weeks, cyber week. I also added the target window's weekend days, Fridays and day count.
- **This helps:** 0.4282 against 0.4326 for the model alone, a significant gain.

**Which item types are rising or falling.**
- The TREND block covers the item's product type, macro group, silhouette, print, length and fabric:
  - like-for-like 3-month momentum;
  - year-on-year share change;
  - launch rate;
  - item trend minus its type's.
- **No gain** (0.4349).
- Falling-fast items (14.5% of units) stay at WAPE ÷ R3 0.56. Their misses split into items that rebounded (60% of their error) and items that kept falling (40%). Neither the type trend nor any other signal predicts which happens.
- The best signals are promotion sensitivity, a spike in the last 28 days, and time since the year's peak. Added to the residual correction, they lift the whole system (0.3888 → 0.3871), but not the falling-fast group itself.

**Launch curve info.**
- The CURVE block, built from earlier launches only:
  - the typical next-4-month ratio at the item's age (by type);
  - the item's ramp against peers of the same age;
  - first-30-day units;
  - its best month so far.
- **This helps:** 0.4282 alone. Together with the new events it gives the chosen member.

**Item peaks and troughs: how big and when.**
- The PEAK block:
  - best 7-day run over life and over the last year, as a ratio to level, with days since each;
  - peak-anniversary distance;
  - worst 28-day run and days since;
  - number of spikes;
  - best recent week.
- Level as a model feature (0.4332). As a correction signal it helps (D259).

**A separate model for low- and mid-volume items.**
- Trained only on rows below each origin's 60th level percentile and used for those items. Level with the shared model (in the system 0.3963–0.3982).
- The shared model with the new features already moves the small and mid volume fifths by 0.017–0.021, and the bias of the 2nd fifth from −27% to −23%.

**Does the model memorise specific items instead of generalising?**
**No.**
- **Style held out:** in a 5-fold test, each item was predicted by a model that never saw any row of its style. The score moved from 0.4326 to 0.4371 (+1%, not significant).
- **Explicit item ID:** giving the model the item's identity as a feature makes it worse (0.4556), so memory would hurt.
- **Static descriptors:** removing them (LLM scores, attributes, product type) is level on recent origins and better on all 13 (0.487 against 0.503). The chosen member drops them.

**Other sanity and leakage checks.**
- **Truncation test:** the feature tables were rebuilt at 2025-06 and 2025-10 from data cut at the origin. Every feature matches exactly, for the old table and for the new blocks.
- **Training windows:** every member trains only on origins whose outcome was known (checked in the code; asserted in the new harness).
- **Shuffled target:** training on a shuffled target collapses the model (0.75), so nothing leaks through the features.
- **Survivorship (limitation found):** all 1,500 catalogue items were still selling after July 2025, so the backtest never sees an item die. Every backtest number is on survivors. Fixing this needs Edikted's full historical catalogue.
- **New items:** items whose style the model has never seen score 0.505 against 0.411 for seen styles. This is the cold-start gap.

**A Black Friday–specific model.**
- Three versions:
  - trained only on windows containing November;
  - Black Friday rows weighted 3×;
  - plus last year's November uplift of the item and its type.
- None beats the shared model with the new events in the system (0.5088–0.5123 against 0.5094 on Black Friday origins), and all are worse on all 13 origins.
- The Black Friday error is breakouts that every model misses (D244/D247), not a missing November feature.

**What other known-future data could we use?**
- **Already used:**
  - 31 event windows;
  - regional temperature normals (realised temperature as an oracle adds nothing, D157);
  - the target window's calendar and weekday mix;
  - launch age and anniversaries;
  - Google Trends (D181);
  - the site campaign calendar (rejected as a feature, D173; handled as an adjustment, D031).
- **Launch plan:** tested as launches of the same type in the target window. It hurts (0.4632, bias −14%). Cannibalisation probably needs similarity-level counts (same look), not type counts.
- **Would need Edikted's internal plans:**
  - per-item markdown and price schedules: the deepest known-future driver; we only see the site calendar;
  - receipts and inventory plans: stock-outs cap sales;
  - marketing, email, SMS and creator/TikTok schedules;
  - retail and wholesale drop dates;
  - regional school calendars.

**Did the cascade multiply between steps or add?**
- **It multiplies:** R3 × seasonality, the trees predict log(total ÷ 4·level), and the corrections and the buying step are multipliers.
- **The exception is the model stack, an arithmetic mean of the five members in units.** The geometric mean under-forecast (D190).
- The cascade sheet shows each step as an additive log contribution, which is the same multiplicative chain written in logs. Each member's contribution is measured against the seasonality layer, not stacked on the previous member.

**Using the days of the week.** What was tried:
- **Already in buying:** the simulation spreads each month over its days with a weekday × calendar profile before placing orders.
- **Weekday mix of the target window (new):** at most ±2% at the 4-month level. Level as a feature.
- **Each item's weekend share against the site's (new):** level as a feature.
- **Day-level promotion response (new):** the item's lift on promo days and its lift over last year's Black Friday weekend.
  - As model features they hurt (bias −12%).
  - As correction signals, promotion sensitivity is the strongest predictor of falling items' misses (D259).
- **Monthly split (D255):** blending the LightGBM monthly profile with last year's same-type pattern cuts pure month-split error from 0.219 to 0.202. It does not lower buying loss (10.75% against 10.71%), so it is not adopted.

## Status

- `Edikted_Forecasting_Exercise_filled.xlsx` (submitted) is untouched.
- The candidate workbook is still built on p2c32a. Rebuilding it with E:ec_ns_x_geo needs two things:
  - 2026 versions of the new features and the re-fitted corrections, which are not yet built;
  - your decision.
- The scoreboard gains `q_ecns`, `nx_ecns`, `nx_ecns_geo`, `nx_ecnsx_geo` and `nx_u65ec`.

## Addendum: the follow-up ideas (D263–D265)

| Idea | Result |
|---|---|
| **Separate new-items model** (learned imputation of the missing history; Tweedie count model; per-month burst probability; burst mixture; a separate 4-month tree) | Not better as a forecast. Young items: system 0.4285, best blend 0.4302, the count / mixture models alone 0.59–0.63. **The burst probability itself is good:** AUC 0.83–0.88 per month, Brier skill about +0.2, well calibrated at the top (0.45 predicted → 0.44 observed). It adds nothing to the system's own breakout signal (AUC 0.672 vs 0.666), so it is kept as a risk flag, not a forecast input. |
| **Volume-weighted new LightGBM model** | Better alone (0.4248 vs 0.4299), level in the system (0.3869 vs 0.3871 with corrections). |
| **Launch plan at look level** (new colours of the same style, look-alike launches in the window) | **Helps:** 0.3933 vs 0.3977 in the system (significant), better bias. It needs Edikted's launch calendar, which we don't have for July–October 2026, so it can't go into the deliverable. It is evidence that the launch plan is worth about 1% of error. |
| **Corrections on the buying ladder** | Level: 10.35% vs 10.36%. |
| **Chasing breakouts in the peak months** | Loses money: 10.49–10.75% vs 10.36%. With a 21-day lead time, stock ordered on a spike arrives as it fades. |

The modelling ideas are now largely exhausted, at about 0.387 × R3's error. Two changes are still running or untested: the TabPFN and NN members on the new features (the fork, on the GPU) and the look-level launch plan, which needs data. The remaining gap needs information the backtest doesn't contain: launch and markdown plans, early traffic / add-to-cart signals, inventory, and the full historical catalogue.
