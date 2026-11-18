# Forecast notes: Mar–Jun 2025 per style_color (snapshot 2025-02-28)

`python rules.py` reads `data/` and writes `predictions.csv` (626 items × 4 months, ~1 s).
The research scripts are in `research/` (e1–e10 = exploration; bt*.py = backtests).

## 1. Key findings (numbers from the data)

**The panel and its growth**
- There are 652 items. Monthly data starts 2024-01. Panel units grew from 101k (Jan-24) to 405k (Feb-25). The number of selling items grew from 211 to 647, so raw totals mostly measure panel growth.
- `to_forecast` holds every item launched up to Jan-2025: 50 launched in Jan-25, none in Feb-25. Five items have had no sales since January.
- Launch date = min(published_date, first_sale). The median gap between the two is 1 day.

**Like-for-like (LFL) month-over-month index.** This covers items live for more than 30 days and selling in the earlier month:

| Channel | Mar-24 | Apr | May | Jun |
|---|---|---|---|---|
| All units | 0.91 | 1.05 | 1.20 | 0.92 |
| edikted.com only | 0.79 | 0.95 | 1.00 | 0.90 |

- Chained from Feb-24, edikted.com is 0.80 / 0.76 / 0.79 / 0.71 (monthly totals). Per day that is 0.75 / 0.74 / 0.74 / 0.69.

**Channel shifts distort 2024**
- TikTok went from 136 units (Feb-24) to 42k (May-24), about 20% of May-24 units.
- Retail stores opened: LA Grove in May-24, MN in Jun-24, two CA stores in Nov-24.
- Dropship grew roughly 1.4× in spring 2024.
- For the items live in Feb-24, "other" channels went 13.5k → 24k / 37k / 53k / 33k in Mar–Jun-24. In Feb-25 these channels are mature: TikTok is about 1,000/day and dropship about 1,000/day, both flat week to week.

**Discount depth drives web volume strongly**
- The site web discount was 0.60 for Feb 1–20 2024, giving about 4.5k web units/day. On Feb 21–24 it fell to 0.31–0.41 and units fell to about 1.6k/day.
- The 2024 spring average discount was about 0.46, against 0.55 in Feb-24.
- 2025 is promo-heavy: 0.60 in January and 0.61 in February. In the last 3 days of Feb-25 it fell to 0.54, and web units fell about 45% against comparable weekdays.
- Item-level discount depth did not explain residuals in backtests (actual/pred is about 0.92–1.06 across discount buckets), so it is not used per item.

**Base month distortion.** Feb-24 web was +53% per day over Jan-24, a promo month. Feb-25 was only +9% per day over Jan-25. A Feb-anchored seasonal index therefore implies a much bigger spring drop than a Jan-anchored one (Mar per-day factor 0.75 vs 1.04). Section 3 shows this cost +30% bias in a 1-month out-of-time test.

**Product-type seasonality is strong.** Chained LFL Feb-24 → m, relative to the whole panel:

| Type | Mar | Apr | May | Jun |
|---|---|---|---|---|
| Shorts | 1.36 | 2.10 | 2.63 | 3.64 |
| Short Jeans | 2.35 | 2.85 | 3.97 | 4.61 |
| Crop Top | 1.8 | 2.1 | 2.7 | 2.2 |
| Tank Top | 1.12 | 1.12 | 1.21 | 1.23 |
| Skirt | ~1.2 | ~1.2 | ~1.2 | ~1.2 |
| Dress | 1.28 | 1.56 | 1.29 | 1.30 |
| Jeans | 0.81 | 0.83 | 0.75 | 0.69 |
| Hoodie | 1.03 | 0.54 | 0.26 | 0.19 |
| Sweater | 0.96 | 0.56 | 0.32 | 0.32 |
| Sweatpants | 0.96 | 0.61 | 0.33 | 0.25 |
| Jacket | 0.61 | 0.55 | 0.28 | 0.15 |

- Long-sleeve tops and dresses behave like winter wear: 0.78 / 0.53 / 0.30 / 0.25 (n ≈ 32–43). Sleeveless items run 1.14 / 1.35 / 1.33 / 1.22.
- Pooled groups:
  - WINTER: 0.78 / 0.52 / 0.30 / 0.25
  - SUMMER: 1.19 / 1.36 / 1.47 / 1.54
  - CORE: 0.94 / 0.92 / 0.94 / 0.90
- **It repeats year over year.** The relative Jan→Feb moves were almost identical in 2024 and 2025:

| Segment | 2024 | 2025 |
|---|---|---|
| SUMMER | 1.13 | 1.15 |
| WINTER | 0.83 | 0.86 |
| Shorts | 1.20 | 1.20 |
| Tank Top | 1.11 | 1.14 |
| Skirt | 1.28 | 1.35 |

- Feb-25 is winter-heavy: Hoodie 42k, Sweater 32k, Sweatpants 25k, plus long-sleeve tops. This pulls the spring total down even though summer types multiply.

**Lifecycle.** The month-over-month ratio by age, relative to items 12+ months old (median over 13 months):

| Age at start of month | Relative ratio |
|---|---|
| First full month → second | 1.54 |
| Second → third | 1.26 |
| 3–12 months old | 0.95–1.09, essentially no extra decay |

- Items in the panel are survivors with flat lifecycles after month 2. The backtest preferred a milder ramp than these raw ratios.

**Occasion tags**
- The 4th of July tag runs relative 1.8 / 1.6 / 1.4 / 1.4 for Mar–Jun-24. It overlaps heavily with summer types, so it is not used.
- Graduation and St. Patrick's tags have n ≤ 13 and are noisy.
- Valentine's items showed no clear post-February drop (0.89 relative in March).

**Retail.** The flagship store (NYC SoHo) ran 244 units/day in Feb-24 and 268 / 280 / 344 / 592 in Mar–Jun. Deflated for panel growth, that is 1.03 / 0.93 / 1.07 / 1.66: a summer tourist peak.

**Volume.** In backtests the smallest-volume quintile under-forecasts (actual about 2× pred) and the top quintile over-forecasts by about 7%. Additive and power shrinkage did not improve WAPE, so neither is used.

## 2. Rules (rules.py)

`pred(i,m) = base_rate_i × days(m) × level_i(m) × season_seg(m) × ramp_i(h)`

1. **Base rate (units/day).** Start with 0.6 × last-month rate + 0.4 × last-14-day rate, rescaled so the total equals the last-month total. Then blend 80/20 with the (rescaled) last-91-day rate. Rates use only days the item was live. If an item sold nothing in the last 28 days, pred = 0.
2. **Channel level per item.** Take the item's Feb channel mix (web / retail / other), shrunk to the panel mix with 20 pseudo-units. Each channel gets its own per-day factor from the same months of 2024:
   - **web**: 0.6 × Feb-anchored LFL chain + 0.4 × Jan-anchored LFL chain ÷ this year's Jan→Feb per-day LFL move. The result is 0.87 / 0.86 / 0.86 / 0.80. The Jan anchor hedges the Feb-24 promo spike. Its weight is below 0.5 because of the end-of-Feb-25 discount pullback.
   - **retail**: flagship store units/day vs Feb, deflated for panel growth: 1.03 / 0.93 / 1.07 / 1.66.
   - **other** (dropship, TikTok, Flip): flat 1.0. The 2024 history is channel onboarding, and February-25 is flat week to week.
3. **Seasonal profile.**
   - Segment = product_type, with long-sleeve tops-family items split off as `type|LS`.
   - Its chained LFL relative to the panel (Feb-24 → m-24) is shrunk to its group (WINTER / SUMMER / CORE) with K = 8 pseudo-items, then capped to [0.15, 4].
   - Groups use hand lists of types, plus sleeve: long sleeve → WINTER, sleeveless/strap → SUMMER.
4. **Ramp for young items.**
   - Launched in the month before the base month (Jan-25): ×1.25 in March, then ×1.25 × 1.1 for April–June.
   - Launched in Dec-24: ×1.1 throughout.

Forecast totals:

| Month | Units |
|---|---|
| Mar-25 | 392k |
| Apr-25 | 355k |
| May-25 | 351k |
| Jun-25 | 352k |
| 4-month total | 1.45M |

For comparison, the same items sold 400k in Feb-25 (28 days).

## 3. In-data backtests

**A. Split-half by style.** Seasonal and level factors are measured on half of the styles (in the target months themselves, so in-sample in time); forecasts are scored on the other half. Item 4-month totals:

| Origin | WAPE | Bias | ±20% | ±50% |
|---|---|---|---|---|
| 2024-02-29 (Mar–Jun) | 0.419 | −0.205 | 0.24 | 0.68 |
| 2024-05-31 | 0.309 | +0.005 | 0.35 | 0.69 |
| 2024-06-30 | 0.329 | +0.015 | 0.33 | 0.68 |
| 2024-07-31 | 0.326 | +0.012 | 0.31 | 0.63 |
| 2024-08-31 | 0.420 | +0.101 | 0.28 | 0.62 |
| 2024-09-30 | 0.388 | +0.031 | 0.33 | 0.69 |
| 2024-10-31 | 0.343 | −0.052 | 0.32 | 0.66 |
| **Mean** | **0.362** | −0.013 | 0.31 | 0.66 |

- The monthly WAPE averages 0.41.
- Without type seasonality the mean WAPE is 0.457 (±20%: 0.24; ±50%: 0.57).
- The −20% bias at the Feb-24 origin is fully explained by TikTok and retail onboarding in spring 2024. The assumed flat "other" level does not hold for 2024, but it should for 2025.

**B. Out-of-time, truly honest.** The origin is 2025-01-31, factors come from Jan→Feb 2024, and Feb-25 is scored (1 month):

| Variant | WAPE | Bias |
|---|---|---|
| Unscaled rules | 0.361 | +0.30 |
| Level-scaled | 0.193 | 0 |
| Level-scaled, no type seasonality | 0.230 | 0 |

- The unscaled bias comes from the Feb-24 promo spike. The two-anchor web level now hedges this, but only the Feb anchor is available at this origin.
- Type seasonality validates out of time.

**C. Base, ramp and shrinkage variants.** These used oracle-level, scaled tests at 6 origins, plus the split-half and out-of-time tests. The differences were all small (±0.01 WAPE). Choices:

| Setting | Chosen | Range tested | WAPE |
|---|---|---|---|
| Base blend | 14-day 0.4, 91-day 0.2 | — | 0.449 vs 0.475 for last-month only |
| Ramp | 1.25 / 1.1 | 1.2/1.1 to 1.7/1.3 | 0.361–0.374 |
| Shrinkage K | 8 | 2 / 5 / 10 / 20 | 0.205 / 0.196 / 0.193 / 0.192 out-of-time |

## 4. Known weaknesses

- **Aggregate level is the main risk.** It depends on whether edikted.com keeps discounts near 0.6. If it does, March could run near the Feb level (Jan anchor: 1.04/day). If it pulls back to 2024-like levels, March could be about 0.75/day or lower; the last 3 days of Feb-25 ran −45%. The rules take a blend (~0.87). Every 0.1 on the web factor is about 7% bias.
- Store openings or new marketplaces in Mar–Jun-25 are not anticipated, which is an upside risk. The 2024 June retail peak (×1.66) is assumed to recur.
- Seasonal profiles come from one year with small segments (many types have n < 5). Shrinkage helps, but Shorts / Short Jeans multipliers of 3–4.6× by June are bold. They rest on 10 and 5 items in 2024.
- Lifecycle: there is no explicit decay, because the panel shows none. Sell-outs and stock-outs are invisible in the data and will cause misses for individual items.
- Small items are systematically under-forecast, as found in backtests; no fix was found without hurting WAPE.
- Sibling colours, occasion tags and item-level discount are not used.
- Easter moves from Mar-31 (2024) to Apr-20 (2025). This is ignored.

## Changelog
- **v1 (2025-02-28 snapshot)**: first version. Base blend, channel-weighted level (two-anchor web, deflated flagship retail, flat other), product-type × sleeve seasonal profile shrunk to season group, young-item ramp, zero for items with no sales in 28 days.
