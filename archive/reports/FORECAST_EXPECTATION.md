# Forecast expectation, July–October 2026

Written on 17 Nov 2026, before any July–October 2026 sales are known (proposal 14 of the error study). First frozen on candidate_nx3 (D316, md5 `ac444112722d4b161a6c22191cfc2a49`). Re-frozen on candidate_nx4 (D319) after Main's D350: the units are identical; the confidence levels follow the validated rule (D318) and the P10–P90 ranges are widened on the upper side (D317). Nothing below may be tuned after the actuals arrive.

## What is frozen

- **Workbook:** `Edikted_Forecasting_Exercise_filled.xlsx`, md5 `c0f7df24184659889d1809adbfb5a42d` (candidate_nx4, D350). 100 targets, **601,801 units**: July 171,591, August 175,788, September 123,969, October 130,453.
- **Per target** in `data/fk_expectation_2026.csv`: the four months and total, confidence level, the model's own total before the hand rules (620,602), R3 (473,375), the seasonal-naive baseline (593,665), the P10–P90 range and the buying quantity.
- **Confidence levels** (D318): level 1 = narrowest fifth of the range, rising 91-day trend and sales in the same months last year; level 3 = widest fifth; level 2 = the rest. Targets 10 / 82 / 8. Backtest share within ±20%: 58% / 44% / 28%.
- **Ranges** (D317): the lower end as stated, the upper log width × 1.4; the expected coverage below uses the same rule in the backtest.
- **Hand rules:** the three World Cup items carry the cap of the first submission (D033, 34,463 units against the models' 53,270); `S16079_DARK-GRAY`, out of stock, keeps its in-stock forecast (D034).

## What the backtest expects

The same season in the backtest is the June 2025 origin (forecasting July–October 2025). The May and July 2025 origins give the range. Only 100 items will be scored, so the last column shows how much each number moves by chance alone. It is the 10th–90th percentile over 2,000 random draws of 100 items from the June 2025 origin, among items at least as large as the targets (R3 ≥ 1,063; 339 items).

| Metric | June 2025 origin | May–July 2025 origins | Any 100 items (80% band) |
|---|---|---|---|
| Absolute error, Σ|F − y| ÷ Σy | 29% | 27% to 29% | 25% to 31% |
| WAPE ÷ R3 | 0.69 | 0.54 to 0.79 | 0.64 to 0.77 |
| Error ÷ seasonal naive's error | 0.75 | 0.59 to 0.75 | 0.68 to 0.84 |
| Bias, ΣF ÷ Σy − 1 | -10% | -10% to +1% | -13% to -4% |
| Items within ±20% | 39% | 36% to 39% | 37% to 48% |
| Units in items within ±20% | 38% | 38% to 51% | 32% to 46% |
| Typical item ÷ R3 (GMRAE) | 0.75 | 0.58 to 0.93 | 0.56 to 0.73 |
| Actuals inside P10–P90 (widened upper side) | 72% | 72% to 81% | 67% to 77% |
| Actuals above P90 | 18% | 2% to 18% | 17% to 25% |

## How to read the result

- **Inside the 80% band:** the forecast did what the backtest said it would.
- **Worse than the band on absolute error or bias:** something changed that the backtest did not contain, such as a new channel, a different roll-out pace, or a promotion calendar unlike 2025's. Look at the targets that drove it before blaming the models.
- **Coverage below the band:** the ranges are too narrow for this season even after the upper side was widened.
- **Bias:** a negative bias of 5–10% is expected; the backtest under-forecasts growth in this season.
- **Ratio to R3:** depends on how much the season moves. Absolute error is the steadier number.

## How to score it

```
.venv/Scripts/python.exe eda/score_2026_actuals.py <actuals.csv>
```

The actuals file needs one row per target and month (`style_color, month, units`, month as `2026-07` … `2026-10`) or one row per target (`style_color, jul, aug, sep, oct`). Use the same unit definition as the training data: order lines excluding cancellations. The script checks the workbook's md5 against the frozen one and prints every metric above with its expected range.
