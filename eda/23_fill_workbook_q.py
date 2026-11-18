"""Fill a copy of the assignment workbook from the quantile forecast (data/final_forecast_q.csv).
Same cells as 17_fill_workbook.py (G-J, L, M, N on Forecast; B2-B4 on Model explanation). Explanation numbers are filled from the data, not typed."""
import sys, openpyxl, pandas as pd
from openpyxl.styles import Font
src, dst = sys.argv[1], sys.argv[2]
W = pd.read_csv("data/final_forecast_q.csv", index_col=0)
def v(sc):
    r = W.loc[sc]; f = lambda x: f"{x:,.0f}"
    return dict(a=f(r.raw_total), lo=f(r.p25c) if pd.notna(r.p25c) else "", hi=f(r.p75c) if pd.notna(r.p75c) else "", fin=f(r.final_total),
                b0=f(r.b0_total) if pd.notna(r.b0_total) else "", jul=f(r.f_jul), sep=f(r.f_sep), oct=f(r.f_oct), avg=f(r.raw_total / 4), adj=f"{r.adj_pct:+.0%}")
EXPL = {
"S24840_LIGHT-GRAY-MELANGE": "World Cup and 4th of July tee launched 25 May; 6,685 units in June, about 306 a day in late June. Model answer {a} (P25-P75 {lo}-{hi}), treating it as a normal launch. I set July to the late-June pace until the 19 July final and capped Aug-Oct at 75/50/45% of July, the post-event decay of past holiday items. Adjusted {adj}; confidence 3 because the number is judgement.",
"S24847_NAVY": "World Cup shorts launched 25 May, flat at about 30 a day in June (928 units). Model answer {a} (P25-P75 {lo}-{hi}). Same tournament-end rule: late-June pace to 19 July, then 75/50/45% of July for Aug-Oct. Adjusted {adj}; the least certain item in the list.",
"S25039_YELLOW": "World Cup tee launched 25 May; 2,584 units in June and about 120 a day at month end. Model answer {a} (P25-P75 {lo}-{hi}). Same tournament-end rule: late-June pace to 19 July, then 75/50/45% of July. Adjusted {adj}.",
"S16079_DARK-GRAY": "No sales since 10 April, an 81-day zero run that is improbable at its selling rate, so most likely out of stock. The model cannot score it. Per the brief I forecast it as in stock: 32 units a month (Jan-Mar 2026 average) x the month multipliers for mature tops. No model range, so confidence 3.",
"S23794_BLACK": "Launched 29 April and took off in June (7,711 units; late-June pace about 243 a day). Model answer {a}, about {avg} a month, above the current pace; P25-P75 {lo}-{hi} is wide, so confidence 3. Not adjusted: the backtest favoured the model over run-rate methods for young launches, but this is the largest single-item risk.",
"S17159_BLACK-WASHED": "Halloween-tagged faux-leather shorts, steady at about 3,900 a month. Model answer {a} with a narrow P25-P75 ({lo}-{hi}) that sits inside +/-20% of the answer, hence confidence 1. October keeps a Halloween lift (Oct 2025: 9,615).",
"S15999_BLUE": "Summer nylon shorts at about 3,000 a month. Model answer {a}: flat July-August, then a drop of more than 60% from September, as last year (Oct 2025: 1,525). P25-P75 {lo}-{hi} sits inside +/-20%, so confidence 1.",
"S14857_BLUE-WASHED": "Frayed denim shorts, 1,628 units in June. Model answer {a} with the usual summer-shorts drop (September {sep}, October {oct}), in line with the structured baseline ({b0}). P25-P75 {lo}-{hi} sits inside +/-20%, so confidence 1.",
"S15132_WHITE": "Mature denim at 3,800-5,200 a month this spring; Jul-Oct 2025 was 22,698. Model answer {a}, above the structured baseline ({b0}), with the August jeans peak. P25-P75 {lo}-{hi} is wider than +/-20%, so confidence 2.",
"S21376_BLACK": "Launched September 2025, holding at 3,100-3,900 a month. Model answer {a}, close to the baseline ({b0}): flat July-August, then the usual tank-top drop in autumn. P25-P75 {lo}-{hi}, confidence 2.",
"S11233_BLACK": "Oldest item (2021), Halloween-tagged, running at about a third of last year's pace (2,590 in June; Jul-Oct 2025 was 40,794). Model answer {a} keeps an October lift for Halloween; the baseline says {b0}. P25-P75 {lo}-{hi}, confidence 2.",
"S14091_BLUE": "Mature denim in long decline (Jul-Oct 2024 56,161, Jul-Oct 2025 35,695). Model answer {a} with an August back-to-school peak; the baseline says {b0}. P25-P75 {lo}-{hi}, confidence 2.",
"S17954_CREAM": "Heavy cable knit at its summer low (316 in June). Model answer {a}, rising each month into October as last year (Oct 2025: 2,226); about half of last year's Jul-Oct (4,947). P25-P75 {lo}-{hi} is wide, so confidence 3.",
"S17764_BROWN": "Faux-fur outerwear under 100 units a month in summer. Model answer {a}, building to {oct} in October (Oct 2025: 601). P25-P75 {lo}-{hi}: the range is wider than +/-50%, so confidence 3.",
"S20850_BLACK-AND-WHITE": "Hoodie with an autumn turn: model answer {a}, soft over summer and climbing to about {oct} in October. Last year's Jul-Oct (15,156) included its launch peak. P25-P75 {lo}-{hi}, confidence 2.",
"S18184_LIGHT-BLUE": "Swim top that peaked in June (1,952). Model answer {a}, falling from {jul} in July to {oct} in October with the swimwear category. P25-P75 {lo}-{hi}, confidence 3.",
"S17026_RED": "Older swim top, 550 units in June. Model answer {a}, falling from {jul} in July to {oct} in October, close to its October 2025 level (72). P25-P75 {lo}-{hi}, confidence 3.",
"S20878_WHITE": "Graduation dress launched in January; sales fell from 5,820 in May to 1,376 in June as prom season ended. Model answer {a}, about {avg} a month; the baseline ({b0}) is anchored on the spring peak. P25-P75 {lo}-{hi} is wide, so confidence 3.",
"S19431_WHITE": "Graduation dress whose sales collapsed in June (155, against about 890 in each of April and May). Model answer {a}; the baseline says {b0}. Confidence 3 (P25-P75 {lo}-{hi}); the risk is upside if June was a stock gap.",
"S14995_NAVY": "Older tee in decline. About 15% of its recent online demand was lost to a missing core size, which the model does not see. Model answer {a} (P25-P75 {lo}-{hi}). Not grossed up: about 10% of a small number, inside the range.",
"S15557_NAVY": "Homecoming and New Year's Eve sequin dress. Model answer {a}, peaking in September ({sep}) for Homecoming, in line with last autumn (Oct 2025: 960). P25-P75 {lo}-{hi} is wide, so confidence 3.",
"ACC1070-1_MIX": "Winter scarf at its summer low (199 in June), but running at about three times last July's level. Model answer {a}, building to {oct} in October, above last year's Jul-Oct (1,023) for that reason. P25-P75 {lo}-{hi}, confidence 3.",
"S23426_BLACK": "January launch past its peak (3,426 in April, 1,828 in June). Model answer {a} continues the decline; the baseline ({b0}) uses the three-month average and misses the slide. P25-P75 {lo}-{hi}, confidence 3.",
"S23965_BLACK": "Launched 11 April and still near its peak (4,687 in June, about 155 a day). Model answer {a}, with a drop from September. P25-P75 {lo}-{hi} sits inside +/-50%, so confidence 2 despite its age.",
}

APPROACH = ("A quantile model of each item's four-month total. One pooled LightGBM across all 1,500 items rather than a model per item: half of the 100 targets have never been through a July-October and 25 launched in 2026, so they borrow seasonality and launch curves from the other 1,400. "
"Each training row is an item at a month-end; the target is the log ratio of the next four months' units to four times the item's recent level (last 91 days). Seven quantile levels (10th to 90th percentile), each the average of a lean and a full-feature model. "
"Inputs: the last six months relative to that level, age, product type, start month, occasion tags, colour number, 7- and 28-day pace, channel mix, discount, like-for-like seasonal ratios (only items already live a year earlier) averaged over the four target months, and the tagger attributes.\n"
"The answer is the 61st percentile, not the median: on the backtest origins most like the final fit, the median came in 9% low in total and the 61st percentile was unbiased. A separate monthly model splits the total into months. I never read trend or seasonality off raw totals, because the 1,500 items are survivors added over time.\n"
"Events and judgement: the 2026 campaign calendar mostly repeats 2025, so the model already sees it; I added only the new August Clearance (+7%) and late-September sale (+4%) on each item's edikted.com share. The three World Cup items are cut to their late-June pace until the 19 July final, then decayed like past holiday items. One item with no sales since April is forecast as if in stock.\n"
"Data: cancelled lines excluded (including 145k units cancelled on 14 Oct 2025), launch months treated as partial, published dates converted to New York time, stock-outs checked (about 0.4% of units) and left as they are.")

TESTING = ("Rolling backtest from 13 month-ends (Feb 2025 to Feb 2026), each forecasting the next four months with only the data available at the time, scored on the four-month total per item. The early origins had too little training data (2.5k-4k rows) and over-forecast, so the answer quantile and the interval width were set on the nine origins from June 2025 on (9,031 item forecasts).\n"
"Point accuracy: the quantile median scored WAPE 0.32 against 0.34 for my earlier monthly model, better in 7 of 9 origins. Moving the answer to the 61st percentile takes total bias from -9% to 0% at a small cost per item (within +/-20%: 36% to 34%). Simple baselines score 0.42 (level x seasonal ratio x age) and 0.69 (flat run-rate).\n"
"Calibration: the raw P25-P75 covered only 40% of actuals, so it is widened 1.3x around the median. That gives 50% coverage for P25-P75 and 80% for P10-P90.\n"
"Confidence comes from that calibrated range: 1 if P25-P75 sits inside +/-20% of the answer, 2 if inside +/-50%, otherwise 3. In the backtest the three levels landed within +/-20% 56%, 41% and 27% of the time, and were off by more than 50% 7%, 21% and 40% of the time. Every decision is recorded in log.md.")

WEAK = ("(1) The answer quantile and the interval width were fitted on the same nine origins used to check them, and bias still swings from -20% to +13% between origins. The total could miss by that much in either direction.\n"
"(2) The unbiased 61st percentile trades a little per-item accuracy for a correct total; with the median, the total would be about 10% lower.\n"
"(3) Young launches (Vikkey, Frieda): the model sits above their late-June pace. Vikkey alone is 6% of the total.\n"
"(4) World Cup items: there is no history of this event; the post-tournament decay is my assumption.\n"
"(5) Item-level noise: only about a third of items land within +/-20% even when the total is right, and 56 of the 100 get confidence 3.\n"
"(6) No inventory data: zero runs and broken size runs are inferred from sales; the Emerson top looks out of stock.\n"
"(7) Retail is now 30% of units and growing through new stores (five opened since April); the model has no store-count signal.\n"
"What would fix most of this: inventory and buy plans, the store-opening plan, item-level planned markdowns, and more seasons of history.")

wb = openpyxl.load_workbook(src)
ws = wb["Forecast"]; seen = []
for r in range(2, ws.max_row + 1):
    sc = ws.cell(r, 2).value
    if not sc: continue
    f = W.loc[sc]
    for col, k in zip("GHIJ", ["f_jul", "f_aug", "f_sep", "f_oct"]): ws[f"{col}{r}"] = int(round(f[k]))
    ws[f"L{r}"] = int(f.conf)
    ws[f"M{r}"] = EXPL[sc].format(**v(sc)) if sc in EXPL else None
    ws[f"N{r}"] = int(round(f.raw_total))
    assert str(ws[f"K{r}"].value).startswith("=SUM("), (r, ws[f"K{r}"].value)
    seen.append(sc)
assert len(seen) == 100 and set(seen) == set(W.index), "item mismatch"
me = wb["Model explanation"]
for cell, text in [("B2", APPROACH), ("B3", TESTING), ("B4", WEAK)]:
    old = me[cell].font; me[cell] = text; me[cell].font = Font(name=old.name, sz=old.sz, i=False, color="FF000000")
    r = int(cell[1:]); lines = sum(max(1, -(-len(p) // 115)) for p in text.split("\n"))
    me.row_dimensions[r].height = max(220, lines * 15 + 12)
wb.calculation.fullCalcOnLoad = True
wb.save(dst)
print("written", dst, "| explanations:", sum(s in EXPL for s in seen), "| text lengths:", len(APPROACH), len(TESTING), len(WEAK))
for s in ["S17159_BLACK-WASHED", "S23794_BLACK"]: print(s, "->", EXPL[s].format(**v(s)))
