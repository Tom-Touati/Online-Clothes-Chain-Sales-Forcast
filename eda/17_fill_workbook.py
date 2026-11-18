"""Fill a copy of the assignment workbook from data/final_forecast.csv. Writes only the designated input cells (G-J, L, M, N on Forecast; B2-B4 on Model explanation)."""
import sys, openpyxl, pandas as pd
from openpyxl.styles import Font
src, dst = sys.argv[1], sys.argv[2]
W = pd.read_csv("data/final_forecast.csv", index_col=0)

EXPL = {
"S24840_LIGHT-GRAY-MELANGE": "World Cup and 4th of July tee launched 25 May; 6,685 units in June, about 306 a day in the last two weeks. Model said 32,344, treating it as a normal launch still climbing. I set July to the late-June pace until the 19 July final and capped Aug-Oct at 75/50/45% of July (the post-event decay of past 4th of July and St Patrick's items). Adjusted -29%.",
"S24847_NAVY": "World Cup shorts launched 25 May, flat at about 30 a day through June (928 units). Model said 5,549, with July at almost double June's pace. Same tournament-end rule as the other World Cup items: late-June pace to 19 July, then 75/50/45% of July. Adjusted -59%; the least certain item in the list.",
"S25039_YELLOW": "World Cup tee launched 25 May; 2,584 units in June and about 120 a day at month end. Model said 13,738. Same tournament-end rule: late-June pace to the 19 July final, then 75/50/45% of July for Aug-Oct. Adjusted -34%.",
"S16079_DARK-GRAY": "No sales since 10 April, an 81-day zero run that is improbable at its selling rate, so most likely out of stock. The model cannot score it. Per the brief I forecast it as in stock: 32 units a month (Jan-Mar 2026 average) x the model's month multipliers for mature tops.",
"S23794_BLACK": "Launched 29 April and took off in June (7,711 units, peak week about 2,600). Model said 26,856, about 20% above the late-June daily pace for July, because young items usually peak in months 2-3. Not adjusted: in the backtest the model beat run-rate and launch-curve methods for young items. Biggest single-item risk.",
"S15132_WHITE": "Mature denim, steady at 3,800-5,200 a month this spring. Model said 22,789, matching both the structured baseline (22,931) and last year's Jul-Oct (22,698), with the usual August jeans peak. No item-specific adjustment. Confidence 1: high-volume items where the model and baseline agree landed within +/-20% 56% of the time in the backtest.",
"S21376_BLACK": "Launched September 2025 and holding at 3,100-3,900 a month since spring. Model said 12,240, agreeing with the baseline (12,568): flat July-August, then the usual tank-top drop in autumn. No item-specific adjustment. Confidence 1 on the same backtest rule as Roman jeans.",
"S11233_BLACK": "Oldest item (2021), Halloween-tagged, running at about a third of last year's pace (2,590 in June; Jul-Oct 2025 was 40,794). Model said 18,166, keeping an October lift for Halloween; the baseline says 13,506. No item-specific adjustment.",
"S14091_BLUE": "Mature denim in long decline (Jul-Oct 2024 56,161, Jul-Oct 2025 35,695). Model said 13,465, close to the baseline (13,986), with an August back-to-school peak. No item-specific adjustment; last year is not a usable anchor for items this old.",
"S17954_CREAM": "Heavy cable knit at its summer low (316 in June). Model said 2,687, rising each month into October as it did last year (Oct 2025: 2,226). That is about half of last year's Jul-Oct (4,947), in line with the usual year-on-year decline of older items. No item-specific adjustment.",
"S17764_BROWN": "Faux-fur outerwear under 100 units a month in summer. Model said 833, building to 286 in October (Oct 2025: 601). Low volume and a steep seasonal turn, so I am guessing the direction more than the number.",
"S20850_BLACK-AND-WHITE": "Hoodie with an autumn turn: model said 9,991, soft over summer and climbing to about 2,860 in October. Last year's Jul-Oct (15,156) included its launch peak, so it overstates what to expect now. No item-specific adjustment.",
"S18184_LIGHT-BLUE": "Swim top that peaked in June (1,952). Model said 4,140, falling by about 60% from July to September with the swimwear category. No item-specific adjustment.",
"S17026_RED": "Older swim top, 550 units in June. Model said 524, falling from about 240 in July to 66 in October, close to its October 2025 level (72). No item-specific adjustment.",
"S20878_WHITE": "Graduation dress launched in January; sales fell from 5,820 in May to 1,376 in June as prom season ended. Model said 6,829, continuing at about June's pace; the baseline (13,051) is anchored on the spring peak. Kept the model.",
"S19431_WHITE": "Graduation dress whose sales collapsed in June (155, against about 890 in each of April and May). Model said 780, following the June level; the baseline says 2,273. Kept the model; the risk is upside if June was a stock gap.",
"S14995_NAVY": "Older tee in decline. About 15% of its recent online demand was lost to a missing core size, which the model does not see. Model said 673. Not grossed up: the effect is about 10% of a small number, inside the forecast error.",
"S15557_NAVY": "Homecoming and New Year's Eve sequin dress. Model said 1,492, peaking in September (about 560) for Homecoming, in line with last autumn (Oct 2025: 960). No item-specific adjustment.",
"ACC1070-1_MIX": "Winter scarf at its summer low (199 in June), but running at about three times last July's level. Model said 1,490, building to about 470 in October, above last year's Jul-Oct (1,023) for that reason. No item-specific adjustment.",
"S15999_BLUE": "Summer nylon shorts at about 3,000 a month. Model said 8,363: flat July-August, then a 60% drop from September, as last year (Oct 2025: 1,525). The baseline (11,621) is higher; kept the model.",
"S23426_BLACK": "January launch past its peak (3,426 in April, 1,828 in June). Model said 5,631, continuing the decline; the baseline (13,267) uses the three-month average and misses the slide. Kept the model.",
"S23965_BLACK": "Launched 11 April and still near its peak (4,687 in June, about 155 a day). Model said 14,737, following the launch curve with a drop from September. Young item, so wide error either way.",
"S17159_BLACK-WASHED": "Halloween-tagged faux-leather shorts, steady at about 3,900 a month. Model said 21,035, keeping an October lift (Oct 2025: 9,615). No item-specific adjustment.",
}

APPROACH = ("One pooled LightGBM across all 1,500 items rather than a model per item: half of the 100 targets have never been through a July-October and 25 launched in 2026, so they borrow seasonality and launch curves from the other 1,400. "
"Each training row is an item at a month-end predicting each of the next four months; the target is the log ratio of that month's units to the item's recent level (last 91 days), with an absolute-error loss so the forecast aims at the median. "
"Inputs: the last six months relative to that level, age, product type, target month, horizon, occasion tags, colour number, 7- and 28-day pace, channel mix, discount, like-for-like seasonal ratios (only items already live a year earlier) and the tagger attributes. The forecast is the average of a lean and a full-feature version.\n"
"Seasonality and events: learned from target month x type x age x occasion tag across items. I never read trend or seasonality off raw totals, because the 1,500 items are survivors added over time. The 2026 campaign calendar mostly repeats 2025 (the Labor Day sale, the October sale and the 4th of July push ran on the same dates), so the model already sees it; I added only the new August Clearance (+7%) and late-September sale (+4%) on each item's edikted.com share.\n"
"Judgement: the three World Cup items cut to their late-June pace until the 19 July final, then decayed like past holiday items; one item with no sales since April forecast as if in stock. "
"Data: cancelled lines excluded (including 145k units cancelled on 14 Oct 2025), launch months treated as partial, published dates converted to New York time, stock-outs checked (about 0.4% of units) and left as they are.")

TESTING = ("Rolling backtest from 13 month-ends (Feb 2025 to Feb 2026), each forecasting the next four months with only the data available at the time, scored on the four-month total per item (the confidence scale): WAPE, bias, and share within +/-20% and +/-50%.\n"
"What I learned: (1) two test windows misled me. They showed a -25% bias that averages out over 13 (mean -4%, but +/-18% from one window to the next). "
"(2) The final model scores WAPE 0.36 (0.37 on the 100 targets), with 33% of items within +/-20% and 70% within +/-50%. A flat run-rate scores 0.69, and level x like-for-like seasonal ratio x age decay scores 0.42. "
"(3) Several standard ideas did not pay here and were dropped: Tweedie loss, separate models per horizon or category, launch-curve analogs for young items, sibling-colour features, rescaling to last year, and a downward shade.\n"
"Confidence comes from backtest hit rates. Only high-volume items where the model and the baseline agree landed within +/-20% more than half the time (56%), so just 2 items get a 1. A 3 goes to launches under three months old, items under 100 units a month, the World Cup items and the out-of-stock item. Everything else is a 2: in the backtest about 70% of items landed within +/-50%. Every decision is recorded in log.md.")

WEAK = ("(1) Overall level. At last June's origin the model under-forecast by about 14%, and this year its four-month multiplier for mature items (1.04x recent level) is below what the last two Junes delivered (1.12x and 1.22x). Rescaling hurt across the backtest, so I did not apply it; but if July-October repeats 2025, the total could be 10-15% higher.\n"
"(2) Young launches (seven items under three months old, such as Vikkey and Frieda). The model follows launch curves and sits above their current daily pace; they could miss by 30-50% either way.\n"
"(3) World Cup items. There is no history of this event; the post-tournament decay is my assumption.\n"
"(4) Item-level noise. Even with a perfect overall total, only about a third of items land within +/-20%, so each number carries a wide error band.\n"
"(5) No inventory data. Zero runs and broken size runs are inferred from sales; the Emerson top looks out of stock.\n"
"(6) Retail is now 30% of units and growing through new stores (five opened since April); the model has no store-count signal.\n"
"What would fix most of this: inventory and buy plans, the store-opening plan, item-level planned markdowns, and more seasons of history.")

wb = openpyxl.load_workbook(src)
ws = wb["Forecast"]
seen = []
for r in range(2, ws.max_row + 1):
    sc = ws.cell(r, 2).value
    if not sc: continue
    f = W.loc[sc]
    for col, k in zip("GHIJ", ["f_jul", "f_aug", "f_sep", "f_oct"]): ws[f"{col}{r}"] = int(round(f[k]))
    ws[f"L{r}"] = int(f.conf)
    if sc in EXPL: ws[f"M{r}"] = EXPL[sc]
    ws[f"N{r}"] = int(round(f.raw_total))
    assert str(ws[f"K{r}"].value).startswith("=SUM("), (r, ws[f"K{r}"].value)
    seen.append(sc)
assert len(seen) == 100 and set(seen) == set(W.index), "item mismatch"
me = wb["Model explanation"]
for cell, text in [("B2", APPROACH), ("B3", TESTING), ("B4", WEAK)]:
    old = me[cell].font
    me[cell] = text
    me[cell].font = Font(name=old.name, sz=old.sz, i=False, color="FF000000")
    r = int(cell[1:]); lines = sum(max(1, -(-len(p) // 115)) for p in text.split("\n"))     # column B is 110 wide
    me.row_dimensions[r].height = max(me.row_dimensions[r].height or 0, lines * 15 + 12)
wb.calculation.fullCalcOnLoad = True
wb.save(dst)
print("written", dst, "| explanations:", sum(s in EXPL for s in seen), "| lengths:", len(APPROACH), len(TESTING), len(WEAK))
