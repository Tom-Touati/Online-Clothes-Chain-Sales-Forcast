"""Fill a copy of the workbook from data/final_forecast_q4.csv (after the common-sense audit, D050-D056).
Writes only G-J, L, M, N on Forecast and B2-B4 on Model explanation. Each explanation = a short fact line typed from the history,
followed by sentences generated from the forecast file (model answer and range, own trend, rule applied, final, confidence reason)."""
import sys, openpyxl, pandas as pd, numpy as np
from openpyxl.styles import Font
src, dst = sys.argv[1], sys.argv[2]
W = pd.read_csv("data/final_forecast_q4.csv", index_col=0); tg = pd.read_csv("data/targets.csv").set_index("style_color")
n = lambda x: f"{x:,.0f}"
FACT = {
"S24840_LIGHT-GRAY-MELANGE": "World Cup and 4th of July tee launched 25 May; 6,685 units in June, about 306 a day in late June.",
"S24847_NAVY": "World Cup shorts launched 25 May, flat at about 30 a day in June (928 units).",
"S25039_YELLOW": "World Cup tee launched 25 May; 2,584 units in June, about 120 a day at month end.",
"S16079_DARK-GRAY": "No sales since 10 April, an 81-day zero run that is improbable at its selling rate, so most likely out of stock. Per the brief I forecast it as in stock: 32 units a month (Jan-Mar 2026 average) x the month multipliers for mature tops.",
"S23794_BLACK": "Launched 29 April and took off in June (7,711 units; late-June pace about 243 a day). Young launches often keep growing (46% of comparable ones beat their best month over the next four), so the model level was kept; still the largest single-item risk.",
"S23965_BLACK": "Launched 11 April and still near its peak (4,687 in June, about 155 a day).",
"S17159_BLACK-WASHED": "Halloween-tagged faux-leather shorts, steady at about 3,900 a month; spring 2026 matched spring 2025, when Jul-Oct sold 34,393.",
"S15999_BLUE": "Summer nylon shorts at about 3,000 a month, running at about half of last year; flat July-August, then a drop of more than 60% from September, as last year.",
"S14857_BLUE-WASHED": "Frayed denim shorts, 1,628 units in June, with the usual summer-shorts drop from September.",
"S17299_NAVY": "Babydoll tank top in steady decline (about 0.3x last year all spring), with the usual tank-top drop in autumn.",
"S15132_WHITE": "Mature denim at 3,800-5,200 a month this spring, about 1.2x last spring; Jul-Oct 2025 was 22,698, with the August jeans peak.",
"S21376_BLACK": "Launched September 2025, holding at 3,100-3,900 a month; flat July-August, then the usual tank-top drop in autumn.",
"S11233_BLACK": "Oldest item (2021), Halloween-tagged; spring 2026 ran at about 0.8x spring 2025, when Jul-Oct sold 40,794 with an October peak.",
"S14091_BLUE": "Mature denim in long decline (Jul-Oct 2024 56,161, Jul-Oct 2025 35,695; spring 2026 at about half of spring 2025).",
"S17324_BLUE": "4th of July striped tank running at about 2x last year this spring; Jul-Oct 2025 was 8,300.",
"S17954_CREAM": "Heavy cable knit at its summer low (316 in June); rises each month into October, as last year (Oct 2025: 2,226).",
"S17764_BROWN": "Faux-fur jacket under 100 units a month in summer, running at about 0.8x last year; builds into October (Oct 2025: 601).",
"S20850_BLACK-AND-WHITE": "Hoodie launched June 2025 with an autumn turn; last year's Jul-Oct (15,156) included its launch peak.",
"S18184_LIGHT-BLUE": "Young swim top launched in February that peaked in June (1,952).",
"S18974_BLUE": "Young swim top launched in February, 1,417 units in June.",
"S17026_RED": "Older swim top, 550 units in June; last year it sold only about half its June volume over the whole of Jul-Oct.",
"S20878_WHITE": "Graduation dress launched in January; sales fell from 5,820 in May to 1,376 in June as prom season ended, exiting June at about 27 a day. The model expects a summer rebound like graduation items had in 2025 (July about 1.3x June); the risk is to the downside.",
"S22546_WHITE": "White dress whose sales fell through June, from about 800 a week in May to about 140 at month end. The model expects a summer rebound like graduation items had in 2025; the risk is to the downside.",
"S23916_WHITE": "White lace dress launched in March, Halloween-tagged; sales fell through June, from about 850 a week in May to about 210 at month end. In the June 2025 backtest, Halloween and Homecoming items fading in June still rebounded for their season, so the floor applies.",
"S18184_DARK-GRAY": "Swim top launched March 2025; 857 units in June 2026.",
"S19431_WHITE": "Graduation dress whose sales collapsed in June (155, against about 890 in each of April and May); the risk is upside if June was a stock gap.",
"S14995_NAVY": "Older tee in decline; about 15% of its recent online demand was lost to a missing core size, which the model does not see. Not grossed up: about 10% of a small number.",
"S15557_NAVY": "Homecoming and New Year's Eve sequin dress; spring 2026 ran at about last year's level, when Jul-Oct sold 4,035 with a September-October peak.",
"ACC1070-1_MIX": "Winter scarf at its summer low (199 in June) but running 2.4-3.7x last year all through 2026; October 2025 was 519.",
"S23426_BLACK": "January launch past its peak (3,426 in April, 1,828 in June).",
"S14842_BLACK": "Halloween-tagged faux-leather corset running at about 1.2x last year, when Jul-Oct sold 9,839 with an October peak.",
}
WC = ["S24840_LIGHT-GRAY-MELANGE", "S24847_NAVY", "S25039_YELLOW"]
def expl(sc):
    r = W.loc[sc]; parts = []
    fact = FACT.get(sc)
    if fact is None:
        if r.seg_rule_pct == 0 or abs(r.seg_rule_pct) < 0.03: return None
        fact = (f"{r.ptype} published" if r.seg == "swimwear" else f"{r.seg}-tagged {r.ptype.lower()} published") + f" {tg.loc[sc, 'published_date']}; {n(r.jun)} units in June 2026."
    parts.append(fact)
    if sc == "S16079_DARK-GRAY":
        parts.append(f"Final {n(r.final_total)}. Confidence 3: no model range for this item."); return " ".join(parts)
    parts.append(f"Model answer {n(r.model_answer)} (median; P25-P75 {n(r.p25c)}-{n(r.p75c)}).")
    if r.mature:
        parts.append(f"Its own trend (Jul-Oct 2025 x this year's change x 0.84 ageing) says {n(r.yoy_naive)}; I averaged the two: {n(r.after_yoy_blend)}.")
    if abs(r.seg_rule_pct) >= 0.01:
        if r.seg == "swimwear": parts.append(f"Swimwear rule: capped at 1.82x June, the higher 2024-25 swim ratio ({r.seg_rule_pct:+.0%}).")
        else: parts.append(f"{r.seg} rule: raised to {r.seg_rule_mult:.2f}x June, the 30th percentile of 2025 {r.seg} items, a level tested at the June and July 2025 origins ({r.seg_rule_pct:+.0%}).")
    if r.seg in ("Homecoming", "Halloween", "swimwear") and not r.mature:
        parts.append({"Homecoming": "Months follow the Homecoming pattern, peaking in September.", "Halloween": "Months follow the Halloween pattern, peaking in October.", "swimwear": "Months follow the swimwear pattern, falling from July."}[r.seg])
    if sc in WC:
        parts.append("I set July to the late-June pace until the 19 July final and capped Aug-Oct at 75/50/45% of July, the post-event decay of past holiday items.")
    parts.append(f"Final {n(r.final_total)}.")
    reason = {"number set by judgement": "the number is set by judgement, not the model range", "P25-P75 inside +/-20%": "the P25-P75 range sits inside +/-20% of the answer",
              "P25-P75 inside +/-50%": "the P25-P75 range sits inside +/-50% of the answer", "P25-P75 wider than +/-50%": "the P25-P75 range is wider than +/-50%",
              "capped at 2: model and own trend differ by more than 50%": "capped at 2 because the model and its own trend differ by more than 50%",
              "model and own trend differ by more than 2x": "the model and its own trend differ by more than 2x"}[r.conf_reason]
    parts.append(f"Confidence {int(r.conf)}: {reason}.")
    return " ".join(parts)

APPROACH = ("A quantile model of each item's four-month total, plus each item's own year-on-year trend where it has one. One pooled LightGBM across all 1,500 items rather than a model per item: half of the 100 targets have never been through a July-October and 25 launched in 2026, so they borrow seasonality and launch curves from the other 1,400. "
"Each training row is an item at a month-end; the target is the log ratio of the next four months' units to four times the item's recent level (last 91 days). Seven quantiles (10th-90th percentile), each the average of a lean and a full-feature model. Inputs: the last six months relative to level, age, product type, start month, occasion tags, colour number, 7- and 28-day pace, channel mix, discount, like-for-like seasonal ratios and the tagger attributes.\n"
"The answer is the median. On the backtest origins that match this task (June and July 2025) the median had zero bias; a higher quantile only looked better because of Black Friday windows at other origins. For the 42 items with a full year of history, the answer is a 50/50 blend of the model with the item's own trend (last year's Jul-Oct x this year's year-on-year change x 0.84 for ageing). A separate monthly model sets the monthly shape, blended with the item's own last-year shape where it has one.\n"
"Seasonal events and judgement: swimwear is capped at 1.82x June, and younger Homecoming and Halloween items are floored at 6.15x and 5.51x June (the 30th percentile of 2025 items, a level chosen by backtests at the June and July 2025 origins), with those segments' September and October peaks. The 2026 campaign calendar mostly repeats 2025; I added only the new August Clearance (+7%) and late-September sale (+4%) on each item's edikted.com share. World Cup items are cut to their late-June pace until the 19 July final, then decayed like past holiday items. One out-of-stock item is forecast as if in stock.\n"
"Data: cancelled lines excluded (including 145k units cancelled on 14 Oct 2025), launch months treated as partial, published dates converted to New York time, stock-outs checked (about 0.4% of units) and left as they are.")
TESTING = ("Rolling backtest from 13 month-ends (Feb 2025 to Feb 2026), each forecasting the next four months with only the data available at the time, scored on the four-month total per item. Wherever possible, a choice was fitted on earlier origins and checked on later ones.\n"
"The quantile median scores WAPE 0.32 on the nine recent origins (0.34 for my earlier monthly model). Adding each mature item's own trend cut those items' WAPE from 0.26 to 0.24 over eight origins; at the June and July origins it lifted the share within +/-20% from 30% to 44%. The segment rules were set from the previous year only and tested on 2025: swimwear bias fell from +133% to +9%, and the floor level was chosen where it worked at both the June and July origins.\n"
"Common-sense audit of the final numbers: each segment's Jul-Oct to June ratio and monthly shape sits inside or at the edge of its 2024-2025 range, and no young launch is forecast above the 95th percentile of what comparable launches did.\n"
"Confidence: P25-P75 is widened 1.3x (covering 50%; P10-P90 covers 80%) and centred on the answer. Level 1 if it sits inside +/-20% of the answer, 2 if inside +/-50%, otherwise 3. An item can be at most level 2 where the model and its own trend differ by more than 50%, and is level 3 above 100%. In the backtest, levels 1, 2 and 3 landed within +/-20% about 56-60%, 41% and 27% of the time. Every decision is in log.md.")
WEAK = ("(1) Item-level noise: even the best methods land only 35-45% of items within +/-20%; 35 of the 100 get confidence 3.\n"
"(2) Young launches (Vikkey, Frieda) rest on the model alone; Vikkey is about 6% of the total.\n"
"(3) World Cup items: there is no history of this event; the post-tournament decay is my assumption.\n"
"(4) The segment rules rest on two seasons and small groups (4 swimwear, 5 Homecoming, 14 Halloween targets).\n"
"(5) Bias still swings by up to +/-20% between backtest origins, so the total could miss by that much.\n"
"(6) No inventory data: zero runs and broken size runs are inferred from sales; the Emerson top looks out of stock.\n"
"(7) Retail is 30% of units and growing through new stores; the model has no store-count signal.\n"
"What would fix most of this: inventory and buy plans, the store-opening plan, item-level planned markdowns, and more seasons of history.")

wb = openpyxl.load_workbook(src); ws = wb["Forecast"]; seen = []; n_ex = 0
for r in range(2, ws.max_row + 1):
    sc = ws.cell(r, 2).value
    if not sc: continue
    f = W.loc[sc]
    for col, k in zip("GHIJ", ["f_jul", "f_aug", "f_sep", "f_oct"]): ws[f"{col}{r}"] = int(round(f[k]))
    ws[f"L{r}"] = int(f.conf); e = expl(sc); ws[f"M{r}"] = e; n_ex += e is not None
    ws[f"N{r}"] = int(round(f.raw_total))
    assert str(ws[f"K{r}"].value).startswith("=SUM("), (r, ws[f"K{r}"].value)
    seen.append(sc)
assert len(seen) == 100 and set(seen) == set(W.index), "item mismatch"
me = wb["Model explanation"]
for cell, text in [("B2", APPROACH), ("B3", TESTING), ("B4", WEAK)]:
    old = me[cell].font; me[cell] = text; me[cell].font = Font(name=old.name, sz=old.sz, i=False, color="FF000000")
    rr = int(cell[1:]); lines = sum(max(1, -(-len(p) // 115)) for p in text.split("\n")); me.row_dimensions[rr].height = max(220, lines * 15 + 12)
wb.calculation.fullCalcOnLoad = True; wb.save(dst)
print("written", dst, "| explanations:", n_ex, "| text lengths:", len(APPROACH), len(TESTING), len(WEAK))
for s in ["S17159_BLACK-WASHED", "ACC1070-1_MIX", "S22165_BLACK", "S18184_LIGHT-BLUE", "S24840_LIGHT-GRAY-MELANGE"]: print("-", s, "->", expl(s))
