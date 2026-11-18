"""Fable 5.2 (D389v; user: "Yes, build it" to "Build a corrected workbook (text fixes and better-chosen explanations) for you to approve?").
Writes eda/181n_final_nx7_2026.py from eda/181m_final_nx6_2026.py (the submitted nx6) with the audit's fixes (D389u):
  - Model explanation rewritten (eda/181n_me_text.py): the eight wrong or unsupported statements corrected, about 750 words instead of 1,220;
  - 18 explained items chosen by hand (the four hand-ruled rows, the largest items, the launches most lifted by the launch-weighted tree,
    confidence-1 items, the seasonal risers, a faller, a model split), each with its own two sentences; numbers filled from the data;
  - the buying quantity = the final forecast x the uplift of the five-member P65 mean over the same members' median (it was that mean
    itself, which sat below the forecast for 30 items since the routed tree and the recommender joined);
  - Cascade sheets with readable column names, the log-difference columns dropped (the CSVs keep the code names).
Forecast units, raw model totals, confidence and ranges are unchanged. Output: Edikted_Forecasting_Exercise_candidate_nx7.xlsx."""
s = open("eda/181m_final_nx6_2026.py", encoding="utf-8").read().split("\n")
def idx(prefix, nth=0): return [i for i, l in enumerate(s) if l.startswith(prefix)][nth]
# 1 buying quantity
i = idx('C["buying"] = pd.concat(')
s[i] = ('_b5 = pd.concat([C.rules, W3E.q65.fillna(W3[QB]), C.nn, PC[QB], PD[QB]], axis=1).clip(lower=0).mean(axis=1); _m5 = pd.concat([C.rules, W3E.q50.fillna(W3.q50), C.nn, PC.q50, PD.q50], axis=1).clip(lower=0).mean(axis=1)\n'
        'C["buying"] = C.forecast * (_b5 / _m5.clip(lower=1e-9)).clip(lower=1.0).fillna(1.0)   # D389v: the P65 uplift of the five-member buying mean (D260), applied to the forecast')
# 2 explanations: after the nx6 block's last line
i = idx('for s_, t_ in WCX.items(): C.loc[s_, "explanation"] = t_', 1)
EXPL = r'''
# D389v: explained items chosen by hand; every number is read from the data
_A361 = pd.read_csv("data/final_2026_nx5_d361_cascade.csv", index_col=0).reindex(C.index); _lift = (C.forecast / _A361.forecast - 1)
_u = lambda s, c: f"{float(M.loc[s, c]):,.0f}"; _ly = lambda s: f"{float(M.loc[s, ['2025-07', '2025-08', '2025-09', '2025-10']].sum()):,.0f}"
_r = lambda s: f"{float(M.loc[s, ['2026-04', '2026-05', '2026-06']].mean()):,.0f}"; _lyr = lambda s: f"{float(M.loc[s, ['2025-04', '2025-05', '2025-06']].mean()):,.0f}"
_g = lambda s: f"{C.loc[s, MEM].min():,.0f} to {C.loc[s, MEM].max():,.0f}"; _t = lambda s: f"{int(C.loc[s, 'total']):,}"; _c = lambda s, m: f"{int(C.loc[s, m]):,}"; _l = lambda s: f"{_lift[s]:.0%}"
EXPL = {
 "S23794_BLACK": lambda s: f"Launched 30 April and already the largest item: {_u(s, '2026-06')} units in June, with weekly sales peaking in mid-June. The six models range {_g(s)} and give {_t(s)} (July above June, then fading), including a {_l(s)} lift from the launch-weighted tree. Not adjusted, but the last June week was below the peak, so it is the first item to re-forecast.",
 "S15132_WHITE": lambda s: f"Three years on sale and still rising: {_u(s, '2026-04')} units in April, {_u(s, '2026-06')} in June. Last year it sold {_ly(s)} in July-October from a slightly lower spring rate; the models agree closely ({_g(s)}) and give {_t(s)}, with the August back-to-school peak for jeans. Not adjusted; confidence 1.",
 "S17159_BLACK-WASHED": lambda s: f"Steady at about {_r(s)} a month all spring. Last year it sold {_ly(s)} in July-October from the same rate; the models ({_g(s)}) expect a smaller autumn step for an item now two years old and give {_t(s)}, with October, its Halloween month, the highest. Not adjusted; if last year's step repeats, it will come in higher.",
 "S23965_BLACK": lambda s: f"Launched 12 April; {_u(s, '2026-06')} units in June at a steady 1,000-1,300 a week. The models range {_g(s)} and give {_t(s)} ({_c(s, 'jul')} in July, down to {_c(s, 'oct')} by October), including a {_l(s)} lift from the launch-weighted tree. Not adjusted.",
 "S11233_BLACK": lambda s: f"Five years on sale and declining year on year: about {_r(s)} a month this spring against {_lyr(s)} a year ago. Tagged for Halloween, it peaks in August and again in October, so the models ({_g(s)}) give {_t(s)} ({_c(s, 'oct')} in October), well below last year's {_ly(s)}. Not adjusted.",
 "S20850_BLACK-AND-WHITE": lambda s: f"A hoodie launched last summer; spring sales fell from {_u(s, '2026-04')} in April to {_u(s, '2026-06')} in June, as hoodies do. It sold {_ly(s)} in July-October last year; the models give {_t(s)}, rising to {_c(s, 'oct')} in October. Not adjusted; their wide range ({_g(s)}) is the doubt about a second autumn as strong as its first.",
 "S14091_BLUE": lambda s: f"Three years on sale; about {_r(s)} a month this spring against {_lyr(s)} a year ago, and June fell to {_u(s, '2026-06')}. The models agree ({_g(s)}) on {_t(s)}, with the August peak for jeans, against {_ly(s)} last year. Not adjusted.",
 "S23605_PINK": lambda s: f"On sale since February and in volume since April, at about {_r(s)} a month. The models range {_g(s)} and give {_t(s)}: a little above the spring rate in July and August, then {_c(s, 'oct')} in October, including a {_l(s)} lift from the launch-weighted tree. Not adjusted.",
 "S18184_LIGHT-BLUE": lambda s: f"A bikini top tagged for the 4th of July, on sale since February; June was its best month ({_u(s, '2026-06')} units). The holiday passes early in July and swimwear falls after it, so the forecast is {_c(s, 'jul')} in July and {_c(s, 'oct')} by October, {_t(s)} in total (models {_g(s)}). Not adjusted; if the June rise was mostly holiday demand, July will come in lower.",
 "S15999_BLUE": lambda s: f"A summer short two years on sale, steady at about {_r(s)} a month, half its rate of a year ago ({_lyr(s)}). The models agree ({_g(s)}) on {_t(s)}: July and August a little above the current rate, then the usual drop after summer. Not adjusted; confidence 1.",
 "S21729_GRAY-AND-WHITE": lambda s: f"A sweater launched last autumn; spring sales fell to {_u(s, '2026-06')} in June, as expected. Sweaters climb from August, so the forecast rises to {_c(s, 'oct')} in October, {_t(s)} in total. Not adjusted; the models range widely ({_g(s)}) on how strong its second autumn will be.",
 "S21911_LIGHT-GRAY": lambda s: f"Small in spring (about {_r(s)} a month), but June more than doubled to {_u(s, '2026-06')}. It sells with the sweaters, so the models ({_g(s)}) give {_t(s)}, more than three times the recent rate, on the autumn season and its June momentum. Not adjusted.",
 "S20878_WHITE": lambda s: f"A graduation dress: sales peaked in May ({_u(s, '2026-05')} units) and fell to {_u(s, '2026-06')} in June once the season ended, with the last June week under 200. The models ({_g(s)}) give {_t(s)}, about half the spring rate. Not adjusted, but with graduation over the late-June pace points lower: the clearest downside risk among the larger items.",
 "S21143_BLACK": lambda s: f"About a year on sale; June dipped to {_u(s, '2026-06')} after {_u(s, '2026-05')} in May. It sold {_ly(s)} in July-October last year, in its launch season; the models split ({_g(s)}) and their average gives {_t(s)}. Not adjusted; one of the widest model ranges among the larger items.",
}
C["explanation"] = ""
for s_, fn_ in EXPL.items(): C.loc[s_, "explanation"] = fn_(s_)
for s_ in OVR: C.loc[s_, "explanation"] = ("Out of stock: no sales since 10 April. Forecast as if back in stock, as the brief asks: its January-March rate (about 32 a month) "
    "x the seasonal pattern of tops at least a year old. No model covers it, so the number is a rule; confidence 3.")
for s_, t_ in WCX.items(): C.loc[s_, "explanation"] = t_
assert (C.explanation != "").sum() == 18, (C.explanation != "").sum()
for s_ in EXPL: assert abs(C.loc[s_, "total"] - C.loc[s_, "forecast"]) <= 2, s_   # "Not adjusted" must be true
_upct = round(float((C.buying / C.final.clip(lower=1)).median() - 1) * 100); print("buying / final: median +%d%%, below the forecast for %d items" % (_upct, int((C.buying < C.final - 0.5).sum())))'''
s[i] = s[i] + EXPL
# 3 readable Cascade sheets
i = idx("for name, D in [("); j = idx("    w2 = wb.create_sheet(name)")
REN = r'''_REN = {"R3": "Recent rate x4 (R3)", "season": "Seasonal baseline", "rules": "Model 1: planning rules", "qnx": "Model 2: LightGBM", "nn": "Model 3: neural network", "pf2c": "Model 4: TabPFN, calibrated",
        "pf2d32": "Model 5: TabPFN", "rc": "Model 6: recommender (1/4 weight)", "avg5": "Weighted average", "forecast": "After corrections (raw model total)", "final": "Final (Forecast sheet total)",
        "buying": "Buying quantity", "member_spread": "Highest / lowest model", "confidence": "Confidence"}
_LAY = [c for c in LAY if c != "p2c32a_prev"]
for name, D in [("Cascade (items)", C.reset_index().rename(columns={"index": "style_color"})[["style_color", "product_type"] + _LAY + ["member_spread", "confidence", "P10", "P25", "P75", "P90"]].rename(columns=_REN)),
                ("Cascade (hierarchy)", H[["level", "group"] + _LAY].rename(columns=_REN)), ("Ranges (totals)", TRG)]:'''
s[i:j] = REN.split("\n")
# 4 Model explanation
i = idx("ME = {"); j = idx('wm = wb["Model explanation"]')
ME = open("eda/181n_me_text.py", encoding="utf-8").read().split('"""', 2)[2].strip()   # the Model explanation text lives in eda/181n_me_text.py
s[i:j] = ME.split("\n")
# 5 boxes sized to their text, and the 'Method and decisions' sheet (one row per major decision; DEC in eda/181n_me_text.py)
SHEET = r"""# D389v: boxes sized to their text (the template rows are 220 points; Excel's limit is 409), and the 'Method and decisions' sheet
import math
from openpyxl.styles import Alignment, Font
for row in wm.iter_rows(min_row=2, max_row=4):
    _t = row[1].value or ""; _n = sum(max(1, math.ceil(len(p_) / 118)) for p_ in _t.split("\n")); wm.row_dimensions[row[0].row].height = min(409, max(220, 15.5 * _n + 8))
    print("box", row[0].value, len(_t), "chars,", _n, "lines, height", wm.row_dimensions[row[0].row].height)
wd = wb.create_sheet("Method and decisions", wb.sheetnames.index("Model explanation") + 1)
wd.append(["Each major choice behind the forecast, and the evidence for it. Scores are the forecast's error relative to R3 (the last three months x 4) on the nine most recent backtest month-ends unless said otherwise; lower is better."])
wd.merge_cells("A1:C1"); wd["A1"].alignment = Alignment(wrap_text=True, vertical="top"); wd.row_dimensions[1].height = 34
wd.append(["Area", "Decision", "Why, and the evidence"])
for c_ in wd[2]: c_.font = Font(bold=True)
for a_, d_, w_ in DEC:
    wd.append([a_, d_, w_]); r_ = wd.max_row
    for c_ in wd[r_]: c_.alignment = Alignment(wrap_text=True, vertical="top")
    wd.row_dimensions[r_].height = 15.5 * max(math.ceil(len(d_) / 44), math.ceil(len(w_) / 128)) + 6
wd.column_dimensions["A"].width = 14; wd.column_dimensions["B"].width = 42; wd.column_dimensions["C"].width = 120; wd.freeze_panes = "A3"
wb.save(WB_OUT)"""
i = [k for k, l in enumerate(s) if l == "wb.save(WB_OUT)"]; assert len(i) == 1, i
s[i[0]] = SHEET
t = "\n".join(s)
for a, b in [("Edikted_Forecasting_Exercise_candidate_nx6_TEST.xlsx", "Edikted_Forecasting_Exercise_candidate_nx7_TEST.xlsx"), ("Edikted_Forecasting_Exercise_candidate_nx6.xlsx\")", "Edikted_Forecasting_Exercise_candidate_nx7.xlsx\")"),
             ('"data/final_2026_nx6_total_ranges"', '"data/final_2026_nx7_total_ranges"'), ('"data/final_2026_nx6_cascade" +', '"data/final_2026_nx7_cascade" +'), ('"data/final_2026_nx6_cascade_hierarchy"', '"data/final_2026_nx7_cascade_hierarchy"')]:
    assert t.count(a) == 1, (a, t.count(a)); t = t.replace(a, b)
t = t.replace('"""Fork 2, candidate nx6 (overnight goal; D389n):', '"""candidate nx7 (D389v, Fable 5.2): nx6 with the audit fixes of D389u (corrected and shorter Model explanation, 18 hand-chosen explanations, buying quantity as an\nuplift on the forecast, readable Cascade sheets); generated by eda/181n_make_nx7_builder.py. Forecast units, raw totals, confidence and ranges as nx6. nx6 header follows.\nFork 2, candidate nx6 (overnight goal; D389n):', 1)
open("eda/181n_final_nx7_2026.py", "w", encoding="utf-8").write(t); print("wrote eda/181n_final_nx7_2026.py")
