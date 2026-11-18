"""D303 (fork): one-off patch after the user's decision in the session ("Yes, restore"): the World Cup hand rule is back in the submitted
workbook (candidate_nx3, Main D291), so the page's World Cup text, table and system labels are brought in step. Idempotent."""
p = "eda/error_study_template.html"; s = open(p, encoding="utf-8").read()
if "candidate_nx3" in s: raise SystemExit("already patched")


def cut(a, b):
    i = s.index(a); j = s.index(b, i) + len(b); return i, j


i, j = cut("    <li><b>Three World Cup items are forecast as if the tournament never ends.</b>", "</li>")
s = s[:i] + '''    <li><b>Three World Cup items are capped by hand.</b> Three of the 100 targets are World Cup tops and shorts launched on 25 May. Sales will fall after the final on 19 July, and the models have never seen a World Cup, so they extend the June ramp: {{wc_now}} units, with August above July for the main tee. The first submission capped them by hand; the workbooks built from the model chain afterwards did not carry the cap. It was restored on 17 November: {{wc_rule}} units. The <a href="#targets">2026 targets</a> section shows the three items month by month.</li>''' + s[j:]
i, j = cut('    <p style="margin:10px 0 0">The first submission set July from the late-June daily pace', "</p></div>")
s = s[:i] + '''    <p style="margin:10px 0 0">The hand rule sets July from the late-June daily pace (19 tournament days plus 12 days after the final at three quarters of that pace) and caps August, September and October at 0.75, 0.50 and 0.45 of July, the decay measured on earlier 4th of July and St Patrick's items (D033). The workbooks built from the model chain between 13 and 17 November did not carry it. It was restored on 17 November at your decision: {{wc_rule}} units in place of {{wc_now}}, about {{wc_diff}} fewer, or {{wc_pct}} of the total. The submitted total is now {{sub_total}} units; the other 97 targets are unchanged.</p></div>''' + s[j:]
a = "Error study · submitted system (candidate_nx2) · built {{built}}"; assert a in s; s = s.replace(a, "Error study · submitted system (candidate_nx3) · built {{built}}")
a = "<p><b>System.</b> The submitted forecast (candidate_nx2, D273–D274): arithmetic mean of five members (rule agents, LightGBM, neural net, two TabPFN v2 variants) with the segment and breakout corrections."; assert a in s
s = s.replace(a, "<p><b>System.</b> The submitted forecast (candidate_nx3): arithmetic mean of five members (rule agents, LightGBM, neural net, two TabPFN v2 variants) with the segment and breakout corrections (D273–D274), plus two hand rules on four of the 100 targets (out of stock, World Cup). The backtest scores the model system; the hand rules exist only in the 2026 workbook.")
open(p, "w", encoding="utf-8").write(s)

b = "eda/198g_build_error_study.py"; t = open(b, encoding="utf-8").read()
a = '_old, _new = _wb("Edikted_Forecasting_Exercise_filled_original_2026-11-12.xlsx"), _wb("Edikted_Forecasting_Exercise_filled.xlsx")'; assert a in t
t = t.replace(a, '''_old, _new = _wb("Edikted_Forecasting_Exercise_candidate_nx2.xlsx"), _wb("Edikted_Forecasting_Exercise_filled.xlsx")   # the models without the cap | the submitted workbook
_first = _wb("Edikted_Forecasting_Exercise_filled_original_2026-11-12.xlsx"); assert all(abs(x - y_) < 0.51 for k in WC for x, y_ in zip(_new[k][1], _first[k][1])), "the submitted World Cup rows are not the D033 rows"
_wsub = openpyxl.load_workbook("Edikted_Forecasting_Exercise_filled.xlsx", data_only=True)["Forecast"]; sub_total = sum(float(v) for r_ in _wsub.iter_rows(min_row=2, values_only=True) if r_[1] for v in r_[6:10])''')
a = "<td>first submission</td>"; assert a in t; t = t.replace(a, "<td>models, no cap</td>")
a = "'<tr><td>submitted now</td>'"; assert a in t; t = t.replace(a, "'<tr><td>submitted (hand cap)</td>'")
a = "wc_rule = sum(sum(v[1]) for v in _old.values()); wc_now = sum(sum(v[1]) for v in _new.values())"; assert a in t
t = t.replace(a, "wc_now = sum(sum(v[1]) for v in _old.values()); wc_rule = sum(sum(v[1]) for v in _new.values())   # wc_now = the models' number, wc_rule = the capped, submitted number")
x = '    built="17 Nov 2026",'; assert x in t; t = t.replace(x, x + "\n    sub_total=it(sub_total),")
open(b, "w", encoding="utf-8").write(t); print("patched")
