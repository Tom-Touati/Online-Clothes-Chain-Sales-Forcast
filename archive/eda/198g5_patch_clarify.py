"""D280 (fork): one-off patch after two comments on the page ("I don't know this issue, please expand" on the World Cup item and
"I don't fully understand this" on the evidence item): plainer wording for both summary items and a table of the three World Cup
items (first submission against the submitted workbook) in the 2026 targets section. Idempotent."""
p = "eda/error_study_template.html"; s = open(p, encoding="utf-8").read()
if 'id="t-wc"' in s: raise SystemExit("already patched")


def cut(a, b):
    i = s.index(a); j = s.index(b, i) + len(b); return i, j


i, j = cut("    <li><b>The evidence is thinner than the quoted intervals.</b>", "</li>")
s = s[:i] + '''    <li><b>The backtest is weaker evidence than its intervals suggest, for four reasons.</b> First, each origin forecasts four months and origins are one month apart, so neighbouring origins are tested on mostly the same months: 13 origins behave like about {{neff13}} separate tests. Second, {{variants}} model variants were scored on those same origins and the winners kept; with that many tries, a gain under about {{null_max}} can be luck. Third, two of the adopted steps make the forecast worse on the four earliest origins. Fourth, the 100 targets were chosen knowing how they sold later, so the backtest sees them as items that out-perform: their bias of {{tgt_bias}} comes from that choice and should not be corrected for.</li>''' + s[j:]
i, j = cut("    <li><b>One issue is live in the submitted workbook.</b>", "</li>")
s = s[:i] + '''    <li><b>Three World Cup items are forecast as if the tournament never ends.</b> Three of the 100 targets are World Cup tops and shorts launched on 25 May. The first submission capped them by hand, because sales will fall after the final on 19 July. Every workbook built since comes straight from the models, which have never seen a World Cup and so extend the June ramp: {{wc_now}} units against {{wc_rule}} under the cap, with August above July for the main tee. The <a href="#targets">2026 targets</a> section shows the three items month by month.</li>''' + s[j:]
i, j = cut('  <div class="note"><b>World Cup items.</b>', "</div>")
s = s[:i] + '''  <div class="note"><b>World Cup items.</b> The 2026 World Cup runs from 11 June to 19 July. Three targets are tied to it; all launched on 25 May and were still climbing at the end of June. No model can learn the drop after the final, because the training data contains no World Cup.
    <div class="tscroll" style="margin-top:8px"><table id="t-wc"><thead><tr><th>Item</th><th class="r">June actual</th><th>Workbook</th><th class="r">Jul</th><th class="r">Aug</th><th class="r">Sep</th><th class="r">Oct</th><th class="r">Total</th></tr></thead><tbody>{{wc_rows}}</tbody></table></div>
    <p style="margin:10px 0 0">The first submission set July from the late-June daily pace (19 tournament days plus 12 days after the final at three quarters of that pace) and capped August, September and October at 0.75, 0.50 and 0.45 of July, the decay measured on earlier 4th of July and St Patrick's items (D033). The submitted workbook has no such rule: {{wc_now}} units against {{wc_rule}}, a difference of about {{wc_diff}} units or {{wc_pct}} of the total. Restoring the rule changes the submitted workbook, so it is your decision.</p></div>''' + s[j:]
open(p, "w", encoding="utf-8").write(s)

b = "eda/198g_build_error_study.py"; t = open(b, encoding="utf-8").read()
x = '    built="17 Nov 2026",'; assert x in t
t = t.replace(x, x + '\n    wc_rows=WCROWS,')
y = 'T = dict(\n'; assert y in t
pre = '''import openpyxl
def _wb(f):
    ws = openpyxl.load_workbook(f, data_only=True)["Forecast"]; return {r[1]: (r[2], [float(v) for v in r[6:10]]) for r in ws.iter_rows(min_row=2, values_only=True) if r[1] in WC}
_old, _new = _wb("Edikted_Forecasting_Exercise_filled_original_2026-11-12.xlsx"), _wb("Edikted_Forecasting_Exercise_filled.xlsx")
_m = pd.read_parquet("data/monthly_item.parquet"); _m.columns = [str(c) for c in _m.columns]
WCROWS = ""
for k in WC:
    nm, o_ = _old[k]; _, n_ = _new[k]
    WCROWS += f'<tr><td rowspan="2">{nm}<br><span class="small">{k}</span></td><td class="r" rowspan="2">{it(_m.loc[k, "2026-06"])}</td><td>first submission</td>' + "".join(f'<td class="r">{it(v)}</td>' for v in o_) + f'<td class="r">{it(sum(o_))}</td></tr>'
    WCROWS += '<tr><td>submitted now</td>' + "".join(f'<td class="r">{it(v)}</td>' for v in n_) + f'<td class="r">{it(sum(n_))}</td></tr>'
wc_rule = sum(sum(v[1]) for v in _old.values()); wc_now = sum(sum(v[1]) for v in _new.values())
'''
t = t.replace(y, pre + y, 1)
open(b, "w", encoding="utf-8").write(t)
print("patched")
