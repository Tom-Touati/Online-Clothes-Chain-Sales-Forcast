"""D307-D312 (fork): one-off patch putting the member refits on the error study, after comments on the page ("add the good options then";
"we can add that info as future known"; "also can be a known future"; "anything not related to sales but is a result of new partnerships or
other business decisions can be added as a known future"; "those can be future knowns"; "use that as known future"; "add feature that
states the max sales day in recent history") and the user's requests in the session (last-month inputs; cluster, type and hot / cold trends):
  - a panel 'Inputs added inside the tree model' at the end of the Information section (rows built here from J["refit"], J["refit_future"],
    J["refit_decisions"], J["trend"], J["maxday"], J["future_blocks"])
  - the June 2026 rising and falling products
  - one sentence under the launches table in the Distribution section
Idempotent."""
p = "eda/error_study_template.html"; s = open(p, encoding="utf-8").read()
if 'id="member-inputs"' in s: raise SystemExit("already patched")


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80]); s = s.replace(a, b, 1)


PANEL = '''  <div class="panel" id="member-inputs" style="margin-top:18px"><h3>Inputs added inside the tree model</h3>
    <p>Asked in several comments on this page and in the session: add the useful origin-known signals, recent and trend features, and the business decisions as known futures. Each block was added to the tree model itself (refitted on earlier origins only), which then replaced the submitted tree model in the five-model average. "With corrections" applies the submitted correction layer as it is. With the correction layer re-fitted on the 11-signal version (Main, D293), the system scores 0.3837 on recent origins and 0.4350 on all 13, against 0.3851 and 0.4399 submitted; buying is unchanged.</p>
    <div class="tscroll"><table id="t-mi"><thead><tr><th>Inputs added to the tree model</th><th class="r">Tree model alone</th><th class="r">Five-model average</th><th class="r">With corrections</th><th class="r">Within ±20%</th><th class="r">Origins better (of 13)</th></tr></thead><tbody>{{mi_rows}}</tbody></table></div>
    <p class="small">WAPE ÷ R3 on recent origins. Known futures use the realised values (an exact plan), read from the sales, so they are an upper bound on a real plan. The store count off by about two stores shows how fast that bound shrinks.</p>
    <p style="margin-top:10px"><b>Known on the forecast date.</b> Only the 11 origin-known signals help at every origin ({{mi_sig_c}} against {{mi_base_c}} with corrections). They are stock gaps, size run, size concentration, newest colour, partner share and count, share of the style, live colours, web discount, discount against the site's, and the largest day's share. They are known for June 2026 too, so they can enter the 2026 forecast; Main will ask you whether to build a workbook with them. The last-month inputs help less, and the trend inputs and the best-day inputs do not help.</p>
    <p><b>Business decisions as known futures.</b> Together they are worth the most: {{mi_dec_c}} with corrections and {{mi_dec_h}} within ±20%, better at all 13 origins; {{mi_decsig_c}} with the 11 signals as well. Stores carry most of it; the discount plan and the launch calendar add a little each. Adding availability in the window (size run, stock gaps), which is partly a result of sales, does not add more ({{mi_av_c}}). Using them for July–October 2026 needs Edikted's plan for each of the 100 targets: own stores, partners, discount and launches.</p>
    <p><b>Trends.</b> The ten fastest-rising products at an origin later sold {{tr_hot_g}} of R3 and the ten fastest-falling {{tr_cold_g}}: hot items cool off and falling ones recover, and the models already expect it (bias {{tr_hot_b}} and {{tr_cold_b}} on them). Closeness to either set in the co-purchase embedding carries nothing further (correlation with the miss within ±0.04). At the June 2026 origin the rising set is summer tops and the falling set is knits, sweaters and hoodies: the season turning, which the models already carry.</p>
    <div class="grid2">
      <div><div class="tscroll"><table id="t-hot26"><thead><tr><th>Rising fastest at June 2026</th><th>Type</th><th class="r">Last 3 months ÷ the 3 before</th></tr></thead><tbody>{{tr_hot_rows}}</tbody></table></div></div>
      <div><div class="tscroll"><table id="t-cold26"><thead><tr><th>Falling fastest at June 2026</th><th>Type</th><th class="r">Last 3 months ÷ the 3 before</th></tr></thead><tbody>{{tr_cold_rows}}</tbody></table></div></div>
    </div></div>
'''
rep("  <div class=\"note\"><b>This corrects an earlier ranking.</b>", PANEL + "  <div class=\"note\"><b>This corrects an earlier ranking.</b>")
rep('<p class="small">The model sees one population and forecasts between the two. Stores means own stores with at least one sale of the item in a month.</p></div>',
    '<p class="small">The model sees one population and forecasts between the two. Stores means own stores with at least one sale of the item in a month. With the business decisions given to the tree model as known futures (see <a href="#member-inputs">Information</a>), launches that gained five or more stores go from a bias of {{la_b5_0}} to {{la_b5_1}} and from {{la_h5_0}} to {{la_h5_1}} within ±20%; all launches from {{la_w_0}} to {{la_w_1}} of R3\'s error.</p></div>')
rep("#t-lm td:first-child{min-width:250px}", "#t-lm td:first-child,#t-mi td:first-child{min-width:250px} #t-mi tr.grp td{font-weight:600;padding-top:10px}")
open(p, "w", encoding="utf-8").write(s)

b = "eda/198g_build_error_study.py"; t = open(b, encoding="utf-8").read()
y = 'T = dict(\n'; assert y in t
pre = '''# member refits D307-D312: table rows and tokens
_rf = {d["variant"]: d for d in J["refit"]["rows"]}; _rfu = {d["variant"]: d for d in J["refit_future"]["rows"]}; _rd = {d["variant"]: d for d in J["refit_decisions"]["rows"]}; _tr = {d["variant"]: d for d in J["trend"]["member"]}; _md = {d["variant"]: d for d in J["maxday"]["member"]}
def _mirow(lab, d, base=False):
    w = "" if base else str(d.get("wins13", ""))
    return f'<tr{" class=\\'on\\'" if base else ""}><td>{lab}</td><td class="r">{f3(d["alone"])}</td><td class="r">{f3(d["stack"])}</td><td class="r">{f3(d["corrected"])}</td><td class="r">{pc(d["corrected_hit"], 1)}</td><td class="r">{w}</td></tr>'
_grp = lambda lab: f'<tr class="grp"><td colspan="6">{lab}</td></tr>'
MIROWS = _mirow("none (the submitted tree model)", _rf["ec_ns (refit, check)"], True) + _grp("Known on the forecast date") + _mirow("11 origin-known signals", _rf["+ origin-known signals"]) + _mirow("20 inputs from the last month", _rf["+ last-month inputs"]) \\
    + _mirow("cluster and product-type trends", _tr["+ cluster and type trends"]) + _mirow("closeness to the 10 rising and 10 falling products", _tr["+ closeness to the rising and falling products"]) + _mirow("best sales day in recent history", _md["+ best-day inputs"]) \\
    + _grp("Known futures (business decisions, exact plan)") + _mirow("own stores added and reached", _rd["+ stores added and reached (known future)"]) + _mirow("discount plan", _rd["+ the discount plan (known future)"]) \\
    + _mirow("new colours of the style and same-type launches", _rfu["+ new colours of the style and same-type launches in the window (known future)"]) + _mirow("every business decision: stores, partners, launches, discount", _rd["+ every business decision: stores, partners, launches, discount (known future)"]) \\
    + _mirow("every business decision and the 11 signals", _rd["+ every business decision and the origin-known signals"]) + _mirow("own stores, with the count off by about 2 stores", _rfu["+ the store counts, off by about 2 stores at the forecast date"])
for _k in ["stack", "corrected", "corrected_hit"]: assert all(_k in d for d in [_rf["+ origin-known signals"], _md["+ best-day inputs"]])
_s26 = J["trend"]["sets26"]; _hr = lambda L: "".join(f'<tr><td>{d["name"]}<br><span class="small">{d["item"]}</span></td><td>{d["ptype"]}</td><td class="r">{math.exp(d["trend"]):.1f}×</td></tr>' for d in L)
_la = {(d["variant"][:30], d["g"]): d for d in J["refit_decisions"]["launch"]}; _lv = "+ every business decision: stor"
'''
t = t.replace(y, pre + y, 1)
x = '    built="17 Nov 2026",'; assert x in t
add = '''    mi_rows=MIROWS, mi_base_c=f3(_rf["ec_ns (refit, check)"]["corrected"]), mi_sig_c=f3(_rf["+ origin-known signals"]["corrected"]), mi_dec_c=f3(_rd["+ every business decision: stores, partners, launches, discount (known future)"]["corrected"]),
    mi_dec_h=pc(_rd["+ every business decision: stores, partners, launches, discount (known future)"]["corrected_hit"], 1), mi_decsig_c=f3(_rd["+ every business decision and the origin-known signals"]["corrected"]),
    mi_av_c=f3(_rd["+ decisions and availability in the window (partly a result of sales: a bound)"]["corrected"]),
    tr_hot_g=pc(J["trend"]["flagged"]["hot_growth"]), tr_cold_g=pc(J["trend"]["flagged"]["cold_growth"]), tr_hot_b=sp(J["trend"]["flagged"]["hot_bias"]), tr_cold_b=sp(J["trend"]["flagged"]["cold_bias"]), tr_hot_rows=_hr(_s26["hot"]), tr_cold_rows=_hr(_s26["cold"]),
    la_b5_0=sp(_la[(_lv, "stores added: 5 or more")]["bias_base"]), la_b5_1=sp(_la[(_lv, "stores added: 5 or more")]["bias"]), la_h5_0=pc(_la[(_lv, "stores added: 5 or more")]["hit_base"]), la_h5_1=pc(_la[(_lv, "stores added: 5 or more")]["hit"]),
    la_w_0=f3(_la[(_lv, "all launches under 3 months")]["wape_base"]), la_w_1=f3(_la[(_lv, "all launches under 3 months")]["wape"]),'''
t = t.replace(x, x + "\n" + add); open(b, "w", encoding="utf-8").write(t); print("patched")
