"""D280 (fork): one-off patch adding two panels to the 'Where' section of eda/error_study_template.html (user comment on the page):
other ways to combine the five models, and why error concentration and the +/-20% rate are both true. Idempotent."""
p = "eda/error_study_template.html"; s = open(p, encoding="utf-8").read()
if 'id="combine"' in s: raise SystemExit("already patched")


def rep(a, b):
    global s
    assert a in s, a[:80]; s = s.replace(a, b, 1)


PANELS = '''  <div class="grid2" style="margin-top:6px">
    <div class="panel" id="two-counts"><h3>42% of the error in 5% of forecasts, yet only {{hit_items}} within ±20%: both are true</h3>
      <p>They count different things. The error share is measured in units, so large items dominate it. The hit rate counts every item once, so the many ordinary items dominate it.</p>
      <p>The worst 5% of forecasts by units are large items: a median of {{tc_top_level}} units a month against {{tc_all_level}} overall, each off by a median of {{tc_top_err_units}} units against {{tc_all_err_units}}. Forecasting all of them perfectly would cut WAPE ÷ R3 from {{ratio_recent}} to {{tc_wape_fixed}} but lift the hit rate only from {{hit_items}} to {{tc_hit_fixed}}, because they are 5% of the count.</p>
      <div class="tscroll"><table id="t-two"><thead><tr><th>Volume fifth</th><th class="r">Share of forecasts</th><th class="r">Share of misses beyond ±20%</th><th class="r">Share of the error</th><th class="r">Within ±20%</th></tr></thead><tbody></tbody></table></div>
      <p class="small">The three smallest fifths hold {{tc_small_m}} of the misses and {{tc_small_e}} of the error. Fixing the big misses improves WAPE; raising the hit rate means tightening thousands of ordinary forecasts that are each 25–50% off.</p></div>
    <div class="panel" id="combine"><h3>Other ways to combine the five models</h3>
      <p>Tested after a comment on this page: weight the models by their past score, or keep only a dense cluster of their outputs. All time-honest; scored before the correction layer.</p>
      <div class="tscroll"><table id="t-combine"><thead><tr><th>Combination</th><th class="r">WAPE ÷ R3</th><th class="r">Within ±20%</th><th class="r">Bias</th></tr></thead><tbody></tbody></table></div>
      <p class="small">Nothing beats the plain average. Weights by past score end near equal ({{sv_wmin}}–{{sv_wmax}} each at the last origin) because the five models score alike; sharper weights hurt. Clusters hurt because the outlying model is as often the right one. Agreement tells how safe a forecast is ({{sv_hit_tight}} within ±20% in the tightest fifth, {{sv_hit_wide}} in the widest), not which way to move it. Picking the best model per item afterwards would reach {{sv_bound_hit}}, but nothing known at the origin identifies it; learned weights (D175) failed the same way.</p></div>
  </div>
'''
rep('</section>\n\n<section id="when">', PANELS + '</section>\n\n<section id="when">')
JS = r'''
// ---- the two counts, and other ways to combine the models
{ const t2 = $('#t-two tbody'); J.twocounts.by_fifth.forEach(d => t2.append(h('tr', {}, h('td', { text: `fifth ${d.fifth} (median ${int(d.med_level)} / month)` }), h('td', { class: 'r', text: pct(d.forecasts) }), h('td', { class: 'r', text: pct(d.misses) }), h('td', { class: 'r', text: pct(d.err) }), h('td', { class: 'r', text: pct(d.hit) }))));
  const KEEP = ['arithmetic mean (submitted stack)', 'trimmed mean (drop lowest and highest)', 'geometric mean', 'weights by past score, power 1', 'weights by past score, power 2', 'weights by past score within age band, power 2', 'median of the five', 'drop the member farthest from the median', 'upper four (drop the lowest)', 'weights by past score, power 4', 'densest three (smallest spread)', 'upper three', 'best single member so far', 'best member per item, picked afterwards (bound)'];
  const tc = $('#t-combine tbody'); KEEP.forEach((k, i) => { const d = J.stackvar.rows.find(x => x.variant === k); if (!d) return; tc.append(h('tr', i === 0 ? { class: 'on' } : { 'data-tip': `<b>${d.variant}</b><br>difference in WAPE ÷ R3 against the plain average: ${sf(d.ci[0])} to ${sf(d.ci[1])}<br>all 13 origins: ${fx(d.wape_all, 3)}, ${pct(d.hit_all, 1)} within ±20%` }, h('td', { text: d.variant }), h('td', { class: 'r', text: fx(d.wape, 3) }), h('td', { class: 'r', text: pct(d.hit, 1) }), h('td', { class: 'r', text: sp(d.bias) }))); }); }
'''
rep("\n// ---- when\n", JS + "\n// ---- when\n")
rep("#t-rules td:first-child{min-width:250px}", "#t-rules td:first-child,#t-combine td:first-child{min-width:250px} #t-two td:first-child{white-space:nowrap}")
open(p, "w", encoding="utf-8").write(s)

b = "eda/198g_build_error_study.py"; t = open(b, encoding="utf-8").read()
x = '    built="17 Nov 2026",'; assert x in t
add = '''    tc_top_level=it(TC["top5"]["med_level"]), tc_all_level=it(TC["all"]["med_level"]), tc_top_err_units=it(TC["top5"]["med_err_units"]), tc_all_err_units=it(TC["all"]["med_err_units"]),
    tc_wape_fixed=f3(TC["wape_if_top5_fixed"]), tc_hit_fixed=pc(TC["hit_if_top5_fixed"]), tc_small_m=pc(TC["small3"]["misses"]), tc_small_e=pc(TC["small3"]["err"]),
    sv_wmin=f2(min(v for k, v in SV["weights"][-1].items() if k != "origin")), sv_wmax=f2(max(v for k, v in SV["weights"][-1].items() if k != "origin")),
    sv_hit_tight=pc(SV["spread_hit"][0]["hit"]), sv_hit_wide=pc(SV["spread_hit"][-1]["hit"]), sv_bound_hit=pc(next(d for d in SV["rows"] if "bound" in d["variant"])["hit"]),'''
t = t.replace(x, x + "\n" + add)
y = 'c26 = pd.read_csv("data/final_2026_nx2_cascade.csv", index_col=0)'; assert y in t
t = t.replace(y, 'TC = J["twocounts"]; SV = J["stackvar"]\n' + y)
open(b, "w", encoding="utf-8").write(t)
print("patched")
