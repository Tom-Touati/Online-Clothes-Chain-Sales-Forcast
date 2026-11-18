"""D278 (fork): one-off patch that adds the +/-20% hit-rate section to eda/error_study_template.html and its tokens to
eda/198g_build_error_study.py. Idempotent: does nothing if the section is already there."""
p = "eda/error_study_template.html"; s = open(p, encoding="utf-8").read()
if 'id="hit"' in s: raise SystemExit("already patched")


def rep(a, b):
    global s
    assert a in s, a[:80]; s = s.replace(a, b, 1)


rep('<a href="#summary">Summary</a><a href="#where">Where</a>', '<a href="#summary">Summary</a><a href="#hit">±20%</a><a href="#where">Where</a>')
rep('<div class="tile"><b>{{top5}}</b><span>of the error sits in 5% of the forecasts</span></div>', '<div class="tile"><b>{{hit_items}}</b><span>of item totals within ±20% of actual (R3: {{hit_r3}})</span></div>')
rep('''    <li><b>Absolute error is steady; the ratio to R3 is not.</b>''', '''    <li><b>{{hit_items}} of item totals land within ±20%, and the realistic item-level ceiling is about half.</b> The median forecast is off by {{med_ape}}, so the ±20% line cuts through the middle of the distribution; {{within30}} are within ±30%. No cohort or simple rule reaches 60% in both halves of the backtest; the most confident tenth of items reaches {{clf_top10}}. Knowing the plans and one month of sales would lift the rate to {{hit_plan_early}}. Summed to look or product-type level the same forecasts are within ±20% {{lvl_look}} and {{lvl_type}} of the time.</li>
    <li><b>Absolute error is steady; the ratio to R3 is not.</b>''')
rep('<b>The error is about which items, not how much in total.</b> With every origin', '<b>The error is about which items, not how much in total.</b> {{top5}} of it sits in 5% of the forecasts. With every origin')
SEC = '''<section id="hit">
  <div class="eyebrow">±20% hit rate</div>
  <h2>Four in ten item totals land within ±20%. Why, and how far that can go</h2>
  <p>{{hit_items}} of recent item totals are forecast within ±20% of what sold: {{hit_units}} when weighted by units and {{hit_targets}} for the 100 targets. R3 manages {{hit_r3}}. The median forecast is off by {{med_ape}}, so the ±20% line cuts through the middle of the distribution: {{within25}} are within ±25% and {{within30}} within ±30%.</p>
  <div class="grid2">
    <div class="panel"><h3>Share of item totals within a tolerance</h3>
      <div class="legend"><span><i style="background:var(--accent)"></i>Submitted system</span><span><i style="background:var(--s2)"></i>Seasonal naive</span><span><i style="background:var(--ref)"></i>R3</span></div>
      <div id="c-tol"></div>
      <p class="small">Recent origins. The tolerance is the allowed miss as a share of the actual.</p></div>
    <div class="panel"><h3>What would move it</h3><div id="c-ladder"></div>
      <p class="small">Hit rate of the same forecasts if the named information had been known. The vertical line marks 60%. "Best of the five models" picks the closest model for each item afterwards; it is a bound, not an option.</p></div>
  </div>
  <h3 style="margin-top:18px">The misses, by size</h3>
  <div class="tscroll"><table id="t-bands"><thead><tr><th>Forecast is off by</th><th>Share of forecasts</th><th class="r">Share of units</th><th class="r">Under-forecast</th><th class="r">Median level / month</th><th class="r">Launch under 3 months</th><th class="r">Window with November</th><th class="r">All five models on one side</th><th class="r">Actual inside P10–P90</th></tr></thead><tbody></tbody></table></div>
  <p style="margin-top:12px">Of the forecasts outside ±20%, {{near_share}} are near misses (20–30% off), {{mid_share}} are 30–50% off and {{large_share}} are off by more than half. The bands differ by degree, not by kind. From hits to large misses the items get smaller, the stated range wider and the window more likely to include November. The clearest separator is agreement: all five models are on the same side of the actual in {{same_hit}} of hits, {{same_near}} of near misses and {{same_large}} of 50–100% misses. Misses beyond 50% are {{over50_share}} of forecasts but {{over50_units}} of units, and {{over50_over}} of them are over-forecasts, mostly of small items.</p>
  <div class="grid2">
    <div class="panel"><h3>Level or shape?</h3>
      <p>Shape. With every origin's total right the rate is {{hit_origin_total}}; with every product type's total right, {{hit_type}}. Only with the total of every type × look cell known does it reach {{hit_cell}}.</p>
      <p>An unbiased forecast with today's spread of item errors (robust log standard deviation {{sigma}}) would hit {{normal_hit}}. Reaching 50% needs that spread cut by {{cut50}}, 60% by {{cut60}} and 70% by {{cut70}}.</p>
      <p>Counting noise is not the limit: a forecast equal to the true mean would land within ±20% {{pois_all}} of the time, and {{pois_small}} for the smallest fifth, where the system hits {{hit_fifth1}}.</p></div>
    <div class="panel"><h3>The same forecasts, summed to higher levels</h3><div id="c-levels"></div>
      <p class="small">Share of totals within ±20%, recent origins. Most styles have one live colour, so the style level gains nothing.</p></div>
  </div>
  <div class="grid2">
    <div class="panel"><h3>Cohorts with the lowest hit rate</h3><div class="tscroll"><table id="t-hlow"><thead><tr><th>Cohort</th><th class="r">Forecasts</th><th class="r">Share of units</th><th class="r">Within ±20%</th></tr></thead><tbody></tbody></table></div></div>
    <div class="panel"><h3>Cohorts with the highest hit rate</h3><div class="tscroll"><table id="t-hhigh"><thead><tr><th>Cohort</th><th class="r">Forecasts</th><th class="r">Share of units</th><th class="r">Within ±20%</th></tr></thead><tbody></tbody></table></div></div>
  </div>
  <div class="grid2">
    <div class="panel"><h3>Can a confident group be named in advance?</h3><div id="c-dec"></div>
      <p class="small">A classifier trained only on earlier origins ranks each origin's forecasts by predicted chance of a hit. Bars show the actual hit rate of each tenth, least to most confident. AUC {{clf_auc}}.</p></div>
    <div class="panel"><h3>Simple rules that hold in both halves of the backtest</h3>
      <div class="tscroll"><table id="t-rules"><thead><tr><th>Rule</th><th class="r">First half</th><th class="r">Second half</th><th class="r">Share of forecasts</th><th class="r">Share of units</th></tr></thead><tbody></tbody></table></div>
      <p class="small">{{rules_n}} combinations of up to three conditions, each covering at least 3% of forecasts in both halves. {{rules_50}} reach 50% in both halves; none reaches 55%.</p></div>
  </div>
  <p style="margin-top:14px">A defensible top confidence level exists at about 55–60%, not higher. The most confident tenth hit {{clf_top10}} overall and {{clf_late_lo}}–{{clf_late_hi}} at the last four origins, once the classifier had a year of outcomes to learn from; the least confident fifth hit {{clf_bottom20}}. The width of the forecast range carries most of the signal. For the June 2026 targets the same classifier expects {{t26_mean}} to land within ±20%: {{t26_ge50}} targets have a predicted chance of at least 50% and {{t26_ge60}} of at least 60%. It agrees with the workbook's confidence column, whose levels 1, 2 and 3 average {{t26_c1}}, {{t26_c2}} and {{t26_c3}}.</p>
  <div class="note"><b>Realistic ceiling.</b> At item level on four-month totals: {{hit_items}} today, {{hit_plan}} with the roll-out, promotion, stock and launch plans known, and {{hit_plan_early}} with one month of sales as well. 60% across all items would need the typical item error cut by about a third, which none of the information tested delivers. 60% is available today in three places: at look or product-type level ({{lvl_look}} and {{lvl_type}}; {{lvl_group}} by product group), for the most confident tenth of items, and at a tolerance of ±30% ({{within30}}).</div>
</section>

'''
rep('<section id="where">', SEC + '<section id="where">')
rep('''  <h2 class="grp">Change how the system is judged</h2>
  <div class="props">''', '''  <h2 class="grp">Change how the system is judged</h2>
  <div class="props">
    <div class="prop"><h3>12. Use ±20% where it can be met</h3><dl>
      <dt>Evidence</dt><dd>{{hit_items}} at item level, {{lvl_look}} by look, {{lvl_type}} by product type, {{lvl_group}} by product group. No information tested lifts the item level past {{hit_plan_early}}.</dd>
      <dt>Action</dt><dd>Commit fabric and capacity on look or type totals, where ±20% holds most of the time, and split to items late. State an expected hit rate with each confidence level (about 55–60%, 43% and 35%) and define level 1 by the validated rule: a narrow range, a rising trend and a year of history. Report the unit-weighted rate ({{hit_units}}) beside the item count.</dd>
      <dt>Worth</dt><dd>The decisions that need ±20% get it, and the item-level number is read for what it is: the volatility of single items, not a fault to tune away.</dd></dl></div>''')
rep('<div class="prop"><h3>12. Report three numbers, not one</h3>', '<div class="prop"><h3>13. Report three numbers, not one</h3>')
rep('<div class="prop"><h3>13. Freeze, write down the expectation, and stop tuning on the same nine origins</h3>', '<div class="prop"><h3>14. Freeze, write down the expectation, and stop tuning on the same nine origins</h3>')
rep("    if (r.v2 != null) { add(r.v2, '--ref', 0, 5);", "    if (o.mark != null) t.append(h('i', { style: `left:${pos(o.mark)}%;width:1.5px;background:var(--fg);top:-3px;height:18px;opacity:.45;border-radius:0` }));\n    if (r.v2 != null) { add(r.v2, '--ref', 0, 5);")
JS = r'''
// ---- +/-20% hit rate
{ const HT = J.hit, C = HT.curve, L = HT.level_shape, I = HT.info;
  lineChart($('#c-tol'), C.map(d => '±' + Math.round(d.tol * 100) + '%'), [{ c: '--ref', v: C.map(d => d.r3), end: 'R3' }, { c: '--s2', v: C.map(d => d.seasonal), end: 'seasonal' }, { c: '--accent', v: C.map(d => d.system), end: 'system' }], { max: 1, ticks: [0, .25, .5, .75, 1], fmt: v => pct(v), dots: true, r: 58, markX: 3, label: 'Share of item totals within a tolerance', tip: i => `<b>Within ±${Math.round(C[i].tol * 100)}%</b><br>system ${pct(C[i].system, 1)} (${pct(C[i].units, 1)} of units)<br>seasonal naive ${pct(C[i].seasonal, 1)}<br>R3 ${pct(C[i].r3, 1)}` });
  const lad = [['Today', I.system, 'the submitted system'], ["Each origin's total known", L.origin_total, 'level only'], ["Each product type's total known", L.type_total, ''], ['Roll-out, promotion, stock and launch plans known', I.plan, 'allocation re-fitted'], ['Plans + one month of sales', I.plan_early, 're-forecast after month 1'], ['Each type × look total known', L.cell_total, 'not forecastable (D240)'], ['Best of the five models per item', L.best_member, 'picked afterwards']];
  hbars($('#c-ladder'), lad.map(([a, v, sub]) => ({ label: a, sub, v, text: pct(v), tip: `<b>${a}</b>: ${pct(v, 1)} within ±20%` })), { min: 0, max: 0.8, mark: 0.6 });
  const tb = $('#t-bands tbody'); HT.bands.forEach(b => tb.append(h('tr', {}, h('td', { text: b.band }), h('td', {}, h('div', { style: 'display:flex;gap:8px;align-items:center;min-width:150px' }, h('div', { class: 'pair', style: 'flex:1' }, h('div', { class: 'e', style: `width:${b.share / 0.45 * 100}%;height:9px` })), h('span', { text: pct(b.share) }))),
    h('td', { class: 'r', text: pct(b.units) }), h('td', { class: 'r', text: pct(b.under) }), h('td', { class: 'r', text: int(b.med_level) }), h('td', { class: 'r', text: pct(b.young) }), h('td', { class: 'r', text: pct(b.bf) }), h('td', { class: 'r', text: pct(b.all_same) }), h('td', { class: 'r', text: pct(b.inside) }))));
  hbars($('#c-levels'), HT.levels.map(d => ({ label: d.level, sub: `${int(d.n)} totals, median ${int(d.med_units)} units`, v: d.hit, text: pct(d.hit), tip: `<b>${d.level}</b><br>${pct(d.hit, 1)} within ±20% (R3: ${pct(d.r3, 1)})<br>${pct(d.within10, 1)} within ±10%` })), { min: 0, max: 1, mark: 0.6 });
  const coh = (id, L2) => { const t = $(id + ' tbody'); L2.slice(0, 8).forEach(d => t.append(h('tr', {}, h('td', { html: `${d.g}<br><span class="small">${d.axis}</span>` }), h('td', { class: 'r', text: int(d.n) }), h('td', { class: 'r', text: pct(d.units) }), h('td', { class: 'r', text: pct(d.hit) })))); };
  coh('#t-hlow', HT.cohort_low); coh('#t-hhigh', HT.cohort_high);
  groupedBars($('#c-dec'), HT.clf.deciles.map(d => ({ label: String(d.decile), v: [d.hit], tip: `<b>Tenth ${d.decile}</b> by predicted chance<br>predicted ${pct(d.p)}, actual ${pct(d.hit, 1)} within ±20%` })), [{ c: '--accent' }], { max: 0.7, ticks: [0, .2, .4, .6], fmt: v => pct(v), h: 220, label: 'Actual hit rate by tenth of predicted chance' });
  const tr = $('#t-rules tbody'); HT.rules.filter((d, i, a) => a.findIndex(x => x.first_half === d.first_half && x.second_half === d.second_half) === i).slice(0, 6).forEach(d => tr.append(h('tr', {}, h('td', { text: d.rule }), h('td', { class: 'r', text: pct(d.first_half) }), h('td', { class: 'r', text: pct(d.second_half) }), h('td', { class: 'r', text: pct(d.cover) }), h('td', { class: 'r', text: pct(d.units) })))); }
'''
rep("\n// ---- where\n", JS + "\n// ---- where\n")
open(p, "w", encoding="utf-8").write(s)

b = "eda/198g_build_error_study.py"; t = open(b, encoding="utf-8").read()
x = '    built="17 Nov 2026",'; assert x in t
add = '''    hit_items=pc(HT["overall"]["items"]), hit_units=pc(HT["overall"]["units"]), hit_targets=pc(HT["overall"]["targets"]), hit_r3=pc(H["recent"]["hit_r3"]), med_ape=pc(LS["median_ape"], 1),
    within25=pc(HT["miss_split"]["within25"]), within30=pc(HT["miss_split"]["within30"]), near_share=pc(HT["miss_split"]["near"]), mid_share=pc(HT["miss_split"]["mid"]), large_share=pc(HT["miss_split"]["large"]),
    same_hit=pc(BD["within 20%"]["all_same"]), same_near=pc(BD["20-30% (near miss)"]["all_same"]), same_large=pc(BD["50-100%"]["all_same"]),
    over50_share=pc(BD["50-100%"]["share"] + BD["beyond 100%"]["share"]), over50_units=pc(BD["50-100%"]["units"] + BD["beyond 100%"]["units"]),
    over50_over=pc((BD["50-100%"]["share"] * (1 - BD["50-100%"]["under"]) + BD["beyond 100%"]["share"] * (1 - BD["beyond 100%"]["under"])) / (BD["50-100%"]["share"] + BD["beyond 100%"]["share"])),
    hit_origin_total=pc(LS["origin_total"]), hit_type=pc(LS["type_total"]), hit_cell=pc(LS["cell_total"]), sigma=f2(LS["robust_sigma"]), normal_hit=pc(LS["normal_hit_at_sigma"]),
    cut50=pc(1 - LS["sigma_for_50"] / LS["robust_sigma"]), cut60=pc(1 - LS["sigma_for_60"] / LS["robust_sigma"]), cut70=pc(1 - LS["sigma_for_70"] / LS["robust_sigma"]),
    pois_all=pc(HT["floor"]["overall"]), pois_small=pc(HT["floor"]["by_fifth"][0]["poisson"]), hit_fifth1=pc(HT["floor"]["by_fifth"][0]["hit"]),
    clf_auc=f2(HT["clf"]["auc"]), rules_n=str(HT["rules_n"]), rules_50=str(HT["rules_over50"]), clf_top10=pc(HT["clf"]["top10"]), clf_bottom20=pc(HT["clf"]["bottom20"]),
    clf_late_lo=pc(min(d["hit"] for d in HT["clf"]["top10_by_origin"][-4:]))[:-1], clf_late_hi=pc(max(d["hit"] for d in HT["clf"]["top10_by_origin"][-4:])),
    t26_mean=pc(HT["targets26"]["mean_p"]), t26_ge50=str(HT["targets26"]["ge50"]), t26_ge60=str(HT["targets26"]["ge60"]),
    t26_c1=pc(HT["targets26"]["by_conf"][0]["p"]), t26_c2=pc(HT["targets26"]["by_conf"][1]["p"]), t26_c3=pc(HT["targets26"]["by_conf"][2]["p"]),
    hit_plan=pc(HT["info"]["plan"]), hit_plan_early=pc(HT["info"]["plan_early"]), lvl_look=pc(LV["look (cart + content cluster)"]["hit"]), lvl_type=pc(LV["product type"]["hit"]), lvl_group=pc(LV["product group"]["hit"]),'''
t = t.replace(x, x + "\n" + add)
y = 'c26 = pd.read_csv("data/final_2026_nx2_cascade.csv", index_col=0)'; assert y in t
t = t.replace(y, 'HT = J["hit"]; LS = HT["level_shape"]; BD = {d["band"]: d for d in HT["bands"]}; LV = {d["level"]: d for d in HT["levels"]}\n' + y)
open(b, "w", encoding="utf-8").write(t)
print("patched")
