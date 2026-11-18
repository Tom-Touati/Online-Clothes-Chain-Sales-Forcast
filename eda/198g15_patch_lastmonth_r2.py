"""D304, D305 (fork): one-off patch after the user's requests in the session ("focus also on adding more recent features from the last
month. also lets have an R^2 metric"): an R-squared table in 'Scored in units and per item' and a panel 'More inputs from the last
month' at the end of the Where section (both from eda/198s). Idempotent."""
p = "eda/error_study_template.html"; s = open(p, encoding="utf-8").read()
if 'id="t-r2"' in s: raise SystemExit("already patched")


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80]); s = s.replace(a, b, 1)


R2 = '''    <h3 style="margin-top:18px">R²: how much of the variation the forecast explains</h3>
    <p>Asked in the session. R² is 1 when the forecast explains all the variation between items and 0 when it does no better than giving every item the same average. It is shown three ways, because the answer depends on what is to be explained.</p>
    <div class="tscroll"><table id="t-r2"><thead><tr><th>Recent origins</th><th class="r">R² in units</th><th class="r">R² in logs</th><th class="r">R² of the change against R3</th><th class="r">The same, within an origin</th></tr></thead><tbody></tbody></table></div>
    <p class="small">In units, large items decide it. In logs every item counts once, and R3 already scores {{r2_r3_log}} because a big item stays big. The change against R3 is the part a forecast has to add: there R3 explains nothing by construction, the rule agents explain {{r2_rules_chg}} and the system {{r2_sys_chg}}. By origin the system's R² of the change runs from {{r2_o_lo}} to {{r2_o_hi}} on recent origins. It is lowest at the June and December 2025 origins, where items moved least against their recent level and there was least to explain.</p>
'''
rep('    <h3 style="margin-top:18px">Should the models be trained on percentages?</h3>', R2 + '    <h3 style="margin-top:18px">Should the models be trained on percentages?</h3>')
LM = '''  <div class="panel" id="last-month" style="margin-top:18px"><h3>More inputs from the last month</h3>
    <p>Asked in the session. {{lm_n}} inputs were built from the daily sales of the 28 days before each origin and the 28 days before those: the pace within the month (the last 7 and 14 days, the four weekly totals, the month against the previous 28 days and against the level), its rhythm (days with a sale, days since the last sale and since the best day, the best day over the average day, weekend share), units per order, the cancelled share, and the discount and the TikTok, own-store and partner shares with their change against the previous 28 days.</p>
    <div class="tscroll"><table id="t-lm"><thead><tr><th>Correction trained on earlier origins, with</th><th class="r">Inputs</th><th class="r">WAPE ÷ R3</th><th class="r">Within ±20%</th><th class="r">Change against the reference, 95% interval</th></tr></thead><tbody></tbody></table></div>
    <p style="margin-top:10px">No measurable gain: {{lm_w_all}} with all of them against {{lm_w_ref}} without, and the all-too-low flag stays at AUC {{lm_auc}}. The strongest single inputs lean the right way: days since the last sale ({{lm_rho_ds}}, same sign at {{lm_same_ds}} of 9 origins), the partner share of the last 28 days ({{lm_rho_par}}) and the last 7 days against the month's pace ({{lm_rho_r7}}). For comparison, the 28-day rate and the 91-day trend that the models already use have {{lm_rho_r28}} and {{lm_rho_tr}}. Together the new inputs still do not improve the correction, because they overlap with what the models have: the models are fitted on sales up to the last day before the origin, so the last month is already their starting point.</p></div>
'''
rep('</section>\n\n<section id="when">', LM + '</section>\n\n<section id="when">')
JS = r'''
// ---- R-squared; last-month inputs
{ const tr2 = $('#t-r2 tbody'); J.r2.systems.forEach((d, i, a) => tr2.append(h('tr', i === a.length - 1 ? { class: 'on' } : { 'data-tip': '<b>' + d.system + '</b><br>all 13 origins: ' + fx(d.all13.units, 2) + ' in units, ' + fx(d.all13.log, 2) + ' in logs, ' + fx(d.all13.change, 2) + ' of the change' }, h('td', { text: d.system }), h('td', { class: 'r', text: fx(d.units, 2) }), h('td', { class: 'r', text: fx(d.log, 2) }), h('td', { class: 'r', text: fx(d.change, 2) }), h('td', { class: 'r', text: fx(d.change_within, 2) }))));
  const tlm = $('#t-lm tbody'); J.lastmonth.rows.forEach((d, i) => tlm.append(h('tr', i === 1 ? { class: 'on' } : {}, h('td', { text: d.variant }), h('td', { class: 'r', text: d.n ? String(d.n) : '' }), h('td', { class: 'r', text: fx(d.wape, 4) }), h('td', { class: 'r', text: pct(d.hit, 1) }), h('td', { class: 'r', text: d.ci ? sf(d.ci[0]) + ' to ' + sf(d.ci[1]) : '' })))); }
'''
rep("\n// ---- when\n", JS + "\n// ---- when\n")
rep("#t-surprise td:first-child,#t-meta td:first-child{min-width:250px}", "#t-surprise td:first-child,#t-meta td:first-child,#t-lm td:first-child{min-width:250px} #t-r2 td:first-child{white-space:nowrap}")
open(p, "w", encoding="utf-8").write(s)

b = "eda/198g_build_error_study.py"; t = open(b, encoding="utf-8").read()
x = '    built="17 Nov 2026",'; assert x in t
add = '''    r2_r3_log=f2(_r2["R3"]["log"]), r2_rules_chg=f2(_r2["R3 + rules"]["change"]), r2_sys_chg=f2(_r2["Submitted system"]["change"]), r2_o_lo=f2(min(d["change"] for d in _r2o)), r2_o_hi=f2(max(d["change"] for d in _r2o)),
    lm_n=str(LMO["n_inputs"]), lm_w_all=f"{LMO['rows'][5]['wape']:.4f}", lm_w_ref=f"{LMO['rows'][1]['wape']:.4f}", lm_auc=f2(LMO["flag_lm"]["auc"]), lm_rho_ds=sf(_lms["lm_days_since"]["rho"], 2), lm_same_ds=str(_lms["lm_days_since"]["same"]),
    lm_rho_par=sf(_lms["lm_sh_par"]["rho"], 2), lm_rho_r7=sf(_lms["lm_r7"]["rho"], 2), lm_rho_r28=sf(_lms["f_r28"]["rho"], 2), lm_rho_tr=sf(_lms["trend91"]["rho"], 2),'''
t = t.replace(x, x + "\n" + add)
y = 'T = dict(\n'; assert y in t
t = t.replace(y, '_r2 = {d["system"]: d for d in J["r2"]["systems"]}; _r2o = [d for d in J["r2"]["by_origin"] if d["origin"] >= "2025-06"]; LMO = J["lastmonth"]; _lms = {d["key"]: d for d in LMO["sig"]}\nassert LMO["rows"][5]["variant"] == "+ all last-month inputs" and sorted(_r2o, key=lambda d: d["change"])[0]["origin"] == "2025-06" and sorted(_r2o, key=lambda d: d["change"])[1]["origin"] == "2025-12"\n' + y, 1)
open(b, "w", encoding="utf-8").write(t); print("patched")
