"""D289e-D289g (fork): one-off patch after three comments on the error study:
  "what is the reason it scored so poorly on june"                          -> note 'Why 0.69 at that origin' in the When section
  "how do we identify the flag"                                             -> 'How the flag is identified' + the flag on the 2026 targets, in the all-too-low panel
  "let's add the entropy of the specific item, and info on past peaks"      -> 'Entropy and past peaks as inputs', same panel
All from eda/198r. Idempotent."""
p = "eda/error_study_template.html"; s = open(p, encoding="utf-8").read()
if 'id="t-flagimp"' in s: raise SystemExit("already patched")


def rep(a, b):
    global s
    assert a in s, a[:80]; s = s.replace(a, b, 1)


a = "In that season the five models score 0.77–0.90 alone and the stack 0.74.</div>"
rep(a, a + '''
  <div class="note" id="june-why"><b>Why {{an_ratio}} at that origin.</b> The yardstick was strong; the system was not weak. The system's own error there was {{jn_sys}} of units sold, inside the {{jn_sys_lo}}–{{jn_sys_hi}} of the other recent origins. R3's error was {{jn_r3}}, its lowest ({{jn_r3_lo}} to {{jn_r3_hi}} elsewhere), and the score is the ratio of the two. July–October followed the April–June pattern closely: with both forecasts put on the true total, R3's split between items was off by {{jn_split}}, against a median of {{jn_split_med}} at the other origins, where Black Friday, the holidays and the turn of season reshuffle items. The system was closer than R3 for {{jn_closer}} of items, against {{jn_closer_lo}}–{{jn_closer_hi}} elsewhere. The under-forecast came from growth the origin could not show: the total rose {{jn_growth}} over the recent level, {{jn_st_err}} of the error sits on items that gained two or more own stores (bias {{jn_st_bias}}) and {{jn_m1_err}} on items whose first month already ran 20% above pace (bias {{jn_m1_bias}}). The models also had less history: {{jn_hit}} of items landed within ±20% there, against {{jn_hit_lo}}–{{jn_hit_hi}} at the last four origins.</div>''')
FLAG = '''    <p><b>How the flag is identified.</b> At each origin a classifier is trained on earlier forecasts whose results are known. Its target is yes or no: did all five models end below what sold. Its {{fi_n}} inputs are all known on the forecast date. It gives each new forecast a risk between 0 and 1, and the flag is the fifth of forecasts with the highest risk at that origin. {{fi_top}} of that fifth turn out all-too-low, against {{fi_low}} of the lowest fifth.</p>
    <div class="grid2">
      <div><div class="tscroll"><table id="t-flagimp"><thead><tr><th>Input</th><th class="r">Share of the signal</th><th class="r">All-too-low in its lowest fifth</th><th class="r">In its highest fifth</th></tr></thead><tbody></tbody></table></div>
        <p class="small">The five inputs that carry most of the signal. Items near full price and items already forecast well above R3 are the ones that still fall short. Close agreement between the five raises the rate partly by construction: five forecasts that sit together fall on the same side more often.</p></div>
      <div><div class="tscroll"><table id="t-flag26"><thead><tr><th>2026 target</th><th>Type</th><th class="r">Forecast, 4 months</th><th class="r">Risk</th></tr></thead><tbody></tbody></table></div>
        <p class="small">The flag at the June 2026 origin, trained on all 13 origins without the stated range (not stored for 2026; backtest AUC {{fi_auc_nr}}). {{fi26_n}} of the {{fi26_all}} modelled targets fall in the highest-risk fifth of all items and hold {{fi26_units}} of the target forecast; {{fi26_fy}} of the {{fi26_young}} targets under six months old are among them. Shown: the twelve with the highest risk.</p></div>
    </div>
    <p><b>Entropy and past peaks as inputs.</b> Asked in a comment on this page. Entropy measures how evenly an item's sales are spread: over the last 91 days, over its last 12 months and over channels, plus the share sold on its best seven days. Past peaks: months since the best month, the best month of the last year over the level, months above 1.5× the item's median, whether last year's peak month falls in the forecast window, and last year's same window over the level. Entropy tells how large the miss will be, not which way: evenness over 91 days correlates {{en_size}} with the size of the miss and {{en_dir}} with its direction, and the stated range already carries the size. In the correction: {{en_w_ent}} with entropy, {{en_w_peak}} with past peaks and {{en_w_both}} with both, against {{en_w_ref}} without. The flag stays at AUC {{en_auc}}. A peak block inside the tree model gave the same result earlier (D253).</p>
'''
rep("    <p><b>Model surprise from the months before.</b>", FLAG + "    <p><b>Model surprise from the months before.</b>")
JS = r'''
// ---- the flag: inputs and the 2026 targets
{ const FI = J.flaginfo, ti = $('#t-flagimp tbody'); FI.importance.slice(0, 5).forEach(d => ti.append(h('tr', {}, h('td', { text: d.label }), h('td', { class: 'r', text: pct(d.gain) }), h('td', { class: 'r', text: pct(d.low) }), h('td', { class: 'r', text: pct(d.high) }))));
  const t26 = $('#t-flag26 tbody'); FI.t26.top.forEach(d => t26.append(h('tr', {}, h('td', { html: d.name + '<br><span class="small">' + d.item + '</span>' }), h('td', { text: d.ptype }), h('td', { class: 'r', text: int(d.F) }), h('td', { class: 'r', text: pct(d.p) })))); }
'''
rep("\n// ---- when\n", JS + "\n// ---- when\n")
open(p, "w", encoding="utf-8").write(s)

b = "eda/198g_build_error_study.py"; t = open(b, encoding="utf-8").read()
x = '    built="17 Nov 2026",'; assert x in t
add = '''    jn_sys=pc(JN["june"]["sys_abs"], 1), jn_sys_lo=pc(JN["others"]["sys_abs"][0], 1), jn_sys_hi=pc(JN["others"]["sys_abs"][1], 1), jn_r3=pc(JN["june"]["r3_abs"], 1), jn_r3_lo=pc(JN["others"]["r3_abs"][0], 1), jn_r3_hi=pc(JN["others"]["r3_abs"][1]),
    jn_split=pc(JN["june"]["r3_split"]), jn_split_med=pc(JN["others"]["r3_split"][2]), jn_closer=pc(JN["june"]["closer"]), jn_closer_lo=pc(JN["others"]["closer"][0]), jn_closer_hi=pc(JN["others"]["closer"][1]), jn_growth=pc(JN["june"]["growth"] - 1),
    jn_st_err=pc(JN["cohorts"][0]["err"]), jn_st_bias=sp(JN["cohorts"][0]["bias"]), jn_m1_err=pc(JN["cohorts"][2]["err"]), jn_m1_bias=sp(JN["cohorts"][2]["bias"]), jn_hit=pc(JN["june"]["hit"]),
    jn_hit_lo=pc(min(d["hit"] for d in JN["by_origin"][-4:])), jn_hit_hi=pc(max(d["hit"] for d in JN["by_origin"][-4:])),
    fi_n=str(FI_["n_inputs"]), fi_top=pc(FI_["base"]["top20_rate"]), fi_low=pc(FI_["base"]["low20_rate"]), fi_auc_nr=f2(FI_["no_range"]["auc"]), fi26_n=str(FI_["t26"]["flagged"]), fi26_all=str(FI_["t26"]["n"]), fi26_units=pc(FI_["t26"]["flagged_units"]),
    fi26_fy=str(FI_["t26"]["flagged_young"]), fi26_young=str(FI_["t26"]["young"]),
    en_size=sf(_en["ent_d91"]["rho_size"], 2), en_dir=sf(_en["ent_d91"]["rho"], 2), en_w_ent=f"{EN['rows'][2]['wape']:.4f}", en_w_peak=f"{EN['rows'][3]['wape']:.4f}", en_w_both=f"{EN['rows'][4]['wape']:.4f}", en_w_ref=f"{EN['rows'][1]['wape']:.4f}", en_auc=f2(EN["flag_ent"]["auc"]),'''
t = t.replace(x, x + "\n" + add)
y = 'T = dict(\n'; assert y in t
t = t.replace(y, 'JN = J["june"]; FI_ = J["flaginfo"]; EN = J["entropy"]; _en = {d["key"]: d for d in EN["sig"]}; assert JN["june"]["r3_abs"] < JN["others"]["r3_abs"][0] and JN["cohorts"][0]["g"].startswith("gained 2+") and JN["cohorts"][2]["g"].startswith("first month")\n' + y, 1)
open(b, "w", encoding="utf-8").write(t); print("patched")
