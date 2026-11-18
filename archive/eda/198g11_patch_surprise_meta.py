"""D289b-D289d (fork): one-off patch after three comments on the error study:
  "explain please" (the roll-out plan item in 'When all five models are too low')      -> that item rewritten in plain words
  "let's add surprise metrics regarding model surprise from the months before that"     -> paragraph and table in the same panel (eda/198q)
  "train a secondary model that picks models ... run pca on the models"                 -> panel 'A second model that picks or re-mixes the five' (eda/198q)
Idempotent."""
p = "eda/error_study_template.html"; s = open(p, encoding="utf-8").read()
if 'id="meta-pick"' in s: raise SystemExit("already patched")


def rep(a, b):
    global s
    assert a in s, a[:80]; s = s.replace(a, b, 1)


rep("<li><b>The roll-out plan.</b> {{al_st2}} of them gained two or more stores, twice the rate of other forecasts. Plans alone cut their error by {{al_cut_plan}}.</li>",
    "<li><b>The roll-out plan.</b> This is the business's list of which stores each item will be sent to. {{al_st2}} of these forecasts are items that reached two or more additional own stores during the window, against {{al_st2_r}} of other forecasts. Their sales rose partly because they were on more shelves. The models could not know that on the forecast date; the business did. If the models are told what the business plans ahead (stores and partners added, discount depth, stock and sizes, new colours), the error on these forecasts is {{al_cut_plan}} smaller. The effect is modest because most of these misses are demand, not distribution.</li>")
SUR = '''    <p><b>Model surprise from the months before.</b> Asked in a comment on this page. The first row of the table is only known afterwards, but the same kind of surprise is known for earlier forecasts: at each origin, the forecasts made one to four origins earlier cover months that have since been sold. {{su_n}} inputs measure how far those months ran above or below the pace of the earlier forecast, also against the highest of the five models and relative to the item's product type.</p>
    <div class="tscroll"><table id="t-surprise"><thead><tr><th>Correction trained on earlier origins, with</th><th class="r">WAPE ÷ R3</th><th class="r">Within ±20%</th><th class="r">Change against the reference, 95% interval</th></tr></thead><tbody></tbody></table></div>
    <p style="margin-top:10px">They add nothing the models lack. The latest month's surprise correlates {{su_rho1}} with the new miss, no more than the 91-day trend the models already use ({{su_rho_tr}}); the correction scores {{su_w}} with them against {{su_ref_w}} without; the flag stays at AUC {{su_auc1}}. The models are re-fitted at every origin on sales up to that day, so last month's surprise is already in the level and trend they start from. What they lack is the first month of the new window, which has not happened yet.</p>
'''
rep("    <p>The stated range also needs widening for them:", SUR + "    <p>The stated range also needs widening for them:")
META = '''  <div class="panel" id="meta-pick" style="margin-top:18px"><h3>A second model that picks or re-mixes the five</h3>
    <p>Asked in a comment on this page: train a second model that chooses between the five from high-level features such as product type, and run PCA on the five to work with their independent parts. Every version is trained on origins whose results were known and scored before the correction layer.</p>
    <div class="tscroll"><table id="t-meta"><thead><tr><th>Second model</th><th class="r">WAPE ÷ R3</th><th class="r">Within ±20%</th><th class="r">Bias</th><th class="r">Change against the plain average, 95% interval</th></tr></thead><tbody></tbody></table></div>
    <p style="margin-top:10px">None beats the plain average ({{mp_base}}). The gate cannot tell which model will be closest: it is right {{mp_acc_hi}} of the time from high-level features, and {{mp_acc_pos}} when it also sees where each model sits, against 20% by chance. Taking its first choice costs most ({{mp_gate_pick}}). Using its probabilities as weights is close to equal weights and still behind ({{mp_gate_soft}}). Weights per product type, shrunk towards the overall weights, give {{mp_bayes}}.</p>
    <p>PCA shows why. The five models are mostly one shared view: the first component, with all five moving together, holds {{mp_pc1}} of the variation in their forecasts. The second ({{mp_pc2}}) is the rule agents against the four learned models. The other three hold {{mp_pc_rest}} between them. Regressing what sold on the components gives {{mp_pca_lo}} to {{mp_pca_hi}}, because a fitted model overfits the small components. The five have little independent information to separate. A better stack needs a member that knows something the five do not.</p></div>
'''
rep('  <div class="panel" id="all-low"', META + '  <div class="panel" id="all-low"')
JS = r'''
// ---- surprise inputs; a second model over the five
{ const ts = $('#t-surprise tbody'); J.surprise.rows.forEach((d, i) => ts.append(h('tr', i === 1 ? { class: 'on' } : {}, h('td', { text: d.variant }), h('td', { class: 'r', text: fx(d.wape, 4) }), h('td', { class: 'r', text: pct(d.hit, 1) }), h('td', { class: 'r', text: d.ci ? sf(d.ci[0]) + ' to ' + sf(d.ci[1]) : '' }))));
  const tm = $('#t-meta tbody'); J.metapick.rows.forEach((d, i) => tm.append(h('tr', i === 0 ? { class: 'on' } : {}, h('td', { text: d.variant }), h('td', { class: 'r', text: fx(d.wape, 3) }), h('td', { class: 'r', text: pct(d.hit, 1) }), h('td', { class: 'r', text: sp(d.bias) }), h('td', { class: 'r', text: d.ci ? sf(d.ci[0]) + ' to ' + sf(d.ci[1]) : '' })))); }
'''
rep("\n// ---- when\n", JS + "\n// ---- when\n")
rep("#t-loss td:first-child,#t-timeitem td:first-child{min-width:250px}", "#t-loss td:first-child,#t-timeitem td:first-child,#t-surprise td:first-child,#t-meta td:first-child{min-width:250px}")
open(p, "w", encoding="utf-8").write(s)

b = "eda/198g_build_error_study.py"; t = open(b, encoding="utf-8").read()
x = '    built="17 Nov 2026",'; assert x in t
add = '''    su_n=str(SU["n_inputs"]), su_rho1=sf(_ss["s1"]["rho"], 2), su_rho_tr=sf(_ss["trend91"]["rho"], 2), su_w=f3(SU["rows"][3]["wape"]), su_ref_w=f3(SU["rows"][1]["wape"]), su_auc1=f2(SU["flag_sur"]["auc"]),
    mp_base=f3(MP["rows"][0]["wape"]), mp_acc_hi=pc(_mp["gate, high-level features only: its first choice"]["acc"]), mp_acc_pos=pc(_mp["gate, high-level features and the models' positions: its first choice"]["acc"]),
    mp_gate_pick=f3(_mp["gate, high-level features only: its first choice"]["wape"]), mp_gate_soft=f3(_mp["gate, high-level features only: weights = its probabilities"]["wape"]),
    mp_bayes=f3(next(d for d in MP["rows"] if d["variant"].startswith("Bayes"))["wape"]), mp_pc1=pc(MP["pca"]["share"][0]), mp_pc2=pc(MP["pca"]["share"][1]), mp_pc_rest=pc(sum(MP["pca"]["share"][2:])),
    mp_pca_lo=f3(min(d["wape"] for d in MP["rows"] if d["variant"].startswith("PCA: the actual"))), mp_pca_hi=f3(max(d["wape"] for d in MP["rows"] if d["variant"].startswith("PCA: the actual"))),'''
t = t.replace(x, x + "\n" + add)
y = 'T = dict(\n'; assert y in t
t = t.replace(y, 'SU = J["surprise"]; _ss = {d["key"]: d for d in SU["sig"]}; MP = J["metapick"]; _mp = {d["variant"]: d for d in MP["rows"]}; assert all(d["wape"] > MP["rows"][0]["wape"] for d in MP["rows"][1:]) and MP["pca"]["load2"][0] > 0.8\n' + y, 1)
open(b, "w", encoding="utf-8").write(t); print("patched")
