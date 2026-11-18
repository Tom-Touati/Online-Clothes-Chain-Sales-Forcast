"""D283, D284 (fork): one-off patch after two comments on the error study ("we can add it as a known future - how many stores did it
reach" and "lets do it in origins which are 2 month apart"): a panel in the Distribution section with the known-future store test, a
panel in the Overfit section with the origin-spacing test (both eda/198n), a pointer in the summary, and their tokens. Idempotent."""
p = "eda/error_study_template.html"; s = open(p, encoding="utf-8").read()
if 'id="future-stores"' in s: raise SystemExit("already patched")


def rep(a, b):
    global s
    assert a in s, a[:80]; s = s.replace(a, b, 1)


FUT = '''  <div class="panel" id="future-stores" style="margin-top:18px"><h3>Stores reached, as an input known ahead</h3>
    <p>Asked in a comment on this page: add the number of own stores an item reaches during the forecast window as a known future. Tested like the other inputs: a correction trained on earlier origins only, with each origin's total kept.</p>
    <div class="tscroll"><table id="t-fstores"><thead><tr><th>Inputs</th><th class="r">WAPE ÷ R3</th><th class="r">Within ±20%</th><th class="r">Change against the reference, 95% interval</th></tr></thead><tbody></tbody></table></div>
    <p style="margin-top:10px">It helps: {{fs_w}} against {{fs_ref_w}} without it, and {{fs_hit}} of items within ±20% against {{fs_ref_hit}}. The gain sits where distribution moves. Items that gained five or more stores go from a bias of {{fs_b5_ref}} to {{fs_b5}}; launches under three months go from {{fs_l_ref}} to {{fs_l}} of R3's error.</p>
    <p>Two cautions. First, the backtest reads the count afterwards from the sales: a store counts once the item sells there, and strong sellers are sent to more stores. So part of the signal is demand itself. Items that gained two or more stores were under-forecast by {{fs_web_bias}} on the web alone, against {{fs_all_bias}} in total. A plan made before the window will give less: with the count off by about two stores the score is {{fs_noisy_w}}. Second, the data ends on 30 June, so it does not hold the count for July–October 2026. It has to come from the business: the planned number of own stores for each of the 100 targets. The opening calendar alone does not replace it (see <a href="#hit">±20%</a>).</p></div>
'''
rep('</section>\n\n<section id="overfit">', FUT + '</section>\n\n<section id="overfit">')
SPA = '''  <div class="panel" id="spacing" style="margin-top:18px"><h3>Origins further apart</h3>
    <p>Asked in a comment on this page: score on origins two months apart instead of one. Two months apart still share two of the four forecast months; only four months apart share none. The number of independent tests stays about three at any spacing, because the backtest covers the same 17 months however it is cut. More independent tests need more history, not wider spacing.</p>
    <div class="grid2">
      <div><div class="tscroll"><table id="t-neff"><thead><tr><th>Origins</th><th class="r">Count</th><th class="r">Months shared by neighbours</th><th class="r">Correlation of an item's error</th><th class="r">Independent tests</th></tr></thead><tbody></tbody></table></div>
        <div class="tscroll" style="margin-top:12px"><table id="t-sets"><thead><tr><th>Set of origins</th><th class="r">Origins</th><th class="r">System, WAPE ÷ R3</th><th class="r">R3 + rules</th><th class="r">Within ±20%</th><th class="r">R3 within ±20%</th></tr></thead><tbody></tbody></table></div></div>
      <div><div class="tscroll"><table id="t-steps4"><thead><tr><th>Adopted step</th><th class="r">All 13</th><th>Two sets, 2 months apart</th><th>Four sets, 4 months apart</th></tr></thead><tbody></tbody></table></div>
        <p class="small">One mark per set: blue = the step helped, red = it hurt.</p></div>
    </div>
    <p>What wider spacing does give is a check on sets that do not overlap. The headline holds in all four: the system beats R3 + rules by {{sp_gap_lo}} to {{sp_gap_hi}}, and {{sp_hit_lo}}–{{sp_hit_hi}} of items land within ±20% against {{sp_hr_lo}}–{{sp_hr_hi}} for R3. Of the eight steps, {{sp_n4}} help in all four sets and {{sp_n3}} in three. The new tree member (D253) helps in {{sp_tree_wins}} of the four, so its gain is the least established of the eight.</p></div>
'''
rep('</section>\n\n<section id="confounds">', SPA + '</section>\n\n<section id="confounds">')
rep("13 origins behave like about {{neff13}} separate tests.", '13 origins behave like about {{neff13}} separate tests, and spacing them further apart does not change that (see <a href="#spacing">Origins further apart</a>).')
JS1 = r'''
// ---- stores reached as a known future
{ const tb = $('#t-fstores tbody'); J.future_stores.rows.forEach((d, i) => tb.append(h('tr', i === 2 ? { class: 'on' } : {}, h('td', { text: d.variant }), h('td', { class: 'r', text: fx(d.wape, 3) }), h('td', { class: 'r', text: pct(d.hit, 1) }), h('td', { class: 'r', text: d.ci ? sf(d.ci[0]) + ' to ' + sf(d.ci[1]) : '' })))); }
'''
rep("\n// ---- overfit\n", JS1 + "\n// ---- overfit\n")
JS2 = r'''
// ---- origins further apart
{ const S = J.spacing, tn = $('#t-neff tbody'); S.neff.forEach(d => tn.append(h('tr', {}, h('td', { text: d.spacing === 1 ? '1 month apart (as scored)' : d.spacing + ' months apart' }), h('td', { class: 'r', text: String(d.n) }), h('td', { class: 'r', text: d.shared + ' of 4' }), h('td', { class: 'r', text: fx(d.corr, 2) }), h('td', { class: 'r', text: fx(d.neff, 1) }))));
  const ts = $('#t-sets tbody'); S.sets.forEach(d => ts.append(h('tr', { 'data-tip': '<b>' + d.set + '</b><br>' + d.origins.map(mon).join(', ') + '<br>bias ' + sp(d.bias) }, h('td', { text: d.set }), h('td', { class: 'r', text: String(d.n) }), h('td', { class: 'r', text: fx(d.wape, 3) }), h('td', { class: 'r', text: fx(d.rules, 3) }), h('td', { class: 'r', text: pct(d.hit, 1) }), h('td', { class: 'r', text: pct(d.hit_r3, 1) }))));
  const mk = a => { const st = h('span', { class: 'strip' }); a.forEach((d, j) => st.append(h('i', { style: 'background:var(' + (d < 0 ? '--under' : '--over') + ')', 'data-tip': '<b>set ' + (j + 1) + '</b>: ' + sf(d) }))); return st; };
  const t4 = $('#t-steps4 tbody'); S.steps.forEach(d => t4.append(h('tr', {}, h('td', { text: d.step }), h('td', { class: 'r', text: sf(d.d_all) }), h('td', {}, mk(d.d2), h('div', { class: 'small', text: d.wins2 + '/2' })), h('td', {}, mk(d.d4), h('div', { class: 'small', text: d.wins4 + '/4' }))))); }
'''
rep("\n// ---- confounds\n", JS2 + "\n// ---- confounds\n")
rep("#t-rules td:first-child,#t-combine td:first-child{min-width:250px}", "#t-rules td:first-child,#t-combine td:first-child,#t-fstores td:first-child{min-width:250px} #t-steps4 td:first-child{min-width:190px} #t-sets td:first-child,#t-neff td:first-child{white-space:nowrap}")
open(p, "w", encoding="utf-8").write(s)

b = "eda/198g_build_error_study.py"; t = open(b, encoding="utf-8").read()
x = '    built="17 Nov 2026",'; assert x in t
add = '''    fs_w=f3(FS["rows"][2]["wape"]), fs_ref_w=f3(FS["rows"][1]["wape"]), fs_hit=pc(FS["rows"][2]["hit"], 1), fs_ref_hit=pc(FS["rows"][1]["hit"], 1), fs_noisy_w=f3(FS["rows"][5]["wape"]),
    fs_b5_ref=sp(_fc["items that gained 5+ stores"]["bias_ref"]), fs_b5=sp(_fc["items that gained 5+ stores"]["bias_k"]), fs_l_ref=f3(_fc["launches under 3 months"]["wape_ref"]), fs_l=f3(_fc["launches under 3 months"]["wape_k"]),
    fs_web_bias=pc(FS["web_bias_add2"]), fs_all_bias=pc(FS["bias_add2"]),
    sp_gap_lo=f2(min(d["rules"] - d["wape"] for d in _s4)), sp_gap_hi=f2(max(d["rules"] - d["wape"] for d in _s4)), sp_hit_lo=pc(min(d["hit"] for d in _s4)), sp_hit_hi=pc(max(d["hit"] for d in _s4)),
    sp_hr_lo=pc(min(d["hit_r3"] for d in _s4)), sp_hr_hi=pc(max(d["hit_r3"] for d in _s4)), sp_n4=str(sum(d["wins4"] == 4 for d in SPC["steps"])), sp_n3=str(sum(d["wins4"] == 3 for d in SPC["steps"])),
    sp_tree_wins=str(next(d for d in SPC["steps"] if "D253" in d["step"])["wins4"]),'''
t = t.replace(x, x + "\n" + add)
y = 'T = dict(\n'; assert y in t
t = t.replace(y, 'FS = J["future_stores"]; _fc = {d["g"]: d for d in FS["cohorts"]}; SPC = J["spacing"]; _s4 = SPC["sets"][3:]; assert all(d["rules"] > d["wape"] for d in _s4)\n' + y, 1)
open(b, "w", encoding="utf-8").write(t); print("patched")
