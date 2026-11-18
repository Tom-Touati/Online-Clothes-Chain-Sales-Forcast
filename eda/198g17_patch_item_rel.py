"""D313 (fork): one-off patch after the user's question in the session ("we need a metric that reflects all items, so we need to have a
wape : R3 that sums the ratio rather than taking the ratio of sums, or something similar right?"): an item-equal comparison with R3 in the
panel 'Scored in units and per item' (eda/198z, J["itemrel"]). Idempotent."""
p = "eda/error_study_template.html"; s = open(p, encoding="utf-8").read()
if 'id="t-rel"' in s: raise SystemExit("already patched")


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80]); s = s.replace(a, b, 1)


BLOCK = '''    <h3 style="margin-top:18px">Against R3 item by item</h3>
    <p>Asked in the session: WAPE ÷ R3 is a ratio of sums, so large items decide it. The item-equal version takes each item's own error relative to R3's, (|F − y| + 1) ÷ (|R3 − y| + 1), and then averages over items. How it is averaged matters. A plain average of the ratios is lopsided: an item twice as good counts 0.5 and an item twice as bad counts 2, and items where R3 happened to be almost exact give huge ratios. Even capped at 5× per item, the plain average puts the system at {{ir_mean}}, close to R3, which does not match any other view. The geometric mean (GMRAE) fixes this, since a 2× better and a 2× worse item cancel. It is the standard item-equal choice, together with the median (MdRAE) and the share of items where the forecast is closer than R3.</p>
    <div class="tscroll"><table id="t-rel"><thead><tr><th>Recent origins</th><th class="r">WAPE ÷ R3 (units)</th><th class="r">Geometric mean of item ratios (GMRAE)</th><th class="r">Median item ratio (MdRAE)</th><th class="r">Items closer than R3</th><th class="r">Plain mean of item ratios, capped at 5×</th></tr></thead><tbody></tbody></table></div>
    <p class="small">Below 1 means better than R3; R3 itself scores 1 on every ratio column, and a forecast no better than R3 would be closer for about half the items. All 13 origins: the system's GMRAE is {{ir_g13}} against {{ir_r13}} for R3 + rules. The +1 unit keeps an exact R3 from dividing by zero; with +5 the system's GMRAE is {{ir_g5}}.</p>
    <p>The item-equal view agrees with the unit view on the ranking: the system beats R3 + rules and every single model on GMRAE and on the share closer than R3. It is less flattering: {{ir_g}} against {{ir_w}} in units, because the system gains most on large items. By volume fifth, the system's GMRAE runs from {{ir_f1}} on the smallest items to {{ir_f5}} on the largest. The 11-signal version (D307) helps in units (WAPE ÷ R3 {{ir_sig_w}}) but is level item by item (GMRAE {{ir_sig_g}}): its gain sits on larger items.</p>
'''
rep('    <h3 style="margin-top:18px">R²: how much of the variation the forecast explains</h3>', BLOCK + '    <h3 style="margin-top:18px">R²: how much of the variation the forecast explains</h3>')
JS = r'''
// ---- item-equal comparison with R3
{ const tr = $('#t-rel tbody'); J.itemrel.systems.forEach(d => tr.append(h('tr', d.system === 'Submitted system' ? { class: 'on' } : { 'data-tip': '<b>' + d.system + '</b><br>all 13 origins: GMRAE ' + fx(d.all13.gmrae, 3) + ', ' + pct(d.all13.better, 1) + ' closer than R3' }, h('td', { text: d.system }), h('td', { class: 'r', text: fx(d.wape_r3, 3) }), h('td', { class: 'r', text: fx(d.gmrae, 3) }), h('td', { class: 'r', text: fx(d.mdrae, 3) }), h('td', { class: 'r', text: pct(d.better, 1) }), h('td', { class: 'r', text: fx(d.mean_rel, 3) })))); }
'''
rep("\n// ---- when\n", JS + "\n// ---- when\n")
rep("#t-r2 td:first-child{white-space:nowrap}", "#t-r2 td:first-child,#t-rel td:first-child{white-space:nowrap}")
open(p, "w", encoding="utf-8").write(s)

b = "eda/198g_build_error_study.py"; t = open(b, encoding="utf-8").read()
y = 'T = dict(\n'; assert y in t
t = t.replace(y, '_ir = {d["system"]: d for d in J["itemrel"]["systems"]}; _irs = _ir["Submitted system"]; _irf = J["itemrel"]["by_fifth"]; _irfl = {d["floor"]: d for d in J["itemrel"]["floor"]}\n' + y, 1)
x = '    built="17 Nov 2026",'; assert x in t
t = t.replace(x, x + '''
    ir_mean=f3(_irs["mean_rel"]), ir_g=f3(_irs["gmrae"]), ir_w=f3(_irs["wape_r3"]), ir_g13=f3(_irs["all13"]["gmrae"]), ir_r13=f3(_ir["R3 + rules"]["all13"]["gmrae"]), ir_g5=f3(_irfl[5]["gmrae"]),
    ir_f1=f3(_irf[0]["gmrae"]), ir_f5=f3(_irf[-1]["gmrae"]), ir_sig_w=f"{_ir['Submitted with the 11 signals (D307)']['wape_r3']:.4f} against {_irs['wape_r3']:.4f}", ir_sig_g=f"{_ir['Submitted with the 11 signals (D307)']['gmrae']:.4f} against {_irs['gmrae']:.4f}",''')
open(b, "w", encoding="utf-8").write(t); print("patched")
