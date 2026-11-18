"""D288, D289 (fork): one-off patch after two comment threads on the error study:
  "is it specific times that yield an error on a lot of items or specific items?" (+ an outside seasonal index)  -> panel 'Specific times, or specific items?'
  "show it both normalized by volume and not" / "normalized by the total" / "loss ... on percentage wrong"        -> panel 'Scored in units and per item'
Both from eda/198p. Idempotent."""
p = "eda/error_study_template.html"; s = open(p, encoding="utf-8").read()
if 'id="both-ways"' in s: raise SystemExit("already patched")


def rep(a, b):
    global s
    assert a in s, a[:80]; s = s.replace(a, b, 1)


PAN = '''  <div class="panel" id="both-ways" style="margin-top:18px"><h3>Scored in units and per item</h3>
    <p>Asked in a comment on this page: show the evaluation both ways. In units, every unit sold counts once, so large items dominate (WAPE). Per item, every item counts once and its miss is measured as a percentage of its own total, so the many small items dominate.</p>
    <div class="tscroll"><table id="t-both"><thead><tr><th rowspan="2">Recent origins</th><th class="r" colspan="2">In units</th><th class="r" colspan="4">Per item, as a percentage of its own total</th></tr>
      <tr><th class="r">Error ÷ units sold</th><th class="r">÷ R3</th><th class="r">Median % error</th><th class="r">÷ R3</th><th class="r">Mean % error</th><th class="r">Within ±20%</th></tr></thead><tbody></tbody></table></div>
    <p class="small">Mean % error caps each item at 200%, otherwise a handful of items that sold almost nothing decide it.</p>
    <p>The two views agree on the ranking and on the size of the gain: the system has {{bw_ratio}} of R3's error in units and {{bw_med_ratio}} of its median percentage error. They disagree on where the error sits. The worst 5% of forecasts hold {{bw_top5_units}} of the unit error but {{bw_top5_pct}} of the percentage error. In units, the largest fifth of items holds {{bw_f5_units_err}} of the error; in percentages, the smallest fifth holds {{bw_f1_pct_err}} while selling {{bw_f1_units}} of the units.</p>
    <div class="tscroll"><table id="t-bothfifth"><thead><tr><th>Volume fifth</th><th class="r">Share of units</th><th class="r">Error ÷ units sold</th><th class="r">Share of the unit error</th><th class="r">Median % error</th><th class="r">Share of the percentage error</th><th class="r">Within ±20%</th></tr></thead><tbody></tbody></table></div>
    <h3 style="margin-top:18px">Should the models be trained on percentages?</h3>
    <p>They already are. Every learned member predicts the four-month total as a ratio to the item's own recent level, in logs, and is trained on a quantile loss of that ratio with every item counted equally. Most inputs are normalised the same way: trend, momentum, seasonality and channel shares are ratios to the item's own level. The table moves the tree member step by step from that percentage loss towards a loss on units.</p>
    <div class="tscroll"><table id="t-loss"><thead><tr><th>Tree member alone, trained with</th><th class="r">WAPE ÷ R3</th><th class="r">Median % error</th><th class="r">Within ±20%</th><th class="r">Bias</th></tr></thead><tbody></tbody></table></div>
    <p class="small">A loss on units is worse on both views ({{ls_unit_w}} and {{ls_unit_hit}}, against {{ls_base_w}} and {{ls_base_hit}}). A light weight on volume is slightly better in units ({{ls_w03_w}}) and level per item; in the full system it came out level (D264).</p></div>
  <div class="panel" id="time-item" style="margin-top:18px"><h3>Specific times, or specific items?</h3>
    <p>Asked in a comment on this page. Mostly neither on its own: the miss belongs to one item in one window. The table splits the variation in the miss by what moves together, and shows the score if that total had been known exactly.</p>
    <div class="tscroll"><table id="t-timeitem"><thead><tr><th>What moves together</th><th class="r">Groups per origin</th><th class="r">Share of the variation in the miss</th><th class="r">WAPE ÷ R3 if its total were known</th><th class="r">Within ±20%</th></tr></thead><tbody></tbody></table></div>
    <p class="small">Each row contains the rows above it. Shares are variance components of the log miss, recent origins.</p>
    <p><b>Times.</b> All items being too low or too high together explains {{ti_time}} of the variation. At each origin between {{ti_under_lo}} and {{ti_under_hi}} of items are under-forecast and the total is off by {{ti_bias_lo}} to {{ti_bias_hi}}; knowing every origin's total exactly would give {{ti_tot_w}} and {{ti_tot_hit}} within ±20%. The worst 5% of forecasts ({{ti_w5_n}}) are spread over all nine origins.</p>
    <p><b>Items.</b> The same item is not missed again and again. Between windows that share no month, an item's miss repeats for {{ti_item}} of the variation; the {{ti_item_ov}} seen across overlapping origins is mostly the same window counted several times. Correcting each item by its own earlier miss makes the forecast worse ({{ti_pc25}} at a quarter of the miss, {{ti_pc50}} at half). The worst 5% of forecasts are {{ti_w5_items}} different items. Error by item follows volume: the top {{ti_top5_n}} items hold {{ti_top5_err}} of the error and {{ti_top5_units}} of the units.</p>
    <p><b>An outside seasonal index</b> (how much US fashion e-commerce rises at this time of year) would act on the first row: at best {{ratio_recent}} → {{ti_tot_w}}, with no gain in the hit rate. An index by category would act on the third row, where the ceiling is {{ti_type_w}}; the models already carry Edikted's own seasonality by product type, so only the market-wide part that Edikted's history does not show is left. Not tested: the data holds no outside series.</p></div>
'''
rep('  <div class="panel" id="item-pick"', PAN + '  <div class="panel" id="item-pick"')
rep("They count different things. The error share is measured in units, so large items dominate it. The hit rate counts every item once, so the many ordinary items dominate it.",
    'They count different things. The error share is measured in units, so large items dominate it. The hit rate counts every item once, so the many ordinary items dominate it. Both views side by side are in <a href="#both-ways">Scored in units and per item</a>.')
JS = r'''
// ---- scored both ways; the loss; time or item
{ const B = J.bothways, tb = $('#t-both tbody'); B.systems.forEach((d, i) => tb.append(h('tr', i === 2 ? { class: 'on' } : {}, h('td', { text: d.system }), h('td', { class: 'r', text: pct(d.wape_abs, 1) }), h('td', { class: 'r', text: fx(d.ratio, 3) }), h('td', { class: 'r', text: pct(d.med_ape, 1) }), h('td', { class: 'r', text: fx(d.med_ape_ratio, 3) }), h('td', { class: 'r', text: pct(d.mean_ape, 1) }), h('td', { class: 'r', text: pct(d.hit, 1) }))));
  const tf = $('#t-bothfifth tbody'); B.fifths.forEach(d => tf.append(h('tr', {}, h('td', { text: 'fifth ' + d.fifth + ' (median ' + int(d.med_level) + ' / month)' }), h('td', { class: 'r', text: pct(d.units, 1) }), h('td', { class: 'r', text: pct(d.wape_abs) }), h('td', { class: 'r', text: pct(d.err_units) }), h('td', { class: 'r', text: pct(d.med_ape) }), h('td', { class: 'r', text: pct(d.err_pct) }), h('td', { class: 'r', text: pct(d.hit) }))));
  const tl = $('#t-loss tbody'); J.lossscale.forEach((d, i) => tl.append(h('tr', i === 0 ? { class: 'on' } : {}, h('td', { text: d.variant }), h('td', { class: 'r', text: fx(d.ratio, 3) }), h('td', { class: 'r', text: pct(d.med_ape, 1) }), h('td', { class: 'r', text: pct(d.hit, 1) }), h('td', { class: 'r', text: sp(d.bias) }))));
  const T = J.timeitem, sh = [T.time, T.group_time, T.type_time, T.cell_time], nm = ['Everything at one origin (the time)', 'A product group at one origin', 'A product type at one origin', 'A type × look group at one origin'], tt = $('#t-timeitem tbody');
  T.oracles.forEach((d, i) => tt.append(h('tr', {}, h('td', { text: nm[i] }), h('td', { class: 'r', text: String(d.groups) }), h('td', { class: 'r', text: pct(sh[i]) }), h('td', { class: 'r', text: fx(d.wape, 3) }), h('td', { class: 'r', text: pct(d.hit, 1) }))));
  tt.append(h('tr', { class: 'on' }, h('td', { text: 'Today, nothing known' }), h('td', { class: 'r', text: '' }), h('td', { class: 'r', text: '' }), h('td', { class: 'r', text: fx(T.system, 3) }), h('td', { class: 'r', text: pct(T.system_hit, 1) }))); }
'''
rep("\n// ---- when\n", JS + "\n// ---- when\n")
rep("#t-fstores td:first-child,#t-pick td:first-child{min-width:250px}", "#t-fstores td:first-child,#t-pick td:first-child,#t-loss td:first-child,#t-timeitem td:first-child{min-width:250px} #t-both td:first-child,#t-bothfifth td:first-child{white-space:nowrap}")
open(p, "w", encoding="utf-8").write(s)

b = "eda/198g_build_error_study.py"; t = open(b, encoding="utf-8").read()
x = '    built="17 Nov 2026",'; assert x in t
add = '''    bw_ratio=f3(BWS["ratio"]), bw_med_ratio=f3(BWS["med_ape_ratio"]), bw_top5_units=pc(BWS["top5_units"]), bw_top5_pct=pc(BWS["top5_pct"]), bw_f5_units_err=pc(BWF[4]["err_units"]), bw_f1_pct_err=pc(BWF[0]["err_pct"]), bw_f1_units=pc(BWF[0]["units"], 1),
    ls_base_w=f3(LSC[0]["ratio"]), ls_base_hit=pc(LSC[0]["hit"], 1), ls_unit_w=f3(LSC[4]["ratio"]), ls_unit_hit=pc(LSC[4]["hit"], 1), ls_w03_w=f3(LSC[1]["ratio"]),
    ti_time=pc(TI["time"]), ti_item=pc(TI["item"]), ti_item_ov=pc(TI["item_overlap"]), ti_under_lo=pc(TI["under_range"][0]), ti_under_hi=pc(TI["under_range"][1]), ti_bias_lo=sp(TI["bias_range"][0]), ti_bias_hi=sp(TI["bias_range"][1]),
    ti_tot_w=f3(TI["oracles"][0]["wape"]), ti_tot_hit=pc(TI["oracles"][0]["hit"], 1), ti_type_w=f3(TI["oracles"][2]["wape"]), ti_pc25=f3(TI["past_corr"][0]["wape"]), ti_pc50=f3(TI["past_corr"][1]["wape"]),
    ti_w5_n=it(TI["worst5"]["n"]), ti_w5_items=str(TI["worst5"]["items"]), ti_top5_n=str(TI["top_items"][1]["items"]), ti_top5_err=pc(TI["top_items"][1]["err"]), ti_top5_units=pc(TI["top_items"][1]["units"]),'''
t = t.replace(x, x + "\n" + add)
y = 'T = dict(\n'; assert y in t
t = t.replace(y, 'TI = J["timeitem"]; BWS = next(d for d in J["bothways"]["systems"] if d["system"] == "Submitted system"); BWF = J["bothways"]["fifths"]; LSC = J["lossscale"]; assert "Tweedie" in LSC[4]["variant"] and "0.3" in LSC[1]["variant"] and TI["worst5"]["origins"] == 9\n' + y, 1)
open(b, "w", encoding="utf-8").write(t); print("patched")
