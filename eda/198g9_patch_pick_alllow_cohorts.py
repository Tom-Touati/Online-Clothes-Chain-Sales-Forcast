"""D285-D287 (fork): one-off patch after three comments on the error study:
  "i dont really get this table" (cohorts with the lowest hit rate)   -> both cohort tables in plain words, with a how-to-read note
  "we can pick the top models for each item based on the past"         -> panel 'Picking each item's best models from its past' (eda/198o)
  "All five models too low, 25% of forecasts: how do we handle that"   -> panel 'When all five models are too low' (eda/198o)
Idempotent."""
p = "eda/error_study_template.html"; s = open(p, encoding="utf-8").read()
if 'id="all-low"' in s: raise SystemExit("already patched")


def rep(a, b):
    global s
    assert a in s, a[:80]; s = s.replace(a, b, 1)


# 1 cohort tables in plain words
HEAD = '<thead><tr><th>Cohort</th><th class="r">Forecasts</th><th class="r">Share of units</th><th class="r">Within ±20%</th></tr></thead>'
NEWH = '<thead><tr><th>Group of forecasts</th><th>What sets it apart</th><th class="r">Forecasts</th><th class="r">Share of all units</th><th class="r">Within ±20%</th></tr></thead>'
assert s.count(HEAD) == 2; s = s.replace(HEAD, NEWH)
rep('  <div class="grid2">\n    <div class="panel"><h3>Cohorts with the lowest hit rate</h3>',
    '''  <p style="margin-top:14px"><b>How to read the next two tables.</b> Each row is a group of forecasts that share one property known on the day the forecast is made. "Within ±20%" is the share of that group's forecasts that land within ±20% of what sold; over all forecasts it is {{hit_items}}. The left table lists the eight groups that land there least often, the right table the eight that land there most often. A forecast can be in several groups. "Share of all units" shows how much of the sales the group holds.</p>
  <div class="grid2">
    <div class="panel"><h3>Where the forecast lands within ±20% least often</h3>''')
rep('<div class="panel"><h3>Cohorts with the highest hit rate</h3>', '<div class="panel"><h3>Where it lands within ±20% most often</h3>')
rep("  <div class=\"grid2\">\n    <div class=\"panel\"><h3>Can a confident group be named in advance?</h3>",
    '''  <p style="margin-top:12px">In short: the hard forecasts are small items, items short of stock or sizes, and forecasts the models themselves mark as unsure. Most of these groups hold little of the sales: the first three rows together hold {{hl_small_units}} of units. The two with real volume are the widest-range fifth and the most-disagreement fifth. The easy ones are large, widely distributed basics.</p>
  <div class="grid2">
    <div class="panel"><h3>Can a confident group be named in advance?</h3>''')
rep("h('td', { html: `${d.g}<br><span class=\"small\">${d.axis}</span>` }), h('td', { class: 'r', text: int(d.n) })",
    "h('td', { html: `${d.plain || d.g}<br><span class=\"small\">${d.axis}: ${d.g}</span>` }), h('td', { class: 'small', text: d.why || '' }), h('td', { class: 'r', text: int(d.n) })")
rep('"Best of the five models" picks the closest model for each item afterwards; it is a bound, not an option.',
    '"Best of the five models" picks the closest model for each item afterwards; it is a bound, not an option. Picking from each item\'s past instead was tested and does worse than the plain average (see <a href="#item-pick">Where</a>).')

# 2, 3 two panels at the end of the Where section
PAN = '''  <div class="panel" id="item-pick" style="margin-top:18px"><h3>Picking each item's best models from its past</h3>
    <p>Asked in a comment on this page. For each item, take the five models' errors on that same item at earlier origins whose results were already known, then keep its best one, two or three. {{ip_nohist}} of recent forecasts have no such history and keep the plain average; the rest have a median of {{ip_med}} earlier origins. Scored before the correction layer.</p>
    <div class="tscroll"><table id="t-pick"><thead><tr><th>Choice</th><th class="r">WAPE ÷ R3</th><th class="r">Within ±20%</th><th class="r">Change against the plain average, 95% interval</th></tr></thead><tbody></tbody></table></div>
    <p style="margin-top:10px">Every version is worse than the plain average or level with it. The reason is that the best model for an item does not stay the best. An item's best model at the last known origin is its best again {{ip_persist}} of the time; picking at random gives 20%. Each of the five is the closest for about a fifth of the items ({{ip_share_lo}}–{{ip_share_hi}}), and on any one item their errors move together (correlation {{ip_corr}}): they are mostly all too low or all too high, so small differences that do not repeat decide which is closest. The {{sv_bound_hit}} bound needs to know the answer first.</p></div>
  <div class="panel" id="all-low" style="margin-top:18px"><h3>When all five models are too low</h3>
    <p>Asked in a comment on this page: how do we handle it? {{al_share}} of recent forecasts have all five models below what sold, and they hold {{al_err}} of the error. The actual lands at a median of {{al_over_f}}× the forecast and {{al_over_top}}× the highest of the five. No weighting of the five can reach above the highest one, so the fix has to come from outside the models.</p>
    <div class="grid2">
      <div><div class="tscroll"><table id="t-alllow"><thead><tr><th>Share of forecasts where</th><th class="r">All five too low</th><th class="r">All other forecasts</th></tr></thead><tbody>
        <tr><td>the first month already ran 20% above the forecast's pace</td><td class="r">{{al_m1}}</td><td class="r">{{al_m1_r}}</td></tr>
        <tr><td>the item gained two or more own stores in the window</td><td class="r">{{al_st2}}</td><td class="r">{{al_st2_r}}</td></tr>
        <tr><td>the item gained a partner in the window</td><td class="r">{{al_par}}</td><td class="r">{{al_par_r}}</td></tr>
        <tr><td>the item was a launch under three months old</td><td class="r">{{al_young}}</td><td class="r">{{al_young_r}}</td></tr>
        <tr><td>the actual stayed below the stated P90</td><td class="r">{{al_in}}</td><td class="r">{{al_in_r}}</td></tr></tbody></table></div>
        <p class="small">Recent origins. The first three rows are only known afterwards; they show what was behind the miss.</p></div>
      <div><div id="c-allow"></div>
        <p class="small">A classifier trained only on earlier origins ranks each origin's forecasts by the risk of this miss. Bars: share of each tenth where all five models were too low, lowest risk to highest. AUC {{al_auc}}; the overall share is {{al_share}}.</p></div>
    </div>
    <p style="margin-top:10px">Three things handle it.</p>
    <ol>
      <li><b>Re-forecast after the first month.</b> In {{al_m1}} of these forecasts the first month already ran well above pace. With the plans and one month of sales known, their error falls by {{al_cut_all}} and {{al_hit_all}} of them land within ±20%, against {{al_hit_ref}} without.</li>
      <li><b>The roll-out plan.</b> {{al_st2}} of them gained two or more stores, twice the rate of other forecasts. Plans alone cut their error by {{al_cut_plan}}.</li>
      <li><b>A watch list, for buying and for attention.</b> The fifth of forecasts with the highest risk holds {{al_top_err}} of this error, and {{al_top_rate}} of that fifth turn out all-too-low. Raising the whole fifth by the factor seen at earlier origins ({{al_fac_lo}}–{{al_fac_hi}}) brings the overall bias from {{al_up_bias0}} to {{al_up_bias}} but leaves accuracy where it was ({{al_up_w}} against {{al_up_w0}} over the last {{al_up_n}} origins), because it also raises the flagged items that did not take off. So the flag should not move the forecast. It says which items to re-forecast first and where to buy above the forecast.</li>
    </ol>
    <p>The stated range also needs widening for them: only {{al_in}} of these actuals stay below the P90, against {{al_in_r}} for other forecasts.</p></div>
'''
rep('</section>\n\n<section id="when">', PAN + '</section>\n\n<section id="when">')
JS = r'''
// ---- picking per item from its past; all five too low
{ const tp = $('#t-pick tbody'); J.itempick.rows.forEach((d, i) => tp.append(h('tr', Object.assign(i === 0 ? { class: 'on' } : {}, d.cover != null && d.cover < 1 ? { 'data-tip': '<b>' + d.variant + '</b><br>' + pct(d.cover) + ' of recent forecasts have the history this needs<br>all 13 origins: ' + fx(d.wape_all, 3) } : {}), h('td', { text: d.variant }), h('td', { class: 'r', text: fx(d.wape, 3) }), h('td', { class: 'r', text: pct(d.hit, 1) }), h('td', { class: 'r', text: d.ci ? sf(d.ci[0]) + ' to ' + sf(d.ci[1]) : '' }))));
  const A = J.alllow.clf; groupedBars($('#c-allow'), A.deciles.map(d => ({ label: String(d.decile), v: [d.rate], tip: '<b>Risk tenth ' + d.decile + '</b><br>' + pct(d.rate, 1) + ' had all five models too low<br>' + pct(d.err, 1) + ' of the all-too-low error<br>bias of the tenth ' + sp(d.bias) })), [{ c: '--accent' }], { max: 0.5, ticks: [0, .1, .2, .3, .4, .5], fmt: v => pct(v), h: 220, label: 'Share of forecasts with all five models too low, by tenth of predicted risk' }); }
'''
rep("\n// ---- when\n", JS + "\n// ---- when\n")
rep("#t-fstores td:first-child{min-width:250px}", "#t-fstores td:first-child,#t-pick td:first-child{min-width:250px} #t-alllow td:first-child{min-width:220px} #t-hlow td:nth-child(2),#t-hhigh td:nth-child(2){min-width:170px}")
open(p, "w", encoding="utf-8").write(s)

b = "eda/198g_build_error_study.py"; t = open(b, encoding="utf-8").read()
x = '    built="17 Nov 2026",'; assert x in t
add = '''    hl_small_units=pc(sum(d["units"] for d in HT["cohort_low"][:3]), 1),
    ip_nohist=pc(IP["no_history"]), ip_med=str(int(IP["med_history"])), ip_persist=pc(IP["persist"]), ip_share_lo=pc(min(IP["best_share"].values())), ip_share_hi=pc(max(IP["best_share"].values())), ip_corr=f2(IP["member_corr"]),
    al_share=pc(AL["all_low"]["share"]), al_err=pc(AL["all_low"]["err"]), al_over_f=f2(AL["all_low"]["med_over_F"]), al_over_top=f2(AL["all_low"]["med_over_top"]),
    al_m1=pc(AL["all_low"]["m1_up"]), al_m1_r=pc(AL["rest"]["m1_up"]), al_st2=pc(AL["all_low"]["stores2"]), al_st2_r=pc(AL["rest"]["stores2"]), al_par=pc(AL["all_low"]["partners1"]), al_par_r=pc(AL["rest"]["partners1"]),
    al_young=pc(AL["all_low"]["young"]), al_young_r=pc(AL["rest"]["young"]), al_in=pc(AL["all_low"]["in_range"]), al_in_r=pc(AL["rest"]["in_range"]), al_auc=f2(AL["clf"]["auc"]),
    al_cut_all=pc(AL["info"]["all_cut"]), al_cut_plan=pc(AL["info"]["plan_cut"]), al_hit_all=pc(AL["info"]["hit_all"]), al_hit_ref=pc(AL["info"]["hit_ref"]), al_top_err=pc(AL["clf"]["top20_err"]), al_top_rate=pc(AL["clf"]["top20_rate"]),
    al_fac_lo=f2(min(f["factor"] for f in AL["uplift"]["factors"])), al_fac_hi=f2(max(f["factor"] for f in AL["uplift"]["factors"])), al_up_bias0=sp(AL["uplift"]["bias_base"]), al_up_bias=sp(AL["uplift"]["bias"]),
    al_up_w=f3(AL["uplift"]["wape"]), al_up_w0=f3(AL["uplift"]["wape_base"]), al_up_n=str(len(AL["uplift"]["origins"])),'''
t = t.replace(x, x + "\n" + add)
y = 'T = dict(\n'; assert y in t
pre = '''IP = J["itempick"]; AL = J["alllow"]
# plain-language labels for the two hit-rate cohort tables (user comment: "i dont really get this table"); the build fails if a shown row has none
PLAIN = {("size_run", "under 60% (broken run)"): ("Items selling in under 60% of their sizes", "Most sizes are gone, so sales follow the stock that is left."),
         ("oos", "15% +"): ("Items out of stock on 15% or more of the last 28 days", "Stock gaps held sales back, so the forecast starts too low."),
         ("volume", "fifth 1"): ("The smallest fifth of items", "A few units either way is already more than 20%."),
         ("interval", "low"): ("The fifth of forecasts with the widest stated range", "The models themselves say they are unsure."),
         ("partners", "none"): ("Items sold through no partner", "Mostly small items sold on the web only."),
         ("oos", "some (under 15%)"): ("Items out of stock on some of the last 28 days", "As above: sales were held back by stock."),
         ("burst_hour", "low"): ("Items with one hour holding 6% or more of their online units", "Mostly very small items: with few orders, one hour is a large share."),
         ("spread", "low"): ("The fifth of forecasts where the five models disagree most", "Disagreement marks an unsure forecast."),
         ("interval", "high"): ("The fifth of forecasts with the narrowest stated range", "The models are sure, and they are right more often."),
         ("siblings", "4-6"): ("Styles with 4 to 6 live colours", "Core styles that carry several colours."),
         ("volume", "fifth 5"): ("The largest fifth of items", "Large, steady sellers; chance plays a small part."),
         ("stores", "8-11"): ("Items sold in 8 to 11 own stores", "Already widely distributed, so little roll-out is left."),
         ("ptype", "Sweatpants"): ("Sweatpants", "A basic with steady demand."), ("ptype", "Jeans"): ("Jeans", "A basic with steady demand."), ("ptype", "Hoodie"): ("Hoodies", "A basic with steady demand."),
         ("style_share", "40-80%"): ("Colours holding 40 to 80% of their style's units", "The main colour of an established style.")}
for _side, _L in (("low", HT["cohort_low"]), ("high", HT["cohort_high"])):
    for _d in _L[:8]:
        _k = next((k for k in [(_d["key"], _d["g"]), (_d["key"], _d["g"].split(" (")[0]), (_d["key"], _side)] if k in PLAIN), None); assert _k is not None, (_d["key"], _d["g"])
        _d["plain"], _d["why"] = PLAIN[_k]
        if _d["key"] == "volume": _d["plain"] += " (median " + it(float(_d["g"].split("median ")[1].split("/")[0])) + " units a month)"
'''
t = t.replace(y, pre + y, 1)
open(b, "w", encoding="utf-8").write(t); print("patched")
