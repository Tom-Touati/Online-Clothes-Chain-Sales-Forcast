"""D276 (fork): builds the 'Edikted Error Study' page. data/fk_err_analysis.json (eda/198e, 198e2) + eda/error_study_template.html ->
eda/error_study.html (the published artifact file). Every number in the prose is a {{token}} filled here from the JSON, so text and
charts cannot disagree. '</' inside the JSON is escaped. Also writes eda/error_study_preview.html (wrapped, for a local render)."""
import json, re, sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
J = json.load(open("data/fk_err_analysis.json")); H = J["headline"]; O = J["origins"]; REC = [o for o in O if o["origin"] >= "2025-06"]
import math
_r = lambda v, d: f"{math.floor(v * 10 ** d + 0.5 + 1e-9) / 10 ** d:.{d}f}"   # round half up, as the page's own number formatting does
pc = lambda x, d=0: _r(abs(x) * 100, d) + "%"
sp = lambda x, d=0: ("" if float(_r(abs(x) * 100, d)) == 0 else "+" if x > 0 else "−") + _r(abs(x) * 100, d) + "%"
sf = lambda x, d=3: ("+" if x > 0 else "−" if x < 0 else "") + f"{abs(x):.{d}f}"
f3 = lambda x: f"{x:.3f}"; f2 = lambda x: f"{x:.2f}"; it = lambda x: f"{int(round(x)):,}"
co = J["cohorts"]; grp = lambda k, g: next(d for d in co[k]["groups"] if d["g"] == g); voi = {d["block"].split(" (")[0]: d for d in J["voi"]}
S = J["learning"]["season"]; yd = {d["g"]: d for d in J["young_distribution"]}; cu = {d["tag"]: d for d in J["causes"]["under"]}; cov = {d["tag"]: d for d in J["causes"]["over"]}
ch = J["chain"]; an = J["analogue"]["o6"]; sc = J["store_check"]; mix = J["confounds"]["mix"]; fl = {f["flag"]: f for f in J["targets26"]["flags"]}; cz = J["confounds"]["censor"]
HT = J["hit"]; LS = HT["level_shape"]; BD = {d["band"]: d for d in HT["bands"]}; LV = {d["level"]: d for d in HT["levels"]}
LN = J["launch"]; _ar = pd.read_parquet("data/fk_store_arrivals.parquet"); LN["cross_store_arrivals"] = len(_ar); LN["n_stores"] = int(_ar[_ar.kind == "store"]["loc"].nunique()); LN["n_partners"] = int(_ar[_ar.kind == "partner"]["loc"].nunique())
TC = J["twocounts"]; SV = J["stackvar"]
CE = J["cells"]; NZ = {d["sd"]: d["hit"] for d in CE["noise"]}
c26 = pd.read_csv("data/final_2026_nx2_cascade.csv", index_col=0); WC = ["S24840_LIGHT-GRAY-MELANGE", "S24847_NAVY", "S25039_YELLOW"]
wc_now = float(c26.loc[WC, "total"].sum()); wc_rule = 34463.0   # D033 rule total of the 12 Nov submission (log D275)
import openpyxl
def _wb(f):
    ws = openpyxl.load_workbook(f, data_only=True)["Forecast"]; return {r[1]: (r[2], [float(v) for v in r[6:10]]) for r in ws.iter_rows(min_row=2, values_only=True) if r[1] in WC}
_old, _new = _wb("candidates/Edikted_Forecasting_Exercise_candidate_nx2.xlsx"), _wb("Edikted_Forecasting_Exercise_filled.xlsx")   # the models without the cap | the submitted workbook
_first = _wb("candidates/Edikted_Forecasting_Exercise_filled_original_2026-11-12.xlsx"); assert all(abs(x - y_) < 0.51 for k in WC for x, y_ in zip(_new[k][1], _first[k][1])), "the submitted World Cup rows are not the D033 rows"
_wsub = openpyxl.load_workbook("Edikted_Forecasting_Exercise_filled.xlsx", data_only=True)["Forecast"]; sub_total = sum(float(v) for r_ in _wsub.iter_rows(min_row=2, values_only=True) if r_[1] for v in r_[6:10])
_m = pd.read_parquet("data/monthly_item.parquet"); _m.columns = [str(c) for c in _m.columns]
WCROWS = ""
for k in WC:
    nm, o_ = _old[k]; _, n_ = _new[k]
    WCROWS += f'<tr><td rowspan="2">{nm}<br><span class="small">{k}</span></td><td class="r" rowspan="2">{it(_m.loc[k, "2026-06"])}</td><td>models, no cap</td>' + "".join(f'<td class="r">{it(v)}</td>' for v in o_) + f'<td class="r">{it(sum(o_))}</td></tr>'
    WCROWS += '<tr><td>submitted (hand cap)</td>' + "".join(f'<td class="r">{it(v)}</td>' for v in n_) + f'<td class="r">{it(sum(n_))}</td></tr>'
wc_now = sum(sum(v[1]) for v in _old.values()); wc_rule = sum(sum(v[1]) for v in _new.values())   # wc_now = the models' number, wc_rule = the capped, submitted number
GS = J["groupstore"]; _gp = {d["m"]: d for d in GS["per_store"]}; _gsig = {d["key"]: d for d in GS["sig"]}; _hind = {d["key"]: d for d in J["sig_hind"]}
_yy = [(_gp[m]["retail"] / _gp[q]["retail"], _gp[m]["stores"] / _gp[q]["stores"]) for m in _gp for q in [str(pd.Period(m, "M") - 12)] if q in _gp]   # same month a year earlier
FS = J["future_stores"]; _fc = {d["g"]: d for d in FS["cohorts"]}; SPC = J["spacing"]; _s4 = SPC["sets"][3:]; assert all(d["rules"] > d["wape"] for d in _s4)
IP = J["itempick"]; AL = J["alllow"]
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
TI = J["timeitem"]; BWS = next(d for d in J["bothways"]["systems"] if d["system"] == "Submitted system"); BWF = J["bothways"]["fifths"]; LSC = J["lossscale"]; assert "Tweedie" in LSC[4]["variant"] and "0.3" in LSC[1]["variant"] and TI["worst5"]["origins"] == 9
SU = J["surprise"]; _ss = {d["key"]: d for d in SU["sig"]}; MP = J["metapick"]; _mp = {d["variant"]: d for d in MP["rows"]}; assert all(d["wape"] > MP["rows"][0]["wape"] for d in MP["rows"][1:]) and MP["pca"]["load2"][0] > 0.8
JN = J["june"]; FI_ = J["flaginfo"]; EN = J["entropy"]; _en = {d["key"]: d for d in EN["sig"]}; assert JN["june"]["r3_abs"] < JN["others"]["r3_abs"][0] and JN["cohorts"][0]["g"].startswith("gained 2+") and JN["cohorts"][2]["g"].startswith("first month")
# item volume for the tables that list single items (user: "add an indication of the item volume"): R3 / 4 = the average of the last three months
_r26 = pd.read_parquet("data/fk_err_master_2026.parquet").set_index("style_color").R3
for _w in J["worst"]: _w["vol"] = round(_w["R3"] / 4, 1)
for _w in J["watch"]: _w["vol"] = round(_w["R3"] / 4, 1)
for _w in J["flaginfo"]["t26"]["top"]: _w["vol"] = round(float(_r26[_w["item"]]) / 4, 1)
_r2 = {d["system"]: d for d in J["r2"]["systems"]}; _r2o = [d for d in J["r2"]["by_origin"] if d["origin"] >= "2025-06"]; LMO = J["lastmonth"]; _lms = {d["key"]: d for d in LMO["sig"]}
assert LMO["rows"][5]["variant"] == "+ all last-month inputs" and sorted(_r2o, key=lambda d: d["change"])[0]["origin"] == "2025-06" and sorted(_r2o, key=lambda d: d["change"])[1]["origin"] == "2025-12"
# member refits D307-D312: table rows and tokens
_rf = {d["variant"]: d for d in J["refit"]["rows"]}; _rfu = {d["variant"]: d for d in J["refit_future"]["rows"]}; _rd = {d["variant"]: d for d in J["refit_decisions"]["rows"]}; _tr = {d["variant"]: d for d in J["trend"]["member"]}; _md = {d["variant"]: d for d in J["maxday"]["member"]}
def _mirow(lab, d, base=False):
    w = "" if base else str(d.get("wins13", ""))
    return f'<tr{" class=\'on\'" if base else ""}><td>{lab}</td><td class="r">{d["alone"]:.4f}</td><td class="r">{d["stack"]:.4f}</td><td class="r">{d["corrected"]:.4f}</td><td class="r">{pc(d["corrected_hit"], 1)}</td><td class="r">{w}</td></tr>'
_grp = lambda lab: f'<tr class="grp"><td colspan="6">{lab}</td></tr>'
MIROWS = _mirow("none (the submitted tree model)", _rf["ec_ns (refit, check)"], True) + _grp("Known on the forecast date") + _mirow("11 origin-known signals", _rf["+ origin-known signals"]) + _mirow("20 inputs from the last month", _rf["+ last-month inputs"]) \
    + _mirow("cluster and product-type trends", _tr["+ cluster and type trends"]) + _mirow("closeness to the 10 rising and 10 falling products", _tr["+ closeness to the rising and falling products"]) + _mirow("best sales day in recent history", _md["+ best-day inputs"]) \
    + _grp("Known futures (business decisions, exact plan)") + _mirow("own stores added and reached", _rd["+ stores added and reached (known future)"]) + _mirow("discount plan", _rd["+ the discount plan (known future)"]) \
    + _mirow("new colours of the style and same-type launches", _rfu["+ new colours of the style and same-type launches in the window (known future)"]) + _mirow("every business decision: stores, partners, launches, discount", _rd["+ every business decision: stores, partners, launches, discount (known future)"]) \
    + _mirow("every business decision and the 11 signals", _rd["+ every business decision and the origin-known signals"]) + _mirow("own stores, with the count off by about 2 stores", _rfu["+ the store counts, off by about 2 stores at the forecast date"])
for _k in ["stack", "corrected", "corrected_hit"]: assert all(_k in d for d in [_rf["+ origin-known signals"], _md["+ best-day inputs"]])
_s26 = J["trend"]["sets26"]; _hr = lambda L: "".join(f'<tr><td>{d["name"]}<br><span class="small">{d["item"]}</span></td><td>{d["ptype"]}</td><td class="r">{math.exp(d["trend"]):.1f}×</td></tr>' for d in L)
_la = {(d["variant"][:30], d["g"]): d for d in J["refit_decisions"]["launch"]}; _lv = "+ every business decision: sto"
_ir = {d["system"]: d for d in J["itemrel"]["systems"]}; _irs = _ir["Submitted system"]; _irf = J["itemrel"]["by_fifth"]; _irfl = {d["floor"]: d for d in J["itemrel"]["floor"]}
# proposal status lines (D336): owners from SYNC.md 'the 14 proposals' (D314); results of D293-D296 (Main) and D315-D318 (fork)
EX = J["expect26"]; LY = {(d["member"], d["mode"]): d for d in J["lyremoved"]["rows"]}; IV = J["intervals"]; CR = [d for d in J["confrule"]["rows"] if d["rule"].startswith("level 1 = narrowest")]; CT = J["confrule"]["targets"]["rule"]
_lo = IV["loo_summary"]; _rec = _lo["lower x1.0, upper x1.4"]
PS = {f"ps_{k}": v for k, v in {
  1: "Owner: Fork (data request, plan template). Tested as a known future: own stores 0.3752 and every business decision 0.3706 with corrections (D308–D310). Waiting for Edikted's 2026 plan: the request (D323, EDIKTED_DATA_REQUEST.md) and a plan template per target and month (D321, data/plan_template_2026.csv) are ready, and the plan engine turns a filled template into forecasts.",
  2: "Owner: Fork 2; data request by the Fork (D323). Tested in the per-location recommender: modelling partners apart from direct demand, or taking the realised partner orders as known, gives no measurable gain. Confirmed orders are in the data request.",
  3: "Owner: Main. Done as a no-op layer in the workbook build (D296): forecast capped at stock + inbound once the data exists.",
  4: "Owner: Fork 2 when the data arrives; in the data request (D323).",
  5: "Owner: Main when the data arrives; in the data request (D323).",
  6: "Owner: Fork. Done (D322): updating months 2–4 with the first month's sales gives 0.3687 against 0.3952 on recent origins (bias +2.3%), and 0.644 against 0.738 at the June 2025 origin. eda/month1_update_2026.py runs it on the July 2026 actuals.",
  7: "Owner: Fork. Done (D321, data/fk_launch_desk_2026.csv), with a correction to this proposal: the 4.9× / 2.3× / 1.0× multiples above mix demand into distribution. Holding demand fixed, full roll-out is 1.26× web only and partial 1.19× for the 154 young items with room for five or more doors. The 24 young targets already sit in 14–16 of the 16 doors, so their scenarios differ by 2%.",
  8: "Done. The World Cup cap is in the submitted workbook (D291, D303). Other dated tags among the targets (4th of July, Halloween, Homecoming) showed no bias that needs a rule (D295).",
  9: "Owner: Fork 2. Tested in the per-location recommender: with the stores each item actually reached taken as known, the recommender added to the five models gives 0.3902 against 0.3939 for the five alone, a small gain that needs Edikted's store plan.",
  10: f"Owner: Fork, then Main. Done for the tree member (D315): without last year's window it loses {J['lyremoved']['degradation']['ec_ns']['alone_ly']:.2f} alone and {J['lyremoved']['degradation']['ec_ns']['sys_ly']:.3f} in the system (WAPE ÷ R3, against a control refit). The 'sig' candidate is mixed: better on all 13 origins without last year ({LY[('sig', 'no LY')]['sys_all']:.4f} against {LY[('ec_ns', 'no LY')]['sys_all']:.4f}), worse on recent ones ({LY[('sig', 'no LY')]['sys_rec']:.4f} against {LY[('ec_ns', 'no LY')]['sys_rec']:.4f}). Main (D297, D351): seasonality pooled across items is level (0.3853 against 0.3851; 0.4103 against 0.4106 without last year), and dropping last-year inputs is worse, except the item's own last-year ratio, whose removal is level (0.3849) with a gain below the noise scale without last year (0.4097). Not applied. Every member (D324): without last year's same window each model alone gets worse on all 13 origins by +0.21 (tree), +0.33 (neural net) and +0.44 to +0.46 (TabPFN); in the system, removing it from one member at a time gives 0.4106 (tree), 0.4278 (neural net), 0.4432 and 0.4643 (the two TabPFN), and from every learned member 0.6902 with 30.4% within ±20%, against 0.3851. Much of the system's lead over R3 comes from last year repeating; the tree member is the most robust, TabPFN the least. The per-location recommender relies on it even more (0.48 → 0.99 alone, D340).",
  11: f"Owner: Fork computes, Main applies. Computed (D317): keep the lower side, multiply the upper log width by 1.4. Recent coverage {_lo['as stated']['inside']:.1%} → {_rec['inside']:.1%}, interval score level ({_lo['as stated']['interval_score']:.3f} → {_rec['interval_score']:.3f}); Applied in the submitted workbook (candidate_nx4, D299, D350).",
  12: f"Owner: Fork (a), Main (b, c). Confidence rule (D318): level 1 = narrowest fifth of the range + rising + sold last year. Levels 1 / 2 / 3 land within ±20% {CR[0]['hit_recent']:.1%} / {CR[1]['hit_recent']:.1%} / {CR[2]['hit_recent']:.1%} of the time; the 2026 targets fall {CT['1']} / {CT['2']} / {CT['3']}, the submitted workbook's confidence column since D350. Unit-weighted hit rate on the scoreboard (D294). 2026 totals with 80% ranges by product group, type and look (D320, data/fk_totals_2026.csv): in the backtest product groups land within ±20% 87% of the time, the largest third of types 88% and the largest looks 77%.",
  13: "Owner: Main. Done (D294): the scoreboard carries absolute error, the ratio to the seasonal naive, the ratio to R3 by season, the typical-item ratio (GMRAE) and origin-block intervals.",
  14: f"Owner: Fork (a), Main (b). Done: expectation frozen on the submitted workbook (D316; re-frozen on the submitted candidate_nx4, D319, md5 {EX['md5'][:8]}); early-origin gate on the scoreboard (D294).",
}.items()}
T = dict(
    built="18 Nov 2026",
    ex_md5=EX["md5"][:8], ex_total=it(EX["total"]), ex_abs=pc(EX["central"]["abs_wape"], 1), ex_r3=f2(EX["central"]["wape_r3"]), ex_bias=sp(EX["central"]["bias"]), ex_hit=pc(EX["central"]["hit20"]),
    ex_b_abs_lo=pc(EX["band100"]["abs_wape"][0]), ex_b_abs_hi=pc(EX["band100"]["abs_wape"][1]), ex_b_r3_lo=f2(EX["band100"]["wape_r3"][0]), ex_b_r3_hi=f2(EX["band100"]["wape_r3"][1]), ex_b_cov_lo=pc(EX["band100"]["cover"][0]), ex_b_cov_hi=pc(EX["band100"]["cover"][1]),
    **PS,
    ir_mean=f3(_irs["mean_rel"]), ir_g=f3(_irs["gmrae"]), ir_w=f3(_irs["wape_r3"]), ir_g13=f3(_irs["all13"]["gmrae"]), ir_r13=f3(_ir["R3 + rules"]["all13"]["gmrae"]), ir_g5=f3(_irfl[5]["gmrae"]),
    ir_f1=f3(_irf[0]["gmrae"]), ir_f5=f3(_irf[-1]["gmrae"]), ir_sig_w=f"{_ir['Submitted with the 11 signals (D307)']['wape_r3']:.4f} against {_irs['wape_r3']:.4f}", ir_sig_g=f"{_ir['Submitted with the 11 signals (D307)']['gmrae']:.4f} against {_irs['gmrae']:.4f}",
    mi_rows=MIROWS, mi_base_c=(lambda v: f'{v:.4f}')(_rf["ec_ns (refit, check)"]["corrected"]), mi_sig_c=(lambda v: f'{v:.4f}')(_rf["+ origin-known signals"]["corrected"]), mi_dec_c=(lambda v: f'{v:.4f}')(_rd["+ every business decision: stores, partners, launches, discount (known future)"]["corrected"]),
    mi_dec_h=pc(_rd["+ every business decision: stores, partners, launches, discount (known future)"]["corrected_hit"], 1), mi_decsig_c=(lambda v: f'{v:.4f}')(_rd["+ every business decision and the origin-known signals"]["corrected"]),
    mi_av_c=(lambda v: f'{v:.4f}')(_rd["+ decisions and availability in the window (partly a result of sales: a bound)"]["corrected"]),
    tr_hot_g=pc(J["trend"]["flagged"]["hot_growth"]), tr_cold_g=pc(J["trend"]["flagged"]["cold_growth"]), tr_hot_b=sp(J["trend"]["flagged"]["hot_bias"]), tr_cold_b=sp(J["trend"]["flagged"]["cold_bias"]), tr_hot_rows=_hr(_s26["hot"]), tr_cold_rows=_hr(_s26["cold"]),
    la_b5_0=sp(_la[(_lv, "stores added: 5 or more")]["bias_base"]), la_b5_1=sp(_la[(_lv, "stores added: 5 or more")]["bias"]), la_h5_0=pc(_la[(_lv, "stores added: 5 or more")]["hit_base"]), la_h5_1=pc(_la[(_lv, "stores added: 5 or more")]["hit"]),
    la_w_0=f3(_la[(_lv, "all launches under 3 months")]["wape_base"]), la_w_1=f3(_la[(_lv, "all launches under 3 months")]["wape"]),
    r2_r3_log=f2(_r2["R3"]["log"]), r2_rules_chg=f2(_r2["R3 + rules"]["change"]), r2_sys_chg=f2(_r2["Submitted system"]["change"]), r2_o_lo=f2(min(d["change"] for d in _r2o)), r2_o_hi=f2(max(d["change"] for d in _r2o)),
    lm_n=str(LMO["n_inputs"]), lm_w_all=f"{LMO['rows'][5]['wape']:.4f}", lm_w_ref=f"{LMO['rows'][1]['wape']:.4f}", lm_auc=f2(LMO["flag_lm"]["auc"]), lm_rho_ds=sf(_lms["lm_days_since"]["rho"], 2), lm_same_ds=str(_lms["lm_days_since"]["same"]),
    lm_rho_par=sf(_lms["lm_sh_par"]["rho"], 2), lm_rho_r7=sf(_lms["lm_r7"]["rho"], 2), lm_rho_r28=sf(_lms["f_r28"]["rho"], 2), lm_rho_tr=sf(_lms["trend91"]["rho"], 2),
    sub_total=it(sub_total),
    jn_sys=pc(JN["june"]["sys_abs"], 1), jn_sys_lo=pc(JN["others"]["sys_abs"][0], 1), jn_sys_hi=pc(JN["others"]["sys_abs"][1], 1), jn_r3=pc(JN["june"]["r3_abs"], 1), jn_r3_lo=pc(JN["others"]["r3_abs"][0], 1), jn_r3_hi=pc(JN["others"]["r3_abs"][1]),
    jn_split=pc(JN["june"]["r3_split"]), jn_split_med=pc(JN["others"]["r3_split"][2]), jn_closer=pc(JN["june"]["closer"]), jn_closer_lo=pc(JN["others"]["closer"][0]), jn_closer_hi=pc(JN["others"]["closer"][1]), jn_growth=pc(JN["june"]["growth"] - 1),
    jn_st_err=pc(JN["cohorts"][0]["err"]), jn_st_bias=sp(JN["cohorts"][0]["bias"]), jn_m1_err=pc(JN["cohorts"][2]["err"]), jn_m1_bias=sp(JN["cohorts"][2]["bias"]), jn_hit=pc(JN["june"]["hit"]),
    jn_hit_lo=pc(min(d["hit"] for d in JN["by_origin"][-4:])), jn_hit_hi=pc(max(d["hit"] for d in JN["by_origin"][-4:])),
    fi_n=str(FI_["n_inputs"]), fi_top=pc(FI_["base"]["top20_rate"]), fi_low=pc(FI_["base"]["low20_rate"]), fi_auc_nr=f2(FI_["no_range"]["auc"]), fi26_n=str(FI_["t26"]["flagged"]), fi26_all=str(FI_["t26"]["n"]), fi26_units=pc(FI_["t26"]["flagged_units"]),
    fi26_fy=str(FI_["t26"]["flagged_young"]), fi26_young=str(FI_["t26"]["young"]),
    en_size=sf(_en["ent_d91"]["rho_size"], 2), en_dir=sf(_en["ent_d91"]["rho"], 2), en_w_ent=f"{EN['rows'][2]['wape']:.4f}", en_w_peak=f"{EN['rows'][3]['wape']:.4f}", en_w_both=f"{EN['rows'][4]['wape']:.4f}", en_w_ref=f"{EN['rows'][1]['wape']:.4f}", en_auc=f2(EN["flag_ent"]["auc"]),
    su_n=str(SU["n_inputs"]), su_rho1=sf(_ss["s1"]["rho"], 2), su_rho_tr=sf(_ss["trend91"]["rho"], 2), su_w=f3(SU["rows"][3]["wape"]), su_ref_w=f3(SU["rows"][1]["wape"]), su_auc1=f2(SU["flag_sur"]["auc"]),
    mp_base=f3(MP["rows"][0]["wape"]), mp_acc_hi=pc(_mp["gate, high-level features only: its first choice"]["acc"]), mp_acc_pos=pc(_mp["gate, high-level features and the models' positions: its first choice"]["acc"]),
    mp_gate_pick=f3(_mp["gate, high-level features only: its first choice"]["wape"]), mp_gate_soft=f3(_mp["gate, high-level features only: weights = its probabilities"]["wape"]),
    mp_bayes=f3(next(d for d in MP["rows"] if d["variant"].startswith("Bayes"))["wape"]), mp_pc1=pc(MP["pca"]["share"][0]), mp_pc2=pc(MP["pca"]["share"][1]), mp_pc_rest=pc(sum(MP["pca"]["share"][2:])),
    mp_pca_lo=f3(min(d["wape"] for d in MP["rows"] if d["variant"].startswith("PCA: the actual"))), mp_pca_hi=f3(max(d["wape"] for d in MP["rows"] if d["variant"].startswith("PCA: the actual"))),
    bw_ratio=f3(BWS["ratio"]), bw_med_ratio=f3(BWS["med_ape_ratio"]), bw_top5_units=pc(BWS["top5_units"]), bw_top5_pct=pc(BWS["top5_pct"]), bw_f5_units_err=pc(BWF[4]["err_units"]), bw_f1_pct_err=pc(BWF[0]["err_pct"]), bw_f1_units=pc(BWF[0]["units"], 1),
    ls_base_w=f3(LSC[0]["ratio"]), ls_base_hit=pc(LSC[0]["hit"], 1), ls_unit_w=f3(LSC[4]["ratio"]), ls_unit_hit=pc(LSC[4]["hit"], 1), ls_w03_w=f3(LSC[1]["ratio"]),
    ti_time=pc(TI["time"]), ti_item=pc(TI["item"]), ti_item_ov=pc(TI["item_overlap"]), ti_under_lo=pc(TI["under_range"][0]), ti_under_hi=pc(TI["under_range"][1]), ti_bias_lo=sp(TI["bias_range"][0]), ti_bias_hi=sp(TI["bias_range"][1]),
    ti_tot_w=f3(TI["oracles"][0]["wape"]), ti_tot_hit=pc(TI["oracles"][0]["hit"], 1), ti_type_w=f3(TI["oracles"][2]["wape"]), ti_pc25=f3(TI["past_corr"][0]["wape"]), ti_pc50=f3(TI["past_corr"][1]["wape"]),
    ti_w5_n=it(TI["worst5"]["n"]), ti_w5_items=str(TI["worst5"]["items"]), ti_top5_n=str(TI["top_items"][1]["items"]), ti_top5_err=pc(TI["top_items"][1]["err"]), ti_top5_units=pc(TI["top_items"][1]["units"]),
    hl_small_units=pc(sum(d["units"] for d in HT["cohort_low"][:3]), 1),
    ip_nohist=pc(IP["no_history"]), ip_med=str(int(IP["med_history"])), ip_persist=pc(IP["persist"]), ip_share_lo=pc(min(IP["best_share"].values())), ip_share_hi=pc(max(IP["best_share"].values())), ip_corr=f2(IP["member_corr"]),
    al_share=pc(AL["all_low"]["share"]), al_err=pc(AL["all_low"]["err"]), al_over_f=f2(AL["all_low"]["med_over_F"]), al_over_top=f2(AL["all_low"]["med_over_top"]),
    al_m1=pc(AL["all_low"]["m1_up"]), al_m1_r=pc(AL["rest"]["m1_up"]), al_st2=pc(AL["all_low"]["stores2"]), al_st2_r=pc(AL["rest"]["stores2"]), al_par=pc(AL["all_low"]["partners1"]), al_par_r=pc(AL["rest"]["partners1"]),
    al_young=pc(AL["all_low"]["young"]), al_young_r=pc(AL["rest"]["young"]), al_in=pc(AL["all_low"]["in_range"]), al_in_r=pc(AL["rest"]["in_range"]), al_auc=f2(AL["clf"]["auc"]),
    al_cut_all=pc(AL["info"]["all_cut"]), al_cut_plan=pc(AL["info"]["plan_cut"]), al_hit_all=pc(AL["info"]["hit_all"]), al_hit_ref=pc(AL["info"]["hit_ref"]), al_top_err=pc(AL["clf"]["top20_err"]), al_top_rate=pc(AL["clf"]["top20_rate"]),
    al_fac_lo=f2(min(f["factor"] for f in AL["uplift"]["factors"])), al_fac_hi=f2(max(f["factor"] for f in AL["uplift"]["factors"])), al_up_bias0=sp(AL["uplift"]["bias_base"]), al_up_bias=sp(AL["uplift"]["bias"]),
    al_up_w=f3(AL["uplift"]["wape"]), al_up_w0=f3(AL["uplift"]["wape_base"]), al_up_n=str(len(AL["uplift"]["origins"])),
    fs_w=f3(FS["rows"][2]["wape"]), fs_ref_w=f3(FS["rows"][1]["wape"]), fs_hit=pc(FS["rows"][2]["hit"], 1), fs_ref_hit=pc(FS["rows"][1]["hit"], 1), fs_noisy_w=f3(FS["rows"][5]["wape"]),
    fs_b5_ref=sp(_fc["items that gained 5+ stores"]["bias_ref"]), fs_b5=sp(_fc["items that gained 5+ stores"]["bias_k"]), fs_l_ref=f3(_fc["launches under 3 months"]["wape_ref"]), fs_l=f3(_fc["launches under 3 months"]["wape_k"]),
    fs_web_bias=pc(FS["web_bias_add2"]), fs_all_bias=pc(FS["bias_add2"]),
    sp_gap_lo=f2(min(d["rules"] - d["wape"] for d in _s4)), sp_gap_hi=f2(max(d["rules"] - d["wape"] for d in _s4)), sp_hit_lo=pc(min(d["hit"] for d in _s4)), sp_hit_hi=pc(max(d["hit"] for d in _s4)),
    sp_hr_lo=pc(min(d["hit_r3"] for d in _s4)), sp_hr_hi=pc(max(d["hit_r3"] for d in _s4)), sp_n4=str(sum(d["wins4"] == 4 for d in SPC["steps"])), sp_n3=str(sum(d["wins4"] == 3 for d in SPC["steps"])),
    sp_tree_wins=str(next(d for d in SPC["steps"] if "D253" in d["step"])["wins4"]),
    gs_nr=str(GS["n_rel"]), gs_r_w=f3(GS["rows"][6]["wape"]), gs_r_hit=pc(GS["rows"][6]["hit"], 1),
    gs_ng=str(GS["n_group"]), gs_ns=str(GS["n_store1"]), gs_rho_yoy=sf(_gsig["g_tl_yoy"]["rho"], 2), gs_same_yoy=str(_gsig["g_tl_yoy"]["same"]), gs_g_w=f3(GS["rows"][2]["wape"]), gs_ref_w=f3(GS["rows"][1]["wape"]),
    gs_g_hit=pc(GS["rows"][2]["hit"], 1), gs_ref_hit=pc(GS["rows"][1]["hit"], 1), gs_ci_lo=sf(GS["rows"][2]["ci"][0], 4), gs_ci_hi=sf(GS["rows"][2]["ci"][1], 4), gs_rho_open=sf(_gsig["expo_future"]["rho"], 2), gs_s_w=f3(GS["rows"][4]["wape"]),
    gs_yoy_sales=_r(float(np.median([v[0] for v in _yy])), 1), gs_yoy_stores=_r(float(np.median([v[1] for v in _yy])), 1), gs_rho_hind=sf(_hind["h_stores_chg"]["rho"], 2),
    ce_per=it(CE["cells"]["per_origin"]), ce_med=it(CE["cells"]["med_items"]), ce_single=pc(CE["cells"]["single"]), ce_then=pc(CE["rescale"][0]["hit"]), ce_full=pc(CE["rescale"][1]["hit"]),
    ce_feat_w=f3(CE["feature"]["with_cells"]["wape"]), ce_ref_w=f3(CE["feature"]["ref"]["wape"]), ce_n10=pc(NZ[0.1]), ce_n15=pc(NZ[0.15]), ce_n20=pc(NZ[0.2]), ce_n30=pc(NZ[0.3]), ce_sd_w=pc(CE["cells"]["log_sd_w"]),
    wc_rows=WCROWS,
    tc_top_level=it(TC["top5"]["med_level"]), tc_all_level=it(TC["all"]["med_level"]), tc_top_err_units=it(TC["top5"]["med_err_units"]), tc_all_err_units=it(TC["all"]["med_err_units"]),
    tc_wape_fixed=f3(TC["wape_if_top5_fixed"]), tc_hit_fixed=pc(TC["hit_if_top5_fixed"]), tc_small_m=pc(TC["small3"]["misses"]), tc_small_e=pc(TC["small3"]["err"]),
    sv_wmin=f2(min(v for k, v in SV["weights"][-1].items() if k != "origin")), sv_wmax=f2(max(v for k, v in SV["weights"][-1].items() if k != "origin")),
    sv_hit_tight=pc(SV["spread_hit"][0]["hit"]), sv_hit_wide=pc(SV["spread_hit"][-1]["hit"]), sv_bound_hit=pc(next(d for d in SV["rows"] if "bound" in d["variant"])["hit"]),
    ln_arrivals=it(LN["cross_store_arrivals"]), ln_stores=str(LN["n_stores"]), ln_partners=str(LN["n_partners"]), ln_feats=str(len(LN["sig_all"])),
    ln_rho_all=sf(LN["sig_all"][0]["rho"], 2), ln_rho_gap=sf(LN["sig_gap"][0]["rho"], 2), ln_corr=f3(LN["correction"][2]["wape"]), ln_ref=f3(LN["correction"][1]["wape"]),
    ln_ci_lo=sf(LN["boot_launch_vs_ref"][0], 4), ln_ci_hi=sf(LN["boot_launch_vs_ref"][1], 4), ln_auc0=f2(LN["rollout"]["auc_base"]), ln_auc1=f2(LN["rollout"]["auc_launch"]),
    ln_cross=sf(LN["cross_store"]["corr"], 2), ln_u14=it(LN["cross_store"]["med_u14"]), ln_ss=f2(LN["carry"]["store_store"]), ln_wrate=f2(LN["carry"]["web_vs_rate_per_store"]), ln_wt=f2(LN["carry"]["web_tiktok"]),
    ln_webfirst=pc(LN["carry"]["web_first"]), ln_lead=it(LN["carry"]["lead_median"]), ln_b_none=pc(LN["rollout"]["by_gap"][0]["bias"]), ln_b_half=pc(LN["rollout"]["by_gap"][1]["bias"]),
    ln_u_part=pc(LN["rollout"]["by_gap"][0]["units"] + LN["rollout"]["by_gap"][1]["units"]),
    hit_items=pc(HT["overall"]["items"]), hit_units=pc(HT["overall"]["units"]), hit_targets=pc(HT["overall"]["targets"]), hit_r3=pc(H["recent"]["hit_r3"]), med_ape=pc(LS["median_ape"], 1),
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
    hit_plan=pc(HT["info"]["plan"]), hit_plan_early=pc(HT["info"]["plan_early"]), lvl_look=pc(LV["look (cart + content cluster)"]["hit"]), lvl_type=pc(LV["product type"]["hit"]), lvl_group=pc(LV["product group"]["hit"]), fleet_first=str(J["fleet"][0]["stores"]), fleet_last=str(J["fleet"][-1]["stores"]), n_all=it(H["all"]["n"]), ratio_recent=f3(H["recent"]["ratio"]), ratio_all=f3(H["all"]["ratio"]), abs_recent=pc(H["recent"]["abs"]), r3abs_recent=pc(H["recent"]["r3abs"]),
    top5=pc(H["top_share"]["5"]), top1=pc(H["top_share"]["1"]), top20=pc(H["top_share"]["20"]), neff13=f"{J['neff']['all13']:.0f}", neff9=f"{J['neff']['recent9']:.0f}",
    abs_min=pc(min(o["abs"] for o in REC))[:-1], abs_max=pc(max(o["abs"] for o in REC)), r3_min=pc(min(o["r3abs"] for o in REC))[:-1], r3_max=pc(max(o["r3abs"] for o in REC)),
    sys_vs_sea_recent=f2(J["seasonal_naive"]["system_vs_seasonal_recent"]), sys_vs_sea_all=f2(J["seasonal_naive"]["system_vs_seasonal_all"]),
    an_ratio=f2(an["ratio"]), an_abs=pc(an["abs"]), an_bias=sp(an["bias"]), an_w20=pc(an["within20"]), an_cover=pc(an["cover"]),
    tot_known=f3(J["level_vs_alloc"]["total_known"]), grp_known=f3(J["level_vs_alloc"]["group_known"]),
    under_err=pc(H["under_err_share"]), all_under=pc(H["all_under_err"]), same_side=pc(H["all_under_err"] + H["all_over_err"]),
    n_axes=str(len(co)), best_th=sf(min(c["th_gain"] for c in co.values()), 4), n_axes_worse=str(sum(c["th_gain"] > 0 for c in co.values())),
    yg_big_growth=f"{yd['stores added: 5 or more']['growth']:.1f}", yg_mid_growth=f"{yd['stores added: 2 to 4']['growth']:.1f}", yg_flat_growth=f"{yd['stores about flat']['growth']:.1f}",
    yg_big_bias=pc(yd["stores added: 5 or more"]["bias"]), yg_big_bias_s=sp(yd["stores added: 5 or more"]["bias"]), yg_flat_bias=pc(yd["stores about flat"]["bias"]),
    yg_par_growth=f"{yd['partners added: 2 or more']['growth']:.1f}", yg_par_bias_s=sp(yd["partners added: 2 or more"]["bias"]),
    voi_dist=f3(-voi["distribution"]["delta"]), voi_m1=f3(-voi["early read"]["delta"]), voi_promo=f3(-voi["promotion depth"]["delta"]), voi_avail=f3(-voi["availability"]["delta"]),
    voi_all=f3(-voi["plannable blocks + early read"]["delta"]), voi_ref=f3(J["voi"][0]["wape"]),
    ls_same_med=sf(float(np.median([d["no_same"] for d in S])), 2), ls_same_max=sf(max(d["no_same"] for d in S), 2), ls_other_med=sf(float(np.median([d["other"] for d in S])), 2),
    variants=str(J["multiplicity"]["variant_files"]), decisions=str(J["multiplicity"]["decisions"]), sd_item=f"{J['multiplicity']['sd_item']:.4f}",
    null_max=f3(J["multiplicity"]["null_max"]), null20=f3(J["multiplicity"]["null_max_20"]), tgt_bias=sp(grp("target", "target")["bias"]),
    wc_now=it(wc_now), wc_rule=it(wc_rule), wc_diff=it(round((wc_now - wc_rule) / 100) * 100), wc_pct=pc((wc_now - wc_rule) / float(c26.total.sum()), 1), wc_jul=it(c26.loc[WC[0], "jul"]), wc_aug=it(c26.loc[WC[0], "aug"]),
    w20=pc(H["within20"]), w20_r3=pc(H["within20_r3"]), w50=pc(H["within50"]), w50_r3=pc(H["within50_r3"]), b2x=pc(H["beyond2x"]), b2x_r3=pc(H["beyond2x_r3"]), b2x_err=pc(H["beyond2x_err"]),
    metric_corr=sf(J["confounds"]["metric_corr"], 2), tt_extra=pc(J["channel_growth"]["big_under"]["tt"]), tt_base=pc(J["channel_growth"]["big_under_base"]["tt"]),
    st_add_bias=pc(sc["add2"]["bias"]), st_add_web=pc(sc["add2"]["web_bias"]), st_flat_bias=pc(sc["flat"]["bias"]), st_rho_web=sf(sc["rho_web_growth"], 2), st_rho_res=sf(sc["rho_resid"], 2),
    chain_from=f3(ch[0]["frm"]), tab_early=sf(ch[3]["d_early"]), arith_early=sf(ch[4]["d_early"]), first_early=sf(ch[0]["d_early"]),
    alive=it(J["confounds"]["alive_at_end"]), n_cat=it(J["confounds"]["n_catalogue"]),
    tg_growth=sp(np.expm1(J["confounds"]["target_growth"]["target"])), ot_growth=sp(np.expm1(J["confounds"]["target_growth"]["other"])), tg_top100=str(J["confounds"]["target_rank"]["in_top100"]),
    cens_units=pc(cz["units_flagged"], 1), cens_over=pc(cz["flagged_share_of_over_err"], 1),
    items_first=it(mix[0]["n"]), items_last=it(mix[-1]["n"]), stores_first=it(mix[0]["stores"]), stores_last=it(mix[-1]["stores"]), tt_first=pc(mix[0]["tiktok"]), tt_last=pc(mix[-1]["tiktok"]),
    big_n=pc(J["big"]["n_share"]), big_err=pc(J["big"]["err_share"]), big_under=pc(J["big"]["under_err"]), big_over=pc(J["big"]["over_err"]),
    u_store=pc(cu["own stores added (2+)"]["err_share"]), u_m1=pc(cu["month 1 already 1.5x the forecast pace"]["err_share"]),
    o_none=pc(cov["none of the first seven tags"]["err_share"]), o_m1=pc(cov["month 1 already under 0.67x the forecast pace"]["err_share"]),
    t_n=str(J["targets26"]["n"]), t_total=it(J["targets26"]["total"]), t_r3=it(J["targets26"]["r3"]), t_far=pc(fl["forecast far above R3"]["units"]),
    t_young=pc(fl["young (under 6 months)"]["units"]), t_tt=pc(fl["TikTok 15% +"]["units"]), t_exp=pc(J["targets26"]["exp_abs"]),
    young_err=pc(grp("age", "under 3 months")["err_share"]), young_units=pc(grp("age", "under 3 months")["unit_share"]),
    rho_m1=sf(next(d for d in J["sig_hind"] if d["key"] == "h_m1r")["rho"], 2),
    cov_all=pc(J["coverage"]["overall"]), cov_above=pc(J["coverage"]["above"]), cov_below=pc(J["coverage"]["below"]), widen13=pc(next(d for d in J["analogue"]["widen"] if d["k"] == 1.3)["cover"]),
)
V = [["w", "holds on recent", f"Loses on the early four ({sf(ch[0]['d_early'])}): with under a year of history the rule agents were better."],
     ["g", "holds everywhere", f"Wins {ch[1]['wins13']} of 13 origins; the origin-block interval stays below zero."],
     ["w", "small, consistent", f"Inside the selection-noise band, but it wins {ch[2]['wins13']} of 13 origins, the early ones included."],
     ["w", "holds on recent", f"Reverses on the early four ({sf(ch[3]['d_early'])})."],
     ["w", "holds on recent", f"Reverses on the early four ({sf(ch[4]['d_early'])})."],
     ["b", "not established", "The origin-block interval crosses zero and the step is inside the selection-noise band."],
     ["w", "small, consistent", f"Wins {ch[6]['wins9']} of 9 recent origins; larger on the early four ({sf(ch[6]['d_early'])})."],
     ["g", "holds on recent", "Not applied on the early four: there are no earlier outcomes to learn from."]]
t = open("eda/error_study_template.html", encoding="utf-8").read()
need = set(re.findall(r"\{\{(\w+)\}\}", t)); miss = need - set(T); assert not miss, f"missing tokens: {miss}"
for k in need: t = t.replace("{{" + k + "}}", T[k])
assert t.count("__DATA__") == 1 and t.count("__VERDICTS__") == 1
html = t.replace("__VERDICTS__", json.dumps(V, ensure_ascii=False)).replace("__DATA__", json.dumps(J, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
open("eda/error_study.html", "w", encoding="utf-8").write(html)
wrap = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0;font:14px system-ui}</style></head><body>'
open("eda/error_study_preview.html", "w", encoding="utf-8").write(wrap + html + "</body></html>")
print("wrote eda/error_study.html", len(html) // 1024, "KB;", len(need), "tokens")
for k in sorted(T): print(f"  {k} = {T[k]}")
