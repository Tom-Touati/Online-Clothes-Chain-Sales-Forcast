"""Fork 2, candidate nx6 (overnight goal; D389n): the new nx5 (eda/181l, the submission) with the confidence column by the hybrid rule of D389k:
level 1 by the D318 rule (narrowest fifth of the range + rising + sold last year), level 3 where the item's calibrated ranges give less than
even odds of landing within +/-50% (p50 < 0.5, CDF interpolated in logs through P10 / P25 / P50 / P75 / P90), the hand-rule rows at 3, else 2.
Backtest, claims met by the scale's own definitions: held out 0.750 -> 0.813, Jul-Oct 25 0.699 -> 0.742, Mar-Jun 26 0.755 -> 0.838.
Writes Edikted_Forecasting_Exercise_candidate_nx6.xlsx and data/final_2026_nx6_cascade*.csv. Everything else as eda/181l (header below).
Fork 2 (user, 18 Nov 03:30: "make the new system nx5 with the new calibration and routing for new vs old items. prepare the final submission
totally"): the new nx5 = D361's nx5 (Huber tree, NN, rules, both TabPFN members, rc_deepfm at a quarter weight) with
  routing    the tree member for items under 6 months old at 2026-06 is the Huber tree trained with new items' rows x5 (D389h; eda/187
             huber_new5), the plain Huber tree for the rest -> data/q_backtest_ls_huber_routed_2026.parquet (307 of 1,423 items routed)
  corrections re-fitted on the routed system (eda/218_base_nx5r.py, 179x2 / 179b on the CPU server, eda/218c_geo_nx5r.py ->
             data/geo_nx5r_x_2026.parquet; backtest recent 0.3806, held out 0.2926, all 13 0.4357)
  ranges     the D329 per-segment method refitted on this system's own backtest errors with Jul-Oct 2025 and Mar-Jun 2026 weighted x4
             (eda/219_ranges_nx5r.py -> data/ranges_nx5r_2026.csv) for P10 / P25 / P75 / P90 (replaces the D329 ratios and the D370b width model)
  text       the Model explanation adds the data artefacts (EDA_REPORT section 4) and the routing / range calibration.
The D361 nx5 files are kept as *_nx5_d361*. Writes Edikted_Forecasting_Exercise_candidate_nx5.xlsx and data/final_2026_nx5_cascade*.csv.
D369 header follows.
D369: P25 / P75 from the fork's width model (data/fk_ranges_2026_wm.csv, D370b; purged backtest pinball 0.3244 vs 0.3316, honest 49.4% vs 48.0%
inside), P10 / P90 stay on the D329 per-segment ratios; P25 >= P10 and P75 <= P90 enforced. A "Ranges (totals)" sheet: the fork's studentised
conformal rule (D371: item half-ranges in quadrature plus the shared origin term c) on the workbook totals of the 100 targets, by product type and style.
D368: ranges from the fork's per-segment calibration (data/fk_ranges_2026.csv, D329 / D366) for P10 / P25 / P75 / P90.
D360 (user, in Fork 2's session: "build the candidate with the quarter-weight recommender"): as eda/181j (D357) with rc_deepfm at a quarter
of one member's weight (D358: the only recommender weight that passes the last-year-removed gate) and the 2026 corrections re-fitted for that
weighted system (data/geo_ecnsnnhubrq_x_2026.parquet; backtest 0.3811 x R3). Writes Edikted_Forecasting_Exercise_candidate_nx5.xlsx.
D357 header follows.
D357: as eda/181i (D299) with the D353 Huber tree member (data/srv4/q_backtest_ls_huber_2026.parquet) and Fork 2's location recommender
(rc_deepfm at 2026-06, data/rc_recsys_preds.parquet) as a sixth member, the corrections re-fitted on that six-member system
(data/geo_ecnsnnhubrc_x_2026.parquet; backtest 0.3786 x R3, D355). Writes Edikted_Forecasting_Exercise_candidate_nx5.xlsx.
D299 header follows.
D299: as eda/181h (D291) plus (a) the fork's D318 confidence rule (#12b): level 1 = narrowest fifth of the conformal TabPFN P10-P90 log width
among all 1,423 items at 2026-06 AND rising 91-day trend AND sold in the same months last year, level 3 = widest fifth, else 2; the hand-rule
rows (World Cup, out of stock) stay at 3 (data/fk_confidence_2026.csv); (b) the fork's D317 ranges (#11): P10 = final x q10 / q50, P90 = final
x (q90 / q50)^1.4 (upper log width x 1.4), on the workbook total after the hand rules; (c) optional tree member from env MEMBER (a
data/q_backtest_nx_<MEMBER>_2026.parquet q50 column) with its corrections (env GEO, a data/geo_*_2026.parquet), defaults = the submitted ones.
Writes Edikted_Forecasting_Exercise_candidate_nx4.xlsx and data/final_2026_nx4_cascade*.csv.
D291 header follows.
D291: as eda/181g (D273) plus the D033 World Cup rule restored on S24840_LIGHT-GRAY-MELANGE, S25039_YELLOW and S24847_NAVY (user, in the fork's
session: "Yes, restore"): their monthly units are the first submission's D033 values (July = late-June pace to the 19 July final, Aug-Oct
capped at 75 / 50 / 45% of July), confidence 3, the model's own total kept in the "Model raw total" column; buying = the rule total x the
system's own buying / forecast ratio for the item. Writes Edikted_Forecasting_Exercise_candidate_nx3.xlsx and data/final_2026_nx3_cascade*.csv.
D273 header follows.
D273: as eda/181f (D262) with the NN member replaced by the fork's D272 NN on the D252 event + launch-curve inputs (finalSec + finalSec_s10)
and the corrections re-fitted on that system (data/geo_ecnsnn_x_2026.parquet; backtest 0.3851 x R3's error). Writes Edikted_Forecasting_Exercise_candidate_nx2.xlsx.
D262 header follows.
D262: the July-October 2026 candidate on the D253-D259 system (user, 18 Nov: build the 2026 candidate on the new system), as a cascade
like eda/179f (D235). A NEW workbook, Edikted_Forecasting_Exercise_candidate_nx.xlsx; the submitted Edikted_Forecasting_Exercise_filled.xlsx
and the earlier candidate (candidate_final) are never touched.
Members at the June 2026 origin (4-month totals): rules, the new tree member (D253: D198 + 15 more events + launch curves, no static
descriptors, refitted at 2026-06 on every origin with a known outcome; data/q_backtest_nx_ec_ns_2026.parquet), NN, conformal TabPFN
(pf2c) and TabPFN dyn 32 (pf2d32).
  avg5      = arithmetic mean of the five members at P50                                                    (E:ec_ns, D253)
  forecast  = avg5 x the time-honest corrections re-fitted on all 13 backtest origins: geometric mean of the volume x age x discount
              segment re-allocation (D245) and the breakout / falling-item residual model (D247 + D259 signals), each origin's total
              kept (data/geo_ec_ns_x_2026.parquet; backtest 0.3871 x R3's error, D259)
  buying    = arithmetic mean of rules, the volume-weighted new tree member at P65 (D260), NN, pf2c and pf2d32 at P65 (u65ec, 10.36%)
Monthly split, confidence, the D034 override and the explanation layer as eda/179f.
Writes data/final_2026_nx_cascade.csv, data/final_2026_nx_cascade_hierarchy.csv and Edikted_Forecasting_Exercise_candidate_nx.xlsx."""
import sys, os, shutil, numpy as np, pandas as pd, openpyxl
sys.stdout.reconfigure(encoding="utf-8")
TEST = os.environ.get("FINAL_TEST") == "1"   # dry run: missing fork members replaced by qdyn, outputs to *_TEST files
WB_IN, WB_OUT = "Edikted_Forecasting_Exercise.xlsx", ("Edikted_Forecasting_Exercise_candidate_nx6_TEST.xlsx" if TEST else "Edikted_Forecasting_Exercise_candidate_nx6.xlsx")
assert WB_OUT not in ("Edikted_Forecasting_Exercise_filled.xlsx", "Edikted_Forecasting_Exercise_candidate_final.xlsx", "Edikted_Forecasting_Exercise_candidate_nx.xlsx", "Edikted_Forecasting_Exercise_candidate_nx2.xlsx", "Edikted_Forecasting_Exercise_candidate_nx3.xlsx", "Edikted_Forecasting_Exercise_candidate_nx4.xlsx")
wb0 = openpyxl.load_workbook(WB_IN); ws0 = wb0["Forecast"]
TG = [r[1].value for r in ws0.iter_rows(min_row=2) if r[1].value]; print("targets", len(TG))
it = pd.read_parquet("data/items.parquet").set_index("style_color")
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
R3 = 4 * M.reindex(TG)[["2026-04", "2026-05", "2026-06"]].mean(1)
F = pd.read_parquet("data/features.parquet"); F = F[F.origin == "2026-06"]
SEA = F.assign(st=F.m3 * np.exp(F.lfl_type.fillna(0))).groupby("style_color").st.sum().reindex(TG).fillna(R3)
RU = pd.read_csv("agent_eval/2026-06/predictions_verified.csv", dtype={"month": str}).pivot(index="style_color", columns="month", values="pred").reindex(TG)
RU.columns = ["jul", "aug", "sep", "oct"]
NNm = sum(pd.read_parquet(f"data/{f}.parquet").pivot(index="style_color", columns="h", values="q50") for f in ["nn_backtest_preds_finalSec", "nn_backtest_preds_finalSec_s10"]) / 2
NNm = NNm.reindex(TG); NNm.columns = ["jul", "aug", "sep", "oct"]
fk = lambda m: pd.read_parquet(f"data/fk_2026_{m}.parquet" if (os.path.exists(f"data/fk_2026_{m}.parquet") or not TEST) else "data/fk_2026_qdyn.parquet").drop_duplicates("style_color").set_index("style_color").reindex(TG)
QD, PC, PD, W3 = fk("qdyn"), fk("pf2c"), fk("pf2d32"), fk("w03")
C = pd.DataFrame(index=TG); C["R3"] = R3; C["season"] = SEA
NX = pd.read_parquet(os.environ.get("MEMF", "data/q_backtest_ls_huber_routed_2026.parquet")).drop_duplicates("style_color").set_index("style_color").reindex(TG)
W3E = pd.read_parquet("data/q_backtest_nx_w03ec_2026.parquet").drop_duplicates("style_color").set_index("style_color").reindex(TG)
C["rules"] = RU.sum(1); C["qnx"] = NX.q50; C["nn"] = NNm.sum(1); C["pf2c"] = PC.q50; C["pf2d32"] = PD.q50; C["qdyn_prev"] = QD.q50
for c in ["rules", "nn"]: C[c] = C[c].fillna(C[["qnx", "pf2c", "pf2d32"]].mean(1))
MEM = ["rules", "qnx", "nn", "pf2c", "pf2d32"]
_rc = pd.read_parquet("data/rc_recsys_preds.parquet"); _rc = _rc[_rc.origin == "2026-06"].drop_duplicates("style_color").set_index("style_color").rc_deepfm
C["rc"] = _rc.reindex(TG).values; C["rc"] = C.rc.fillna(C[MEM].mean(1)); MEM = MEM + ["rc"]   # D357: sixth member, filled as eda/181a
_W = np.array([1.0] * (len(MEM) - 1) + [0.25]); _X = C[MEM].clip(lower=0).values; _ok = np.isfinite(_X)   # D360: rc_deepfm at a quarter weight
C["avg5"] = np.where(_ok.any(1), np.nansum(np.where(_ok, _X, 0) * _W, 1) / np.maximum((_ok * _W).sum(1), 1e-9), np.nan)          # NaN-aware, like the plain mean
GEO = pd.read_parquet(os.environ.get("GEO", "data/geo_nx5r_x_2026.parquet")).set_index("style_color").reindex(TG)
assert np.allclose(GEO.F.dropna(), C["avg5"][GEO.F.notna()], rtol=1e-6), "stack differs from the correction base"
C["correction"] = (GEO.geo / GEO.F.clip(lower=1e-9)).fillna(1.0); C["forecast"] = C["avg5"] * C.correction
C["p2c32a_prev"] = C[["rules", "qdyn_prev", "nn", "pf2c", "pf2d32"]].clip(lower=0).mean(1)
QB = "q65"   # D260: u65ec, 10.36%
C["buying"] = pd.concat([C.rules, W3E.q65.fillna(W3[QB]), C.nn, PC[QB], PD[QB]], axis=1).clip(lower=0).mean(1); print("buying column: u65ec at", QB)
C["member_spread"] = C[MEM].clip(lower=1).max(1) / C[MEM].clip(lower=1).min(1)
C["P10"] = C.forecast * PC.q10 / PC.q50.clip(lower=1); C["P90"] = C.forecast * PC.q90 / PC.q50.clip(lower=1)   # D266: TabPFN conformal relative range on the final forecast (backtest coverage 80.6%)
L = lambda a, b: np.log((C[a] + 1) / (C[b] + 1))
C["d_season"] = L("season", "R3")
for m in MEM: C[f"d_{m}"] = L(m, "season")
C["d_stack"] = L("avg5", "season"); C["d_corr"] = L("forecast", "avg5"); C["d_total_vs_R3"] = L("forecast", "R3"); C["d_buying"] = L("buying", "forecast")
prof = np.exp((np.log(RU.div(RU.sum(1), axis=0).clip(lower=1e-4)) + np.log(NNm.div(NNm.sum(1), axis=0).clip(lower=1e-4))) / 2)
prof = prof.div(prof.sum(1), axis=0).fillna(0.25)
for mth in ["jul", "aug", "sep", "oct"]: C[mth] = (C.forecast * prof[mth]).round().astype(int)
# D034 override: S16079_DARK-GRAY is out of stock since April (no model member forecasts it; the rule agent says 0). The brief says
# to assume every item is in stock, so it keeps the D034 in-stock forecast (Jan-Mar 2026 rate x the mature-top month multipliers).
OOS = pd.read_csv("data/cand_rwxs_final_forecast_q3.csv", index_col=0); OVR = []
for s_ in [x for x in TG if C.loc[x, ["qnx", "pf2c", "pf2d32"]].isna().all() and C.loc[x, "rules"] == 0 and x in OOS.index]:
    for mth in ["jul", "aug", "sep", "oct"]: C.loc[s_, mth] = int(OOS.loc[s_, "f_" + mth])
    C.loc[s_, ["avg5", "forecast", "buying"]] = OOS.loc[s_, ["f_jul", "f_aug", "f_sep", "f_oct"]].sum(); C.loc[s_, "explanation"] = ("Out of stock since April; "
        "forecast as if back in stock (as the brief asks): its January-March rate x the seasonal pattern of tops at least a year old. No model member covers it.")
    OVR.append(s_); print("in-stock override for", s_)
# D291: the D033 World Cup rule, values from the first submission (12 Nov) rows
WC = {"S24840_LIGHT-GRAY-MELANGE": ([8554, 6416, 4277, 3849], "World Cup and 4th of July tee launched 25 May; 6,685 units in June, about 306 a day in late June"),
      "S25039_YELLOW": ([3366, 2524, 1683, 1515], "World Cup tee launched 25 May; 2,584 units in June, about 120 a day at month end"),
      "S24847_NAVY": ([844, 633, 422, 380], "World Cup shorts launched 25 May, flat at about 30 a day in June (928 units)")}
WCX = {}
for s_, (vals, fact) in WC.items():
    if s_ not in C.index: continue
    model_total = float(C.loc[s_, "forecast"]); ratio = float(C.loc[s_, "buying"]) / max(model_total, 1)
    for mth, v in zip(["jul", "aug", "sep", "oct"], vals): C.loc[s_, mth] = int(v)
    C.loc[s_, "buying"] = sum(vals) * ratio
    WCX[s_] = (f"{fact}. The models answer {model_total:,.0f} (P10-P90 {C.loc[s_, 'P10']:,.0f}-{C.loc[s_, 'P90']:,.0f}), not knowing the tournament ends on 19 July. "
               f"I set July to the late-June pace until the final and capped Aug-Oct at 75/50/45% of July, the post-event decay of past holiday items. Final {sum(vals):,}. "
               "Confidence 3: one season, event-driven demand.")
    print("World Cup rule:", s_, "model", round(model_total), "-> rule", sum(vals), "| buying", round(C.loc[s_, "buying"]))
# D296 (error-study proposal #3): stock cap, a no-op until Edikted sends stock. If data/stock_2026.csv exists (style_color, on_hand at 30 Jun
# 2026, inbound_jul, inbound_aug, inbound_sep, inbound_oct), each item's cumulative monthly forecast is capped at on-hand + cumulative inbound,
# the cut is recorded per item (stock_cut) and in the explanation; the uncapped number stays in "Model raw total". Without the file nothing changes.
C["stock_cut"] = 0.0
if os.path.exists("data/stock_2026.csv"):
    ST = pd.read_csv("data/stock_2026.csv").set_index("style_color").reindex(C.index)
    for s_ in C.index[ST.on_hand.notna().values]:
        avail = float(ST.loc[s_, "on_hand"]) + np.cumsum([float(ST.loc[s_, f"inbound_{m_}"]) if pd.notna(ST.loc[s_, f"inbound_{m_}"]) else 0.0 for m_ in ["jul", "aug", "sep", "oct"]])
        f_ = C.loc[s_, ["jul", "aug", "sep", "oct"]].astype(float).values; cum = np.minimum(np.cumsum(f_), avail); capped = np.diff(np.r_[0, cum]).round()
        C.loc[s_, "stock_cut"] = f_.sum() - capped.sum()
        for m_, v_ in zip(["jul", "aug", "sep", "oct"], capped): C.loc[s_, m_] = int(max(v_, 0))
    print("stock cap applied:", int((C.stock_cut > 0).sum()), "items cut,", int(C.stock_cut.sum()), "units")
else: print("stock cap: no data/stock_2026.csv, no change")
C["total"] = C[["jul", "aug", "sep", "oct"]].sum(1)
w = (PC.q75 - PC.q25) / (2 * PC.q50.clip(lower=1)); conf = np.where(w <= 0.2, 1, np.where(w <= 0.5, 2, 3))
C["confidence"] = np.minimum(3, conf + (C.member_spread > 2).astype(int))
for s_ in WCX: C.loc[s_, "confidence"] = 3
CF = pd.read_csv("data/fk_confidence_2026.csv").set_index("style_color").level.reindex(C.index)   # D299: the fork's D318 rule
C["confidence_old"] = C.confidence; C["confidence"] = CF.fillna(C.confidence).astype(int)
for s_ in list(WCX) + OVR: C.loc[s_, "confidence"] = 3
print("confidence (D318 rule):", C.confidence.value_counts().sort_index().to_dict())
C["product_type"] = it.product_type.reindex(TG).values; C["style"] = pd.Index(TG).str.split("_").str[0]
EX = pd.read_parquet("data/fk_2026_explain.parquet").set_index("style_color").reindex(TG) if os.path.exists("data/fk_2026_explain.parquet") else None
_AGE = pd.read_parquet("data/dec_base_nx5r26.parquet").query("origin == '2026-06'").set_index("style_color").age_m.reindex(TG)
_PL = [("life cycle", "its longer-run sales trajectory"), ("Portfolio baseline", "the usual drift of items below their recent rate"), ("Upcoming events", "the autumn events (back-to-school, Halloween, the Black Friday run-up)"),
       ("momentum", "its recent weekly momentum"), ("hangover", "a fade after a recent event"), ("weather", "the weather season where it sells"), ("climate", "its regional climate mix"),
       ("Last-year", "how it sold last summer and autumn"), ("last year", "how it sold last summer and autumn"), ("cluster", "the seasonal pattern of items bought with it"), ("style", "its style attributes"), ("maturity", "its size and maturity")]
def _plain(cat):
    for k, v in _PL:
        if isinstance(cat, str) and k.lower() in cat.lower(): return v
    return str(cat).split(" (")[0].lower()
def explain(s):
    """two plain sentences (Fork 2, new nx5: the brief asks for 'one or two sentences per item: what drove your number, what the model said, whether you adjusted it')"""
    r = C.loc[s]; pc = lambda v: f"{np.exp(v) - 1:+.0%}"; a = _AGE.get(s, np.nan)
    agep = (f"launched {a:.0f} month{'s' if round(a) != 1 else ''} ago" if a < 12 else f"on sale for {a / 12:.0f} year{'s' if round(a / 12) > 1 else ''}") if np.isfinite(a) else "with a short history"
    s1 = (f"{r.product_type}, {agep}, selling about {r.R3 / 4:,.0f} a month in April-June; its type usually moves {pc(r.d_season)} from there into July-October"
          if abs(r.d_season) >= 0.03 else f"{r.product_type}, {agep}, selling about {r.R3 / 4:,.0f} a month in April-June; its type is about flat into July-October")
    drv = ""
    if EX is not None and isinstance(EX.loc[s].get("cat1", None), str):
        d1, p1, d2, p2 = EX.loc[s, "cat1"], float(EX.loc[s, "cat1_pct"]), EX.loc[s, "cat2"], float(EX.loc[s, "cat2_pct"])
        drv = f", mostly on {_plain(d1)} ({'up' if p1 > 0 else 'down'}) and {_plain(d2)} ({'up' if p2 > 0 else 'down'})"
    rt = " (for a launch, the tree model trained to weigh launches more)" if np.isfinite(a) and a < 6 else ""
    dis = (f"; the models disagree ({r[MEM].min():,.0f} to {r[MEM].max():,.0f})" + (", so confidence is 3" if int(r.confidence) == 3 else "")) if r.member_spread > 2.5 else ""
    s2 = f"The models{rt} give {r.forecast:,.0f}, {pc(r.d_total_vs_R3)} on the recent rate{drv}{dis}. Not adjusted."
    return s1 + ". " + s2
pick = list(C.assign(a=C.d_total_vs_R3.abs()).sort_values("a", ascending=False).index[:10]) + list(C.sort_values("member_spread", ascending=False).index[:8])
pick = list(dict.fromkeys(pick))[:18]; C["explanation"] = ""
for s in pick: C.loc[s, "explanation"] = explain(s)
for s in OVR: C.loc[s, "explanation"] = ("Out of stock since April; forecast as if back in stock (as the brief asks): its January-March rate x the seasonal "
    "pattern of tops at least a year old (D034). No model member covers it.")
for s_, t_ in WCX.items(): C.loc[s_, "explanation"] = t_   # D291
C["final"] = C[["jul", "aug", "sep", "oct"]].sum(1)
_RG = pd.read_csv("data/ranges_nx5r_2026.csv", comment="#").drop_duplicates("style_color").set_index("style_color").reindex(C.index)   # D368 / D366: the fork's D329 per-segment (Mondrian) ratios
for _q in ("10", "25", "75", "90"): C[f"P{_q}"] = ((C.final + 1) * _RG[f"r{_q}"] - 1).clip(lower=0).round()   # P = (P50 + 1) x r - 1 on the workbook total
_VQ = pd.DataFrame({"l": ((C.final + 1) * _RG.r25 - 1).clip(lower=0), "h": ((C.final + 1) * _RG.r75 - 1).clip(lower=0)})   # D371 spreads use the adopted D329 halves (as calibrated)
_WM = pd.read_csv("data/fk_ranges_2026_wm.csv", comment="#").drop_duplicates("style_color").set_index("style_color").reindex(C.index)   # D369 / D370b width model
C["P25"] = np.maximum(C.P25, C.P10); C["P75"] = np.minimum(C.P75, C.P90)   # new nx5: all four from the focus-weighted per-segment refit (data/ranges_nx5r_2026.csv)
_ok = C.P10.notna(); assert ((C.P10 <= C.P25) & (C.P25 <= C.final) & (C.final <= C.P75) & (C.P75 <= C.P90))[_ok].all()
# D389k / D389n: hybrid confidence (D318 level 1; level 3 where the calibrated ranges give p(within +/-50%) < 0.5; hand-rule rows 3)
_L = np.log1p(np.column_stack([C.P10, C.P25, C.final, C.P75, C.P90]).astype(float).clip(min=0)); _L = np.maximum.accumulate(_L, axis=1)
def _cdf(x):
    o = []
    for i_, v_ in enumerate(np.log1p(np.maximum(np.asarray(x, float), 0))):
        k_ = _L[i_]
        if not np.isfinite(k_).all(): o.append(np.nan); continue
        o.append(np.interp(v_, np.r_[k_[0] - 2 * max(k_[1] - k_[0], .05), k_, k_[4] + 2 * max(k_[4] - k_[3], .05)], [0, .1, .25, .5, .75, .9, 1]))
    return np.array(o)
C["p_within20"] = (_cdf(1.2 * C.final) - _cdf(0.8 * C.final)).round(3); C["p_within50"] = (_cdf(1.5 * C.final) - _cdf(0.5 * C.final)).round(3)
C["confidence_d318"] = C.confidence
C["confidence"] = np.where(CF.reindex(C.index).values == 1, 1, np.where(C.p_within50.fillna(1).values < 0.5, 3, 2)).astype(int)
for s_ in list(WCX) + OVR: C.loc[s_, "confidence"] = 3
print("confidence (hybrid, D389n):", C.confidence.value_counts().sort_index().to_dict())
for s in pick: C.loc[s, "explanation"] = explain(s)   # rewritten so the text matches the new levels
for s in OVR: C.loc[s, "explanation"] = ("Out of stock since April; forecast as if back in stock (as the brief asks): its January-March rate x the seasonal "
    "pattern of tops at least a year old (D034). No model member covers it.")
for s_, t_ in WCX.items(): C.loc[s_, "explanation"] = t_
# D369: totals ranges by the fork's D371 rule, never the sum of item ranges
_c, _qs = 0.1278, {"P10": -1.801, "P25": -0.774, "P75": 0.921, "P90": 1.525}
_A = pd.DataFrame({"F": C.final, "v50l": ((C.final - _VQ.l) / 0.6745) ** 2, "v50h": ((_VQ.h - C.final) / 0.6745) ** 2, "product_type": C.product_type, "style": C["style"], "all": "all 100 targets"})
_TR = []
for _lab, _key in [("total", "all"), ("product type", "product_type"), ("style", "style")]:
    _g = _A.groupby(_key).agg(P50=("F", "sum"), items=("F", "size"), v50l=("v50l", "sum"), v50h=("v50h", "sum")).reset_index().rename(columns={_key: "group"}); _g = _g[_g["items"] >= 3]
    _s = np.sqrt(((np.sqrt(_g.v50l) + np.sqrt(_g.v50h)) / 2 / (_g.P50 + 1)) ** 2 + _c ** 2)
    for _nm, _q in _qs.items(): _g[_nm] = (np.exp(np.log(_g.P50 + 1) + _q * _s) - 1).round()
    _g.insert(0, "level", _lab); _TR.append(_g[["level", "group", "items", "P10", "P25", "P50", "P75", "P90"]])
TRG = pd.concat(_TR, ignore_index=True); TRG.to_csv("data/final_2026_nx6_total_ranges" + ("_TEST" if TEST else "") + ".csv", index=False)

C.to_csv("data/final_2026_nx6_cascade" + ("_TEST" if TEST else "") + ".csv")
LAY = ["R3", "season"] + MEM + ["avg5", "forecast", "final", "p2c32a_prev", "buying"]   # final = the workbook total after the D034 / D033 rules
H = pd.concat([C[LAY].sum().to_frame("total").T.assign(level="total", group="all 100 targets"),
               C.groupby("product_type")[LAY].sum().assign(level="product type").reset_index().rename(columns={"product_type": "group"}),
               C.groupby("style")[LAY].sum().assign(level="style").reset_index().rename(columns={"style": "group"})], ignore_index=True)
H.to_csv("data/final_2026_nx6_cascade_hierarchy" + ("_TEST" if TEST else "") + ".csv", index=False)
shutil.copyfile(WB_IN, WB_OUT); wb = openpyxl.load_workbook(WB_OUT); ws = wb["Forecast"]
hdr = {c.value: c.column for c in ws[1]}
for row in ws.iter_rows(min_row=2):
    s = row[1].value
    if s not in C.index: continue
    r = C.loc[s]
    for col, v in [("Jul-2026 units", r.jul), ("Aug-2026 units", r.aug), ("Sep-2026 units", r.sep), ("Oct-2026 units", r.oct), ("Total Jul-Oct", r.total),
                   ("Confidence (1-3)", int(r.confidence)), ("Explanation (at least 15 items of your choice)", r.explanation or None),
                   ("Model raw total Jul-Oct (optional, before your adjustments)", round(float(r.forecast)))]:
        ws.cell(row=row[0].row, column=hdr[col], value=int(v) if isinstance(v, (np.integer, int, float)) and col != "Explanation (at least 15 items of your choice)" else v)
for name, D in [("Cascade (items)", C.reset_index().rename(columns={"index": "style_color"})[["style_color", "product_type"] + LAY + ["d_season"] + [f"d_{m}" for m in MEM] + ["d_stack", "d_corr", "d_total_vs_R3", "d_buying", "member_spread", "confidence", "P10", "P25", "P75", "P90"]]),
                ("Cascade (hierarchy)", H[["level", "group"] + LAY]), ("Ranges (totals)", TRG)]:
    w2 = wb.create_sheet(name); w2.append(list(D.columns))
    for rr in D.itertuples(index=False): w2.append([float(x) if isinstance(x, (np.floating,)) else x for x in rr])
ME = {
 "Your approach": ("Data first. The history has artefacts that would mislead a model read naively, and each was handled: "
   "the 1,500 items are a survivor panel that enters over time (190 published before 2024, 324 in 2026), so total units grow about sixfold mostly from more items, and trend and seasonality are read only from like-for-like cohorts, never from raw totals; "
   "a mass cancellation on 14-15 October 2025 (145k units, 74% of the day) is excluded with all cancelled lines, as the target says, and those days are not used to learn October; "
   "own stores rolled out from 1 to 16, so store openings are treated as distribution steps (stores per item are an input), not as demand growth; "
   "the launch month is partial (0.32x of the next months), so it is scaled by days live; 85 items sold before their published date, and published_date is stored at midnight Israel time, so an item starts at the earlier of its first sale and its NY-time publish date; "
   "discounts outside 0-1 are clipped; zero runs right after steady selling (possible stock-outs) are treated as missing, not as zero demand, as the brief asks; "
   "and small things (two near-empty days in 2024, mixed state codes, 0.03% possible duplicate lines, items with several names or types) were checked and left or normalised. "
   "Each item's July-October total starts as the weighted average of six forecasts that look at the problem differently: "
   "(1) explicit planning rules (recent daily rate x like-for-like seasonal and event indices by product type x launch ramp and fade), written and checked month by month; "
   "(2) a LightGBM model of the 4-month total relative to the item's recent rate (for items under six months old, a version trained with new items counted five times, which reads launches better), using its sales history, where it sits on the typical launch curve for its age and type, "
   "its product type's seasonality, the normal temperature where it sells, a calendar of about 30 US events and sales periods (back-to-school, Halloween, Prime days, "
   "Black Friday, cyber week ...) and the seasonal pattern of the items it is bought with (shopping-cart neighbours); "
   "(3) a neural network whose item embedding is built hierarchically (product type, then style, then colour) with a learned encoder of the weekly sales curve, given the same events and launch-curve inputs; "
   "(4, 5) TabPFN v2, a tabular foundation model, on similar features, one version with calibrated ranges; (6) a recommender-style model that forecasts each item at each place it sells (four web regions, TikTok, each own store, each wholesale partner) and adds them up; it gets a quarter of the weight of the others, because it leans more on last year's pattern. "
   "The average is then corrected by two small models learned only from earlier forecast errors: systematic biases by volume, age and discount group, "
   "and signals of breakouts and of fading items (weekly momentum, growth of look-alike items, promotion sensitivity, a recent spike); the overall total is kept. "
   "Two hand rules override the models where they cannot know the cause: the three World Cup items follow the late-June pace to the 19 July final and then "
   "the measured post-event decay of past holiday items, and the out-of-stock item is forecast as if back in stock, as the brief asks. "
   "The monthly split follows the rules' and the network's monthly profiles. "
   "Every number is shown as a cascade in the 'Cascade' sheets: recent rate x4 (R3), then seasonality, then each model, their average, the correction, "
   "rolled up by style, type and total."),
 "How you tested it": ("A rolling backtest at 13 month-ends (Feb 2025 - Feb 2026): every model and correction is trained only on data before the month-end and scored "
   "on the next four months, always against a simple recency forecast (R3 = last 3 months x 4). On the nine most recent origins the final forecast has 0.381 x R3's error, and 0.293 on the five latest origins, which were never used to choose anything "
   "(WAPE), better than the rules, each model alone and their plain average (0.39). Confidence was set by a rule tested the same way: 1 only for items with the narrowest forecast range that are also rising and sold in the same months last year "
   "(about 60% of them landed within +/-20%); 3 where the item's calibrated range gives less than even odds of landing within +/-50% (61% of those missed by more than 50%) and for the four hand-ruled items; 2 for the rest (83% landed within +/-50%). Judged by the scale's own definitions, 81% of the stated levels held on the five latest origins, against 75% for a rule that put the widest fifth of ranges at 3. About four in ten item totals land within +/-20% overall, and a 1 means "
   "better than even odds, not certainty. The 'Cascade' sheet gives each item a P10-P90 and a P25-P75 range: the item's own uncertainty (from the calibrated TabPFN model) scaled per age and volume group so that 10% and 25% of past actuals fall outside each side, with last July-October and the latest March-June counting four times as much; left one origin out, about 80% and 50% of actual totals fell inside them (on last July-October only 70%: that season ran below the forecast, while March-June 2026 ran above it). Ranges for totals are in the 'Ranges (totals)' sheet: they are not the sum of item ranges, because item misses partly move together; they allow for that shared part. Leakage checks: features rebuilt from data cut at the month-end are identical, "
   "a model trained on shuffled outcomes collapses, and a model that never saw an item's style is only 1% worse, so it generalises rather than memorising items. "
   "A buying simulation (orders placed from the forecast, profit compared with perfect foresight) showed that under-forecasting costs more than over-forecasting, "
   "so the 'Cascade' sheet also gives a buying quantity (an upper forecast, about the 65th percentile), which loses the least profit. "
   "Around ninety other ideas were tested the same way and dropped when they did not help (campaign calendar, Google Trends, hierarchical reconciliation, "
   "neural sequence models, separate Black Friday and low-volume models ...); the decision log records each one."),
 "Where it is weak": ("The largest misses are young launches that break out (sales growing 3-10x within the window), which every model misses together; "
   "about half of them show within the first month, so a monthly re-forecast of young items catches them. Items bought together or looking alike also rise and fall "
   "together within the window, and nothing in past sales predicts that shared shock; the launch and markdown plan, early sell-through and the stock plan would help most. "
   "Least reliable: young launches, items with the widest forecast ranges, often where the six models disagree (confidence 3), and the out-of-stock item, forecast as if back in stock as the brief asks. "
   "The backtest only contains items still selling in 2025-26, so it never saw an item being discontinued. "
   "The forecast ran about 5% below actuals on the recent backtest origins (about level over all 13); the buying quantity corrects for that. "
   "For the 24 items under six months old, the launch-weighted tree lifts the forecast by about 11% in total: the backtest supports the direction (new items ran 7-12% below the forecast) but never saw a lift this large, so these are the first items to re-forecast once July is known.")}
wm = wb["Model explanation"]
for row in wm.iter_rows(min_row=2):
    if row[0].value in ME: wm.cell(row=row[0].row, column=2, value=ME[row[0].value])
wb.save(WB_OUT)
tot = C[["jul", "aug", "sep", "oct", "total"]].sum(); print("forecast total", int(tot.total), dict(tot.astype(int)), "| buying total", int(C.buying.sum()), "| R3 total", int(C.R3.sum()))
print("confidence", C.confidence.value_counts().sort_index().to_dict(), "| explanations", int((C.explanation != "").sum())); print(H[H.level != "style"].round(0).to_string(index=False))
