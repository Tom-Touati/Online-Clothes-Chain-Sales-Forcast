"""D291: as eda/181g (D273) plus the D033 World Cup rule restored on S24840_LIGHT-GRAY-MELANGE, S25039_YELLOW and S24847_NAVY (user, in the fork's
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
WB_IN, WB_OUT = "Edikted_Forecasting_Exercise.xlsx", ("candidates/Edikted_Forecasting_Exercise_candidate_nx3_TEST.xlsx" if TEST else "candidates/Edikted_Forecasting_Exercise_candidate_nx3.xlsx")
assert WB_OUT not in ("Edikted_Forecasting_Exercise_filled.xlsx", "candidates/Edikted_Forecasting_Exercise_candidate_final.xlsx", "candidates/Edikted_Forecasting_Exercise_candidate_nx.xlsx", "candidates/Edikted_Forecasting_Exercise_candidate_nx2.xlsx")
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
NX = pd.read_parquet("data/q_backtest_nx_ec_ns_2026.parquet").drop_duplicates("style_color").set_index("style_color").reindex(TG)
W3E = pd.read_parquet("data/q_backtest_nx_w03ec_2026.parquet").drop_duplicates("style_color").set_index("style_color").reindex(TG)
C["rules"] = RU.sum(1); C["qnx"] = NX.q50; C["nn"] = NNm.sum(1); C["pf2c"] = PC.q50; C["pf2d32"] = PD.q50; C["qdyn_prev"] = QD.q50
for c in ["rules", "nn"]: C[c] = C[c].fillna(C[["qnx", "pf2c", "pf2d32"]].mean(1))
MEM = ["rules", "qnx", "nn", "pf2c", "pf2d32"]
C["avg5"] = C[MEM].clip(lower=0).mean(1)
GEO = pd.read_parquet("data/geo_ecnsnn_x_2026.parquet").set_index("style_color").reindex(TG)
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
C["product_type"] = it.product_type.reindex(TG).values; C["style"] = pd.Index(TG).str.split("_").str[0]
EX = pd.read_parquet("data/fk_2026_explain.parquet").set_index("style_color").reindex(TG) if os.path.exists("data/fk_2026_explain.parquet") else None
def explain(s):
    r = C.loc[s]; top = max(MEM, key=lambda m: abs(r[f"d_{m}"])); pct = lambda v: f"{np.exp(v) - 1:+.0%}"
    t = (f"Recent rate x4 (R3) {r.R3:,.0f}; its product type's seasonal pattern moves that {pct(r.d_season)} to {r.season:,.0f}. "
         f"The five models range {r[MEM].min():,.0f}-{r[MEM].max():,.0f} (furthest from the seasonal baseline: {top}, {pct(r[f'd_{top}'])}). "
         f"Their average is {r.avg5:,.0f}; corrections learned from past forecast errors of similar items (volume, age, discount, weekly momentum, promotion sensitivity) "
         + (f"leave it at {r.forecast:,.0f}, " if abs(r.d_corr) < 0.005 else f"move it {pct(r.d_corr)} to {r.forecast:,.0f}, ") +
         f"{pct(r.d_total_vs_R3)} against R3. For buying, the newsvendor quantity is {r.buying:,.0f}.")
    if EX is not None and isinstance(EX.loc[s].get("why_text", None), str): t += " Drivers (SHAP of the previous tree model, which the new one extends with event and launch-curve features): " + EX.loc[s, "why_text"].replace("Tree member", "that model")
    return t
pick = list(C.assign(a=C.d_total_vs_R3.abs()).sort_values("a", ascending=False).index[:10]) + list(C.sort_values("member_spread", ascending=False).index[:8])
pick = list(dict.fromkeys(pick))[:18]; C["explanation"] = ""
for s in pick: C.loc[s, "explanation"] = explain(s)
for s in OVR: C.loc[s, "explanation"] = ("Out of stock since April; forecast as if back in stock (as the brief asks): its January-March rate x the seasonal "
    "pattern of tops at least a year old (D034). No model member covers it.")
for s_, t_ in WCX.items(): C.loc[s_, "explanation"] = t_   # D291
C["final"] = C[["jul", "aug", "sep", "oct"]].sum(1)
C.to_csv("data/final_2026_nx3_cascade" + ("_TEST" if TEST else "") + ".csv")
LAY = ["R3", "season"] + MEM + ["avg5", "forecast", "final", "p2c32a_prev", "buying"]   # final = the workbook total after the D034 / D033 rules
H = pd.concat([C[LAY].sum().to_frame("total").T.assign(level="total", group="all 100 targets"),
               C.groupby("product_type")[LAY].sum().assign(level="product type").reset_index().rename(columns={"product_type": "group"}),
               C.groupby("style")[LAY].sum().assign(level="style").reset_index().rename(columns={"style": "group"})], ignore_index=True)
H.to_csv("data/final_2026_nx3_cascade_hierarchy" + ("_TEST" if TEST else "") + ".csv", index=False)
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
for name, D in [("Cascade (items)", C.reset_index().rename(columns={"index": "style_color"})[["style_color", "product_type"] + LAY + ["d_season"] + [f"d_{m}" for m in MEM] + ["d_stack", "d_corr", "d_total_vs_R3", "d_buying", "member_spread", "confidence", "P10", "P90"]]),
                ("Cascade (hierarchy)", H[["level", "group"] + LAY])]:
    w2 = wb.create_sheet(name); w2.append(list(D.columns))
    for rr in D.itertuples(index=False): w2.append([float(x) if isinstance(x, (np.floating,)) else x for x in rr])
ME = {
 "Your approach": ("Each item's July-October total starts as the plain average of five forecasts that look at the problem differently: "
   "(1) explicit planning rules (recent daily rate x like-for-like seasonal and event indices by product type x launch ramp and fade), written and checked month by month; "
   "(2) a LightGBM model of the 4-month total relative to the item's recent rate, using its sales history, where it sits on the typical launch curve for its age and type, "
   "its product type's seasonality, the normal temperature where it sells, a calendar of about 30 US events and sales periods (back-to-school, Halloween, Prime days, "
   "Black Friday, cyber week ...) and the seasonal pattern of the items it is bought with (shopping-cart neighbours); "
   "(3) a neural network whose item embedding is built hierarchically (product type, then style, then colour) with a learned encoder of the weekly sales curve, given the same events and launch-curve inputs; "
   "(4, 5) TabPFN v2, a tabular foundation model, on similar features, one version with calibrated ranges. "
   "The average is then corrected by two small models learned only from earlier forecast errors: systematic biases by volume, age and discount group, "
   "and signals of breakouts and of fading items (weekly momentum, growth of look-alike items, promotion sensitivity, a recent spike); the overall total is kept. "
   "Two hand rules override the models where they cannot know the cause: the three World Cup items follow the late-June pace to the 19 July final and then "
   "the measured post-event decay of past holiday items, and the out-of-stock item is forecast as if back in stock, as the brief asks. "
   "The monthly split follows the rules' and the network's monthly profiles. "
   "Every number is shown as a cascade in the 'Cascade' sheets: recent rate x4 (R3), then seasonality, then each model, their average, the correction, "
   "rolled up by style, type and total."),
 "How you tested it": ("A rolling backtest at 13 month-ends (Feb 2025 - Feb 2026): every model and correction is trained only on data before the month-end and scored "
   "on the next four months, always against a simple recency forecast (R3 = last 3 months x 4). On the nine most recent origins the final forecast has 0.385 x R3's error "
   "(WAPE), better than the rules, each model alone and their plain average (0.40). The confidence levels were checked the same way: items the rule marks 1 landed within +/-20% 53% of the time (92% within +/-50%), level 2 44% (83%), "
   "level 3 33% (68%), so the levels rank risk correctly but a 1 is closer to an even chance than a promise; the P10-P90 range in the 'Cascade' sheet held 81% of actual totals (80% intended). Leakage checks: features rebuilt from data cut at the month-end are identical, "
   "a model trained on shuffled outcomes collapses, and a model that never saw an item's style is only 1% worse, so it generalises rather than memorising items. "
   "A buying simulation (orders placed from the forecast, profit compared with perfect foresight) showed that under-forecasting costs more than over-forecasting, "
   "so the 'Cascade' sheet also gives a buying quantity (an upper forecast, about the 65th percentile), which loses the least profit. "
   "Around ninety other ideas were tested the same way and dropped when they did not help (campaign calendar, Google Trends, hierarchical reconciliation, "
   "neural sequence models, separate Black Friday and low-volume models ...); the decision log records each one."),
 "Where it is weak": ("The largest misses are young launches that break out (sales growing 3-10x within the window), which every model misses together; "
   "about half of them show within the first month, so a monthly re-forecast of young items catches them. Items bought together or looking alike also rise and fall "
   "together within the window, and nothing in past sales predicts that shared shock; the launch and markdown plan, early sell-through and the stock plan would help most. "
   "Least reliable: young launches, items whose five models disagree by more than 2x (confidence 3), and the out-of-stock item, forecast as if back in stock as the brief asks. "
   "The backtest only contains items still selling in 2025-26, so it never saw an item being discontinued. "
   "The forecast ran about 5% below actuals on the recent backtest origins (about level over all 13); the buying quantity corrects for that.")}
wm = wb["Model explanation"]
for row in wm.iter_rows(min_row=2):
    if row[0].value in ME: wm.cell(row=row[0].row, column=2, value=ME[row[0].value])
wb.save(WB_OUT)
tot = C[["jul", "aug", "sep", "oct", "total"]].sum(); print("forecast total", int(tot.total), dict(tot.astype(int)), "| buying total", int(C.buying.sum()), "| R3 total", int(C.R3.sum()))
print("confidence", C.confidence.value_counts().sort_index().to_dict(), "| explanations", int((C.explanation != "").sum())); print(H[H.level != "style"].round(0).to_string(index=False))
