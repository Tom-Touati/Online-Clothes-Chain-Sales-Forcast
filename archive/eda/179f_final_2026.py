"""D235: the final Jul-Oct 2026 forecast as a cascade of stacked models (user: "for explainability sake, keep the cascade of
hierarchial models that stack up"), and a NEW workbook. The submitted Edikted_Forecasting_Exercise_filled.xlsx is never touched.
Members at the June 2026 origin (4-month totals): rules (June 2026 rule agent, agent_eval/2026-06), NN (finalS + finalS_s10, monthly
P50 averaged), and the fork's members data/fk_2026_{qdyn,pf2c,pf2d32,w03}.parquet (q10..q90).
  forecast  = arithmetic mean of rules, D198 quantile (qdyn), NN, conformal TabPFN (pf2c), TabPFN dyn (pf2d32) at P50    (p2c32a, D231)
  buying    = arithmetic mean of rules, volume-weighted LightGBM (w03), NN, pf2c, pf2d32 with w03 / pf2c / pf2d32 at P65
              (the newsvendor quantity: the D236 ladder bottoms at q65-q70, 10.38% loss; q65 used, q60 if absent)
Monthly split: the geometric mean of the rule agent's and the NN's monthly profiles (D121), normalised.
Confidence (1-3): the conformal TabPFN member's P25-P75 width around its median (inside +/-20% -> 1, inside +/-50% -> 2, else 3),
raised by one level where the members disagree by more than 2x.
CASCADE per target, each layer stacking on the previous (log contributions shown): R3 (last 3 months x 4) -> x the product type's
like-for-like seasonality (D159's seasonal layer) -> each member -> the stack -> the buying quantity. Rolled up by style, product type
and total. Explanations (>= 15 targets, the largest movers and the disagreement cases) are generated from the cascade and, if present,
the fork's data/fk_2026_explain.parquet categories.
Writes data/final_2026_cascade.csv, data/final_2026_cascade_hierarchy.csv and Edikted_Forecasting_Exercise_candidate_final.xlsx."""
import sys, os, shutil, numpy as np, pandas as pd, openpyxl
sys.stdout.reconfigure(encoding="utf-8")
TEST = os.environ.get("FINAL_TEST") == "1"   # dry run: missing fork members replaced by qdyn, outputs to *_TEST files
WB_IN, WB_OUT = "Edikted_Forecasting_Exercise.xlsx", ("Edikted_Forecasting_Exercise_candidate_final_TEST.xlsx" if TEST else "Edikted_Forecasting_Exercise_candidate_final.xlsx")
assert WB_OUT != "Edikted_Forecasting_Exercise_filled.xlsx"
wb0 = openpyxl.load_workbook(WB_IN); ws0 = wb0["Forecast"]
TG = [r[1].value for r in ws0.iter_rows(min_row=2) if r[1].value]; print("targets", len(TG))
it = pd.read_parquet("data/items.parquet").set_index("style_color")
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
R3 = 4 * M.reindex(TG)[["2026-04", "2026-05", "2026-06"]].mean(1)
F = pd.read_parquet("data/features.parquet"); F = F[F.origin == "2026-06"]
SEA = F.assign(st=F.m3 * np.exp(F.lfl_type.fillna(0))).groupby("style_color").st.sum().reindex(TG).fillna(R3)
RU = pd.read_csv("agent_eval/2026-06/predictions_verified.csv", dtype={"month": str}).pivot(index="style_color", columns="month", values="pred").reindex(TG)
RU.columns = ["jul", "aug", "sep", "oct"]
NNm = sum(pd.read_parquet(f"data/{f}.parquet").pivot(index="style_color", columns="h", values="q50") for f in ["nn_backtest_preds_finalS", "nn_backtest_preds_finalS_s10"]) / 2
NNm = NNm.reindex(TG); NNm.columns = ["jul", "aug", "sep", "oct"]
fk = lambda m: pd.read_parquet(f"data/fk_2026_{m}.parquet" if (os.path.exists(f"data/fk_2026_{m}.parquet") or not TEST) else "data/fk_2026_qdyn.parquet").drop_duplicates("style_color").set_index("style_color").reindex(TG)
QD, PC, PD, W3 = fk("qdyn"), fk("pf2c"), fk("pf2d32"), fk("w03")
C = pd.DataFrame(index=TG); C["R3"] = R3; C["season"] = SEA
C["rules"] = RU.sum(1); C["qdyn"] = QD.q50; C["nn"] = NNm.sum(1); C["pf2c"] = PC.q50; C["pf2d32"] = PD.q50
for c in ["rules", "nn"]: C[c] = C[c].fillna(C[["qdyn", "pf2c", "pf2d32"]].mean(1))
MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]
C["forecast"] = C[MEM].clip(lower=0).mean(1)
QB = "q65" if all("q65" in X.columns for X in (W3, PC, PD)) else "q60"   # D236: the ladder bottoms at q65-q70 (10.38%)
C["buying"] = pd.concat([C.rules, W3[QB], C.nn, PC[QB], PD[QB]], axis=1).clip(lower=0).mean(1); print("buying column at", QB)
SL = fk("sel"); C["qsc32a_alt"] = pd.concat([C.rules, C.qdyn, SL.q50, C.nn, C.pf2c, C.pf2d32], axis=1).clip(lower=0).mean(1)   # accuracy alternative (D231)
C["member_spread"] = C[MEM].clip(lower=1).max(1) / C[MEM].clip(lower=1).min(1)
L = lambda a, b: np.log((C[a] + 1) / (C[b] + 1))
C["d_season"] = L("season", "R3")
for m in MEM: C[f"d_{m}"] = L(m, "season")
C["d_stack"] = L("forecast", "season"); C["d_total_vs_R3"] = L("forecast", "R3"); C["d_buying"] = L("buying", "forecast")
prof = np.exp((np.log(RU.div(RU.sum(1), axis=0).clip(lower=1e-4)) + np.log(NNm.div(NNm.sum(1), axis=0).clip(lower=1e-4))) / 2)
prof = prof.div(prof.sum(1), axis=0).fillna(0.25)
for mth in ["jul", "aug", "sep", "oct"]: C[mth] = (C.forecast * prof[mth]).round().astype(int)
# D034 override: S16079_DARK-GRAY is out of stock since April (no model member forecasts it; the rule agent says 0). The brief says
# to assume every item is in stock, so it keeps the D034 in-stock forecast (Jan-Mar 2026 rate x the mature-top month multipliers).
OOS = pd.read_csv("data/cand_rwxs_final_forecast_q3.csv", index_col=0); OVR = []
for s_ in [x for x in TG if C.loc[x, ["qdyn", "pf2c", "pf2d32"]].isna().all() and C.loc[x, "rules"] == 0 and x in OOS.index]:
    for mth in ["jul", "aug", "sep", "oct"]: C.loc[s_, mth] = int(OOS.loc[s_, "f_" + mth])
    C.loc[s_, ["forecast", "buying"]] = OOS.loc[s_, ["f_jul", "f_aug", "f_sep", "f_oct"]].sum(); C.loc[s_, "explanation"] = ("Out of stock since April; "
        "forecast as if back in stock (as the brief asks): its January-March rate x the seasonal pattern of tops at least a year old. No model member covers it.")
    OVR.append(s_); print("in-stock override for", s_)
C["total"] = C[["jul", "aug", "sep", "oct"]].sum(1)
w = (PC.q75 - PC.q25) / (2 * PC.q50.clip(lower=1)); conf = np.where(w <= 0.2, 1, np.where(w <= 0.5, 2, 3))
C["confidence"] = np.minimum(3, conf + (C.member_spread > 2).astype(int))
C["product_type"] = it.product_type.reindex(TG).values; C["style"] = pd.Index(TG).str.split("_").str[0]
EX = pd.read_parquet("data/fk_2026_explain.parquet").set_index("style_color").reindex(TG) if os.path.exists("data/fk_2026_explain.parquet") else None
def explain(s):
    r = C.loc[s]; top = max(MEM, key=lambda m: abs(r[f"d_{m}"])); pct = lambda v: f"{np.exp(v) - 1:+.0%}"
    t = (f"Recent rate x4 (R3) {r.R3:,.0f}; its product type's seasonal pattern moves that {pct(r.d_season)} to {r.season:,.0f}. "
         f"The five models range {r[MEM].min():,.0f}-{r[MEM].max():,.0f} (furthest from the seasonal baseline: {top}, {pct(r[f'd_{top}'])}). "
         f"Their average, {r.forecast:,.0f}, is {pct(r.d_total_vs_R3)} against R3. For buying, the newsvendor quantity is {r.buying:,.0f}.")
    if EX is not None and isinstance(EX.loc[s].get("why_text", None), str): t += " Drivers (SHAP of the tree member): " + EX.loc[s, "why_text"].replace("Tree member", "the tree model")
    return t
pick = list(C.assign(a=C.d_total_vs_R3.abs()).sort_values("a", ascending=False).index[:10]) + list(C.sort_values("member_spread", ascending=False).index[:8])
pick = list(dict.fromkeys(pick))[:18]; C["explanation"] = ""
for s in pick: C.loc[s, "explanation"] = explain(s)
for s in OVR: C.loc[s, "explanation"] = ("Out of stock since April; forecast as if back in stock (as the brief asks): its January-March rate x the seasonal "
    "pattern of tops at least a year old (D034). No model member covers it.")
C.to_csv("data/final_2026_cascade" + ("_TEST" if TEST else "") + ".csv")
LAY = ["R3", "season"] + MEM + ["forecast", "qsc32a_alt", "buying"]
H = pd.concat([C[LAY].sum().to_frame("total").T.assign(level="total", group="all 100 targets"),
               C.groupby("product_type")[LAY].sum().assign(level="product type").reset_index().rename(columns={"product_type": "group"}),
               C.groupby("style")[LAY].sum().assign(level="style").reset_index().rename(columns={"style": "group"})], ignore_index=True)
H.to_csv("data/final_2026_cascade_hierarchy" + ("_TEST" if TEST else "") + ".csv", index=False)
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
for name, D in [("Cascade (items)", C.reset_index().rename(columns={"index": "style_color"})[["style_color", "product_type"] + LAY + ["d_season"] + [f"d_{m}" for m in MEM] + ["d_stack", "d_total_vs_R3", "d_buying", "member_spread", "confidence"]]),
                ("Cascade (hierarchy)", H[["level", "group"] + LAY])]:
    w2 = wb.create_sheet(name); w2.append(list(D.columns))
    for rr in D.itertuples(index=False): w2.append([float(x) if isinstance(x, (np.floating,)) else x for x in rr])
ME = {
 "Your approach": ("Each item's July-October total is the plain average of five forecasts that look at the problem differently: "
   "(1) explicit planning rules (recent daily rate x like-for-like seasonal and event indices by product type x launch ramp and fade), written and checked month by month; "
   "(2) a LightGBM quantile model of the 4-month total relative to the item's recent rate, using its sales history, its product type's seasonality, "
   "the normal temperature where the item sells, a US events calendar (back-to-school, Halloween, Black Friday ...), the seasonal pattern of the items it is bought with "
   "(shopping-cart neighbours) and item attributes scored by a language model (warmth, party, festival ...); "
   "(3) a neural network whose item embedding is built hierarchically (product type, then style, then colour) with a learned encoder of the weekly sales curve; "
   "(4, 5) TabPFN v2, a tabular foundation model, on the same features, one version with calibrated ranges. "
   "Seasonality and events enter as things known in advance; new items borrow from their product type, from look-alike items (product photos) and from items bought with them. "
   "The monthly split follows the rules' and the network's monthly profiles. "
   "Every number is shown as a cascade in the 'Cascade' sheets: recent rate x4 (R3), then seasonality, then each model, then their average, rolled up by style, type and total."),
 "How you tested it": ("A rolling backtest at 13 month-ends (Feb 2025 - Feb 2026): every model is trained only on data before the month-end and scored on the next four months, "
   "always against a simple recency forecast (R3 = last 3 months x 4). On the nine most recent origins the final average has 0.40 x R3's error (WAPE), beats the rule set "
   "and each of the five models on its own, and its confidence ranges come from a model whose quantiles were recalibrated on earlier origins. "
   "A buying simulation (orders placed from the forecast, profit compared with perfect foresight) showed that under-forecasting costs more than over-forecasting, "
   "so the 'Cascade' sheet also gives a buying quantity (an upper forecast, about the 65th percentile), which loses the least profit. "
   "Around seventy other ideas were tested the same way and dropped when they did not help (campaign calendar, Google Trends, hierarchical reconciliation, "
   "neural sequence models, data augmentation ...); the decision log records each one."),
 "Where it is weak": ("Errors are shared by 'looks': items bought together or looking alike rise and fall together within the window (their errors correlate about 0.4), "
   "and nothing in the past sales predicts that shared shock. Launches in the same look explain part of it, so the launch and marketing plan for July-October would help most, "
   "followed by early-sales signals (first-week sell-through, add-to-carts) and the stock plan. "
   "Least reliable: young launches (only weeks of history), items whose five models disagree by more than 2x (marked confidence 3), and the out-of-stock item, "
   "which is forecast as if back in stock as the brief asks. The forecast ran about 5% below actuals on the recent backtest origins (about level over all 13); the buying quantity corrects for that.")}
wm = wb["Model explanation"]
for row in wm.iter_rows(min_row=2):
    if row[0].value in ME: wm.cell(row=row[0].row, column=2, value=ME[row[0].value])
wb.save(WB_OUT)
tot = C[["jul", "aug", "sep", "oct", "total"]].sum(); print("forecast total", int(tot.total), dict(tot.astype(int)), "| buying total", int(C.buying.sum()), "| R3 total", int(C.R3.sum()))
print("confidence", C.confidence.value_counts().sort_index().to_dict(), "| explanations", int((C.explanation != "").sum())); print(H[H.level != "style"].round(0).to_string(index=False))
