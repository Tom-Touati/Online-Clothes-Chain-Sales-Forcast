"""Score the partly recursive tree ensemble (D131) from data/rec_backtest_preds.parquet.
Monthly: item-month WAPE and shape error (true total given) of direct / recursive / partly against the stored LightGBM ensemble, NN, rules.
Totals (standard table, R3-relative with the R3 + rules row): each version's 4-month sum alone, as a fourth member of the three-way, and in
place of the quantile model; plus the rules + NN + partly monthly shape."""
import numpy as np, pandas as pd, sys, importlib.util
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
T = pd.read_parquet("data/fixes_items.parquet")
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
q = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))
LQ = np.log1p(q[["q10", "q25", "q50", "q75", "q90"]].values); D = {c: LQ[:, j] - LQ[:, 2] for j, c in enumerate(["q10", "q25", "q50", "q75", "q90"])}
Rc = pd.read_parquet("data/rec_backtest_preds.parquet")
b = pd.read_parquet("data/best_backtest_preds.parquet")[["origin", "style_color", "h", "pred"]].rename(columns={"pred": "lgb_stored"})
n = pd.read_parquet("data/nn_backtest_preds_S13x6.parquet")[["origin", "style_color", "h", "q50"]].rename(columns={"q50": "nn"})
X = Rc.merge(b, on=["origin", "style_color", "h"]).merge(n, on=["origin", "style_color", "h"]); RR = []
for o in X.origin.unique():
    r = pd.read_csv(f"agent_eval/{o}/predictions_verified.csv", dtype={"month": str}); r["h"] = [(pd.Period(m, "M") - pd.Period(o, "M")).n for m in r.month]; RR.append(r.assign(origin=o))
X = X.merge(pd.concat(RR)[["origin", "style_color", "h", "pred"]].rename(columns={"pred": "rules"}), on=["origin", "style_color", "h"])
for c in ["direct", "recursive", "partly", "lgb_stored", "nn", "rules"]: X[c] = X[c].clip(lower=0.01)
X["geo_r_nn"] = np.exp((np.log(X.rules) + np.log(X.nn)) / 2); X["geo_r_nn_partly"] = np.exp((np.log(X.rules) + np.log(X.nn) + np.log(X.partly)) / 3)
X["geo_r_nn_rec"] = np.exp((np.log(X.rules) + np.log(X.nn) + np.log(X.recursive)) / 3)
g = X.groupby(["origin", "style_color"]); C = ["lgb_stored", "direct", "recursive", "partly", "nn", "rules", "geo_r_nn", "geo_r_nn_partly", "geo_r_nn_rec"]
for c in C: X[c + "_s"] = X[c] / g[c].transform("sum") * g.y.transform("sum")
for nm, XR in [("recent origins", X[X.origin >= "2025-06"]), ("all 13 origins", X)]:
    w = lambda c: (XR[c] - XR.y).abs().sum() / XR.y.sum()
    print(f"== monthly, {nm}: item-month WAPE and shape error"); print(pd.DataFrame({"item-month WAPE": {c: w(c) for c in C}, "shape error": {c: w(c + "_s") for c in C}}).round(4).to_string())
XR = X[X.origin >= "2025-06"]
print("item-month WAPE by horizon (recent):"); print(pd.DataFrame({c: XR.groupby("h").apply(lambda x: (x[c] - x.y).abs().sum() / x.y.sum()) for c in ["direct", "recursive", "partly"]}).round(3).to_string())
tot = X.groupby(["origin", "style_color"])[["direct", "recursive", "partly"]].sum().reset_index().rename(columns={"direct": "dir_t", "recursive": "rec_t", "partly": "par_t"})
T = T.merge(tot, on=["origin", "style_color"], how="left")
lg = lambda c: np.log(T[c].clip(lower=0.5))
for c in ["dir_t", "rec_t", "par_t"]:
    T[f"four_{c}"] = np.exp((lg("rules") + lg("qmodel") + lg("nn") + lg(c)) / 4); T[f"r_{c}_nn"] = np.exp((lg("rules") + lg(c) + lg("nn")) / 3)
COLS = ["qmodel", "dir_t", "rec_t", "par_t", "r_q_nn", "four_par_t", "four_rec_t", "r_par_t_nn", "r_rec_t_nn"]
NM = {"qmodel": "quantile model", "dir_t": "LightGBM direct (4-month sum)", "rec_t": "LightGBM recursive", "par_t": "LightGBM partly recursive", "r_q_nn": "rules + quantile + NN",
      "four_par_t": "rules + quantile + NN + partly", "four_rec_t": "rules + quantile + NN + recursive", "r_par_t_nn": "rules + partly + NN", "r_rec_t_nn": "rules + recursive + NN"}
REC = [o for o in sorted(T.origin.unique()) if o >= "2025-06"]
for nm, Z in [("nine recent origins", T[T.origin.isin(REC)]), ("all 13 origins", T), ("recent, 100 targets", T[T.origin.isin(REC) & T.target])]:
    print(f"\n######## {nm}"); print(sl.table(Z, COLS, D, NM).round(4).to_string())
Z = T[T.origin.isin(REC)]; print("\nbootstrap WAPE(x) - WAPE(y), recent:")
for a, bb in [("rec_t", "dir_t"), ("par_t", "dir_t"), ("par_t", "qmodel"), ("four_par_t", "r_q_nn"), ("four_rec_t", "r_q_nn"), ("r_par_t_nn", "r_q_nn")]: print(f"  {a} - {bb}:", np.round(sl.boot(Z, a, bb), 4))
