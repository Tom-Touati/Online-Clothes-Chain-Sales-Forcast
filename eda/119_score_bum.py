"""Score the expected-value bottom-up models (D125b/D126): per location (4-month, Tweedie: data/bu_items_tw.parquet) and per location x month
(Tweedie: data/bum_items_tw.parquet), summed to items. Standard table on 4-month totals with the bottom-ups alone, with the quantile model,
inside the three-way, and as a fourth member; then monthly: item-month WAPE and shape error (true total given) of the fine bottom-up
against the members and the rules + NN shape."""
import numpy as np, pandas as pd, sys, importlib.util, os
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
T = pd.read_parquet("data/fixes_items.parquet")
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
q = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))
LQ = np.log1p(q[["q10", "q25", "q50", "q75", "q90"]].values); D = {c: LQ[:, j] - LQ[:, 2] for j, c in enumerate(["q10", "q25", "q50", "q75", "q90"])}
COLS, NM = ["qmodel", "r_q_nn"], {"qmodel": "quantile model (item level)", "r_q_nn": "rules + quantile + NN"}
lg = lambda c: np.log(T[c].clip(lower=0.5))
for f, c, nm in [("data/bu_items.parquet", "bu", "bottom-up per location (median)"), ("data/bu_items_tw.parquet", "butw", "bottom-up per location (expected value)"),
                 ("data/bum_totals_tw.parquet", "bumtw", "bottom-up per location x month (expected value)")]:
    if not os.path.exists(f): continue
    U = pd.read_parquet(f); col = "bu" if "bu" in U.columns else "bum"
    T = T.merge(U[["origin", "style_color", col]].rename(columns={col: c}), on=["origin", "style_color"], how="left"); T[c] = T[c].fillna(T.qmodel)
    T[c + "_q"] = np.exp((lg(c) + lg("qmodel")) / 2); T["r_" + c + "_nn"] = np.exp((lg("rules") + lg(c) + lg("nn")) / 3)
    T["four_" + c] = np.exp((lg("rules") + lg("qmodel") + lg("nn") + lg(c)) / 4)
    COLS += [c, c + "_q", "r_" + c + "_nn", "four_" + c]
    NM |= {c: nm, c + "_q": f"  {c} x quantile", "r_" + c + "_nn": f"  rules + {c} + NN", "four_" + c: f"  rules + quantile + NN + {c}"}
REC = [o for o in sorted(T.origin.unique()) if o >= "2025-06"]
for nm, Z in [("nine recent origins", T[T.origin.isin(REC)]), ("all 13 origins", T), ("recent, 100 targets", T[T.origin.isin(REC) & T.target])]:
    print(f"\n######## {nm}"); print(sl.table(Z, COLS, D, NM).round(4).to_string())
Z = T[T.origin.isin(REC)]
print("\nbootstrap WAPE(x) - WAPE(y), recent:")
for a, b in [(c, "qmodel") for c in ["butw", "bumtw"] if c in T] + [(f"r_{c}_nn", "r_q_nn") for c in ["butw", "bumtw"] if c in T] + [(f"four_{c}", "r_q_nn") for c in ["butw", "bumtw"] if c in T]:
    print(f"  {a} - {b}:", np.round(sl.boot(Z, a, b), 4))
if os.path.exists("data/bum_items_tw.parquet"):
    B = pd.read_parquet("data/bum_items_tw.parquet")
    b = pd.read_parquet("data/best_backtest_preds.parquet")[["origin", "style_color", "h", "pred", "y"]].rename(columns={"pred": "lgb"})
    n = pd.read_parquet("data/nn_backtest_preds_S13x6.parquet")[["origin", "style_color", "h", "q50"]].rename(columns={"q50": "nn"})
    X = b.merge(n, on=["origin", "style_color", "h"]).merge(B[["origin", "style_color", "h", "bum"]], on=["origin", "style_color", "h"], how="left"); RR = []
    for o in X.origin.unique():
        r = pd.read_csv(f"agent_eval/{o}/predictions_verified.csv", dtype={"month": str}); r["h"] = [(pd.Period(m, "M") - pd.Period(o, "M")).n for m in r.month]; RR.append(r.assign(origin=o))
    X = X.merge(pd.concat(RR)[["origin", "style_color", "h", "pred"]].rename(columns={"pred": "rules"}), on=["origin", "style_color", "h"])
    X["bum"] = X.bum.fillna(X.lgb)
    for c in ["lgb", "nn", "rules", "bum"]: X[c] = X[c].clip(lower=0.01)
    X["geo_r_nn"] = np.exp((np.log(X.nn) + np.log(X.rules)) / 2); X["geo_r_nn_bum"] = np.exp((np.log(X.nn) + np.log(X.rules) + np.log(X.bum)) / 3)
    g = X.groupby(["origin", "style_color"]); C = ["lgb", "nn", "rules", "bum", "geo_r_nn", "geo_r_nn_bum"]
    for c in C: X[c + "_s"] = X[c] / g[c].transform("sum") * g.y.transform("sum")
    XR = X[X.origin >= "2025-06"]; w = lambda c: (XR[c] - XR.y).abs().sum() / XR.y.sum()
    print("\n== monthly, recent origins: item-month WAPE (level + shape) and shape error (true total given)")
    print(pd.DataFrame({"item-month WAPE": {c: w(c) for c in C}, "shape error": {c: w(c + "_s") for c in C}}).round(4).to_string())
    print("aggregate bias by target month, fine bottom-up:", (XR.assign(tm=[str(pd.Period(o, 'M') + h) for o, h in zip(XR.origin, XR.h)]).groupby("tm")[["bum", "y"]].sum().pipe(lambda t: t.bum / t.y - 1)).round(3).to_dict())
