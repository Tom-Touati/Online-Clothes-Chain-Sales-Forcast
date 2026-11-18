"""GENERATED from eda/112 by the D136 test: last-year calibration applied only to the Black Friday month (November) and to November-December.
Original: Last-year seasonal calibration of the monthly level (D124), time-honest (only last year's actuals and the origin's history).
For origin o and horizon h: cohort = forecast-set items that sold in each of the three months o-14..o-12 (live a year ago).
  r_LY = sum over the cohort of units in month o+h-12 / sum of their mean monthly units over o-14..o-12   (last year's like-for-like ratio)
  r_F  = sum over the cohort of the forecast for month o+h / sum of their mean monthly units over o-2..o  (what the forecast implies)
  f_h  = (r_LY / r_F) ** a, applied to every item's month-h forecast; a = 0.5 (damped) or 1.
Forecast = the three-way's monthly geometric mean of rules, LightGBM-split-to-quantile-P50 and NN (scaled so the 4 months sum to the
three-way total), so 4-month totals start equal to the three-way. Scored on 4-month totals (standard table) and item-month WAPE."""
import numpy as np, pandas as pd, sys, importlib.util
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
T = pd.read_parquet("data/fixes_items.parquet")
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
q = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))
LQ = np.log1p(q[["q10", "q25", "q50", "q75", "q90"]].values); D = {c: LQ[:, j] - LQ[:, 2] for j, c in enumerate(["q10", "q25", "q50", "q75", "q90"])}
b = pd.read_parquet("data/best_backtest_preds.parquet")[["origin", "style_color", "h", "pred", "y"]].rename(columns={"pred": "lgb"})
n = pd.read_parquet("data/nn_backtest_preds_S13x6.parquet")[["origin", "style_color", "h", "q50"]].rename(columns={"q50": "nn"})
X = b.merge(n, on=["origin", "style_color", "h"]); R = []
for o in X.origin.unique():
    r = pd.read_csv(f"agent_eval/{o}/predictions_verified.csv", dtype={"month": str}); r["h"] = [(pd.Period(m, "M") - pd.Period(o, "M")).n for m in r.month]; R.append(r.assign(origin=o))
X = X.merge(pd.concat(R)[["origin", "style_color", "h", "pred"]].rename(columns={"pred": "rules"}), on=["origin", "style_color", "h"])
X["geo"] = np.exp((np.log(X.lgb.clip(lower=.01)) + np.log(X.nn.clip(lower=.01)) + np.log(X.rules.clip(lower=.01))) / 3)
X = X.merge(T[["origin", "style_color", "r_q_nn"]], on=["origin", "style_color"])
X["base"] = X.geo / X.groupby(["origin", "style_color"]).geo.transform("sum") * X.r_q_nn
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
P = lambda o, k: str(pd.Period(o, "M") + k); FAC = []
for a in ["nov0.5", "nov1.0", "novdec0.5", "events0.5", "julaug0.5"]: X[f"cal{a}"] = X.base
for o, x in X.groupby("origin"):
    items = x.style_color.unique(); b3 = [P(o, -14), P(o, -13), P(o, -12)]
    if b3[0] not in M.columns: continue
    coh = [s for s in items if (M.loc[s, b3] > 0).all()]
    if len(coh) < 30: continue
    for h in range(1, 5):
        rly = M.loc[coh, P(o, h - 12)].sum() / M.loc[coh, b3].mean(axis=1).sum()
        xs = x[(x.h == h) & x.style_color.isin(coh)]; rf = xs.base.sum() / M.loc[xs.style_color, [P(o, -2), P(o, -1), P(o, 0)]].mean(axis=1).sum()
        FAC.append((o, h, len(coh), round(rly, 3), round(rf, 3)))
        tmn = (pd.Period(o, "M") + h).month
        for a, ms, p in [("nov0.5", (11,), 0.5), ("nov1.0", (11,), 1.0), ("novdec0.5", (11, 12), 0.5), ("events0.5", (7, 8, 11), 0.5), ("julaug0.5", (7, 8), 0.5)]:
            if tmn in ms: X.loc[x.index[x.h.values == h], f"cal{a}"] = x.base[x.h == h] * (rly / rf) ** p
F = pd.DataFrame(FAC, columns=["origin", "h", "cohort", "r_LY", "r_F"]); F["f(a=1)"] = (F.r_LY / F.r_F).round(3)
print(F.pivot(index="origin", columns="h", values="f(a=1)").to_string())
CC = ["calnov0.5", "calnov1.0", "calnovdec0.5", "calevents0.5", "caljulaug0.5"]; tot = X.groupby(["origin", "style_color"])[CC].sum().reset_index()
T = T.merge(tot, on=["origin", "style_color"], how="left")
for c in CC: T[c] = T[c].fillna(T.r_q_nn)
COLS = ["r_q_nn"] + CC; REC = [o for o in sorted(T.origin.unique()) if o >= "2025-06"]
for nm, Z in [("nine recent origins", T[T.origin.isin(REC)]), ("all origins with a cohort", T[T.origin.isin(F.origin.unique())]), ("recent, 100 targets", T[T.origin.isin(REC) & T.target])]:
    print(f"\n######## {nm}"); print(sl.table(Z, COLS, D).round(4).to_string())
Z = T[T.origin.isin(REC)]; print("\nbootstrap WAPE(cal) - WAPE(three-way), recent:", {c: np.round(sl.boot(Z, c, "r_q_nn"), 4).tolist() for c in CC})
XR = X[X.origin >= "2025-06"]
print("item-month WAPE recent:", {c: round((XR[c] - XR.y).abs().sum() / XR.y.sum(), 4) for c in ["base"] + CC})
XR = XR.assign(tm=[P(o, h) for o, h in zip(XR.origin, XR.h)]); g = XR.groupby("tm")[["y", "base"] + CC].sum()
print("bias by target month:"); print((g[["base"] + CC].div(g.y, axis=0) - 1).round(3).T.to_string())

B = T[T.origin.isin(["2025-07", "2025-08", "2025-09", "2025-10"])]
print(chr(10) + "Black Friday windows only (origins Jul-Oct 2025):"); print(sl.table(B.reset_index(drop=True), COLS, {k: v[T.origin.isin(["2025-07", "2025-08", "2025-09", "2025-10"]).values] for k, v in D.items()}).round(4).to_string())
S = T[T.origin.isin(["2025-06", "2025-07"])]
print(chr(10) + "Summer origins (Jun-Jul 2025, windows with back-to-school):")
print(sl.table(S.reset_index(drop=True), COLS, {k: v[T.origin.isin(["2025-06", "2025-07"]).values] for k, v in D.items()}).round(4).to_string())
