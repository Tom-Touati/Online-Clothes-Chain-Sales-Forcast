"""Forecast stability (D142): for the same item and target month, how much does the forecast move between consecutive origins (made at
h + 1 and then at h months ahead)? Median and 90th percentile of |log((F_h + 1) / (F_h+1 + 1))|, and whether the revision moved toward
the actual (share of revisions that reduced the error). Members' monthly forecasts, their geometric mean, the recursive trees."""
import numpy as np, pandas as pd, sys
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
b = pd.read_parquet("data/best_backtest_preds.parquet")[["origin", "style_color", "h", "pred", "y"]].rename(columns={"pred": "lgb"})
n = pd.read_parquet("data/nn_backtest_preds_S13x6.parquet")[["origin", "style_color", "h", "q50"]].rename(columns={"q50": "nn"})
rc = pd.read_parquet("data/rec_backtest_preds.parquet")[["origin", "style_color", "h", "recursive"]]
X = b.merge(n, on=["origin", "style_color", "h"]).merge(rc, on=["origin", "style_color", "h"]); RR = []
for o in X.origin.unique():
    r = pd.read_csv(f"agent_eval/{o}/predictions_verified.csv", dtype={"month": str}); r["h"] = [(pd.Period(m, "M") - pd.Period(o, "M")).n for m in r.month]; RR.append(r.assign(origin=o))
X = X.merge(pd.concat(RR)[["origin", "style_color", "h", "pred"]].rename(columns={"pred": "rules"}), on=["origin", "style_color", "h"])
X["geo"] = np.exp((np.log(X.lgb.clip(lower=.01)) + np.log(X.nn.clip(lower=.01)) + np.log(X.rules.clip(lower=.01))) / 3)
X["tm"] = [str(pd.Period(o, "M") + h) for o, h in zip(X.origin, X.h)]
C = ["lgb", "nn", "rules", "geo", "recursive"]
A = X.merge(X[["style_color", "tm", "h"] + C], on=["style_color", "tm"], suffixes=("", "_prev"))
A = A[A.h_prev == A.h + 1]; A = A[A.tm >= "2025-07"]
rows = {}
for c in C:
    d = np.abs(np.log((A[c] + 1) / (A[c + "_prev"] + 1))); better = np.abs(np.log((A[c] + 1) / (A.y + 1))) < np.abs(np.log((A[c + "_prev"] + 1) / (A.y + 1)))
    rows[c] = {"median |revision|": d.median(), "p90 |revision|": d.quantile(0.9), "revisions toward the actual": better.mean(), "pairs": len(A)}
print("== revisions of the same item-month between consecutive origins (target months Jul 2025 on)"); print(pd.DataFrame(rows).T.round(3).to_string())
