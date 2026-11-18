"""Bias by target month and horizon (D123): item-month level, members' monthly forecasts (rules, LightGBM, NN) and their geometric
mean (the three-way's monthly shape at the item level), recent and all origins. Also the knowable November signal: last year's
like-for-like November / (Jul-Sep mean) ratio on items live through both years, against what each member implied at each origin."""
import numpy as np, pandas as pd, sys
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
b = pd.read_parquet("data/best_backtest_preds.parquet")[["origin", "style_color", "h", "pred", "y"]].rename(columns={"pred": "lgb"})
n = pd.read_parquet("data/nn_backtest_preds_S13x6.parquet")[["origin", "style_color", "h", "q50"]].rename(columns={"q50": "nn"})
X = b.merge(n, on=["origin", "style_color", "h"]); R = []
for o in X.origin.unique():
    r = pd.read_csv(f"agent_eval/{o}/predictions_verified.csv", dtype={"month": str}); r["h"] = [(pd.Period(m, "M") - pd.Period(o, "M")).n for m in r.month]; R.append(r.assign(origin=o))
X = X.merge(pd.concat(R)[["origin", "style_color", "h", "pred"]].rename(columns={"pred": "rules"}), on=["origin", "style_color", "h"])
X["geo"] = np.exp((np.log(X.lgb.clip(lower=.01)) + np.log(X.nn.clip(lower=.01)) + np.log(X.rules.clip(lower=.01))) / 3)
X["tm"] = [str(pd.Period(o, "M") + h) for o, h in zip(X.origin, X.h)]
g = X.groupby("tm")[["y", "lgb", "nn", "rules", "geo"]].sum()
print("== aggregate bias by target month (all origins and horizons pooled)"); print((g[["lgb", "nn", "rules", "geo"]].div(g.y, axis=0) - 1).round(3).to_string())
g2 = X[X.tm == "2025-11"].groupby("h")[["y", "lgb", "nn", "rules", "geo"]].sum()
print("\n== November 2025 by horizon (h = months ahead)"); print((g2[["lgb", "nn", "rules", "geo"]].div(g2.y, axis=0) - 1).round(3).to_string())
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
lfl = M[(M[["2024-07", "2024-08", "2024-09"]].min(axis=1) > 0)]
print("\nknowable last year: like-for-like November 2024 / mean(Jul-Sep 2024) =", round(lfl["2024-11"].sum() / lfl[["2024-07", "2024-08", "2024-09"]].sum().sum() * 3, 3),
      "| November 2024 / October 2024 =", round(lfl["2024-11"].sum() / lfl["2024-10"].sum(), 3))
l2 = M[(M[["2025-07", "2025-08", "2025-09"]].min(axis=1) > 0)]
print("realised this year:   like-for-like November 2025 / mean(Jul-Sep 2025) =", round(l2["2025-11"].sum() / l2[["2025-07", "2025-08", "2025-09"]].sum().sum() * 3, 3),
      "| November 2025 / October 2025 =", round(l2["2025-11"].sum() / l2["2025-10"].sum(), 3))
for o in ["2025-07", "2025-08", "2025-09", "2025-10"]:
    x = X[X.origin == o]; nov = x[x.tm == "2025-11"]; hist = M.loc[x.style_color.unique(), [str(pd.Period(o, "M") - k) for k in range(3)]].sum().sum() / 3
    print(f"origin {o}: implied November / mean of last 3 months (forecast set) lgb {nov.lgb.sum() / hist:.2f} nn {nov.nn.sum() / hist:.2f} rules {nov.rules.sum() / hist:.2f} | actual {nov.y.sum() / hist:.2f}")
