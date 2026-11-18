"""Monthly-shape probe (D120): given each item's true 4-month total, how well does each system split it across the four months?
Shape error = item-month WAPE after scaling each system's four monthly values to the actual 4-month total (so only the split is wrong).
Members' monthly forecasts: rules (agent monthly), LightGBM ensemble monthly, NN monthly P50, the quantile model's construction
(LightGBM split scaled to its P50, so its shape = LightGBM's), R3 flat (equal months), R3 x type seasonality shape (eda/06 lfl_type),
and combinations of shapes (geometric mean of member shares). Also the plain item-month WAPE (level + shape) and error by horizon h."""
import numpy as np, pandas as pd, sys
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
b = pd.read_parquet("data/best_backtest_preds.parquet")[["origin", "style_color", "h", "pred", "y"]].rename(columns={"pred": "lgb"})
n = pd.read_parquet("data/nn_backtest_preds_S13x6.parquet")[["origin", "style_color", "h", "q50"]].rename(columns={"q50": "nn"})
X = b.merge(n, on=["origin", "style_color", "h"])
R = []
for o in X.origin.unique():
    r = pd.read_csv(f"agent_eval/{o}/predictions_verified.csv", dtype={"month": str}); r["h"] = [(pd.Period(m, "M") - pd.Period(o, "M")).n for m in r.month]; R.append(r.assign(origin=o))
X = X.merge(pd.concat(R)[["origin", "style_color", "h", "pred"]].rename(columns={"pred": "rules"}), on=["origin", "style_color", "h"])
F = pd.read_parquet("data/features.parquet", columns=["origin", "style_color", "h", "m3", "lfl_type"])
X = X.merge(F, on=["origin", "style_color", "h"], how="left")
X["flat"] = X.m3; X["sea"] = X.m3 * np.exp(X.lfl_type.fillna(0))
for c in ["lgb", "nn", "rules", "flat", "sea"]: X[c] = X[c].clip(lower=0.01)
X["geo3"] = np.exp((np.log(X.lgb) + np.log(X.nn) + np.log(X.rules)) / 3)          # three-way shape (quantile shape = LightGBM's)
X["geo_r_nn"] = np.exp((np.log(X.nn) + np.log(X.rules)) / 2)
X["geo_sea_r"] = np.exp((np.log(X.sea) + np.log(X.rules)) / 2)
C = ["flat", "sea", "lgb", "nn", "rules", "geo3", "geo_r_nn", "geo_sea_r"]
g = X.groupby(["origin", "style_color"])
for c in C: X[c + "_s"] = X[c] / g[c].transform("sum") * g.y.transform("sum")
# production-style shapes: items with a year of own history (>= 50 units in the same 4 months last year) average the model share
# with their own last-year share (eda/32); others use the model share alone
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
X["ly"] = [M.at[s, str(pd.Period(o, "M") + h - 12)] if (s in M.index and str(pd.Period(o, "M") + h - 12) in M.columns) else np.nan for o, s, h in zip(X.origin, X.style_color, X.h)]
X["ly_tot"] = X.groupby(["origin", "style_color"]).ly.transform("sum"); mat = X.ly_tot >= 50
own = X.ly / X.ly_tot
for base in ["lgb", "geo_r_nn", "geo3"]:
    sh = X[base] / g[base].transform("sum"); mix = np.where(mat, 0.5 * sh + 0.5 * own, sh)
    X[f"prod[{base}]_s"] = mix * g.y.transform("sum"); C = C + [f"prod[{base}]"]
print(f"item-origins with a year of own history: {mat.groupby([X.origin, X.style_color]).first().mean():.0%}")
REC = X.origin >= "2025-06"
def w(x, c): return (x[c] - x.y).abs().sum() / x.y.sum()
rows = {}
for nm, x in [("recent", X[REC]), ("all 13", X)]:
    rows[(nm, "shape error (true total)")] = {c: w(x, c + "_s") for c in C}
print(pd.DataFrame(rows).round(4).to_string())
print("\nshape error by target month (recent origins):")
X["tm"] = [str(pd.Period(o, "M") + h)[-2:] for o, h in zip(X.origin, X.h)]
print(pd.DataFrame({c: X[REC].groupby("tm").apply(lambda x: w(x, c + "_s")) for c in ["flat", "sea", "lgb", "nn", "rules", "geo3", "prod[lgb]", "prod[geo_r_nn]"]}).round(3).to_string())
print("\nplain item-month WAPE by horizon (recent; level + shape):")
print(pd.DataFrame({c: X[REC].groupby("h").apply(lambda x: w(x, c)) for c in ["flat", "lgb", "nn", "rules", "geo3"]}).round(3).to_string())

# July-share check at the June 2025 origin (the deliverable's analogue): predicted vs actual July share of Jul-Oct, units-weighted
J = X[X.origin == "2025-06"]
for c in ["lgb", "nn", "rules", "geo_r_nn", "prod[lgb]", "prod[geo_r_nn]"]:
    s = J[c + "_s"]; jul = J.h == 1
    print(f"June 2025 origin, {c:16s} July share predicted {s[jul].sum() / s.sum():.3f} | actual {J.y[jul].sum() / J.y.sum():.3f} | shape error {w(J, c + '_s'):.3f}")
