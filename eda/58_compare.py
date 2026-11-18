"""Paired comparison of backtest prediction files against the LightGBM ensemble on the 13 rolling origins.
Item-level: WAPE / bias / within-20% on 4-month totals (all items and the 100 targets). Month-level: aggregate error of each month.
Usage: python eda/58_compare.py nn_backtest_preds[_tag] ..."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
T = set(pd.read_csv("data/targets.csv").style_color)
B = pd.read_parquet("data/best_backtest_preds.parquet").rename(columns={"pred": "lgb"})
P = {"lgb": B[["origin", "style_color", "h", "lgb", "y"]]}
for f in sys.argv[1:]:
    x = pd.read_parquet(f"data/{f}.parquet"); P[f] = x[["origin", "style_color", "h", "q50"]].rename(columns={"q50": f})
X = P["lgb"]
for k, v in P.items():
    if k != "lgb": X = X.merge(v, on=["origin", "style_color", "h"], how="inner")
for k in sys.argv[1:]: X[f"blend:{k}"] = 0.5 * X.lgb + 0.5 * X[k]
cols = [c for c in X.columns if c not in ("origin", "style_color", "h", "y")]
def sc(g):
    t = g.groupby("style_color")[cols + ["y"]].sum(); r = {}
    for c in cols:
        ape = (t[c] - t.y).abs() / t.y.clip(lower=1)
        r[(c, "wape")] = (t[c] - t.y).abs().sum() / t.y.sum(); r[(c, "bias")] = t[c].sum() / t.y.sum() - 1; r[(c, "w20")] = (ape <= .2).mean()
    return pd.Series(r)
for seg, m in [("all", np.ones(len(X), bool)), ("targets", X.style_color.isin(T).values)]:
    R = X[m].groupby("origin").apply(sc)
    print(f"\n== {seg}: per origin WAPE"); print(R.xs("wape", axis=1, level=1).round(3).to_string())
    S = pd.DataFrame({"wape": R.xs("wape", axis=1, level=1).mean(), "bias": R.xs("bias", axis=1, level=1).mean(), "w20": R.xs("w20", axis=1, level=1).mean(),
                      "origins_beating_lgb": (R.xs("wape", axis=1, level=1).lt(R[("lgb", "wape")], axis=0)).sum()})
    print(S.round(3).to_string())
X["tm"] = (pd.PeriodIndex(X.origin, freq="M") + X.h).astype(str)
A = X.groupby(["tm", "h"])[cols + ["y"]].sum(); E = np.log(A[cols].div(A.y, axis=0))
print("\n== month-level aggregate |log error| (all items), mean over target months, by horizon"); print(E.abs().groupby(level="h").mean().round(3).to_string())
print("worst months (h=1, log error):"); print(E.xs(1, level="h").round(2).to_string())
