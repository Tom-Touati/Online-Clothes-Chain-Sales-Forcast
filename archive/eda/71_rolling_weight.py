"""Time-honest blend weight: at each origin the NN weight w in {0, .25, .5, .75, 1} is the one with the lowest mean WAPE on earlier
origins whose 4-month outcome is known (o' + 4 <= o); 0.5 until at least 2 such origins exist. Production level (quantile-model
median re-centred on the geometric blend, D085) and monthly level (LightGBM + NN P50), 6-seed NN."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
W = [0, .25, .5, .75, 1.0]; T = set(pd.read_csv("data/targets.csv").style_color)
NM = pd.read_parquet("data/nn_backtest_preds_B13x6.parquet"); B = pd.read_parquet("data/best_backtest_preds.parquet")
X = B.merge(NM[["origin", "style_color", "h", "q50"]], on=["origin", "style_color", "h"]).groupby(["origin", "style_color"])[["pred", "q50", "y"]].sum().reset_index()
Q = pd.read_parquet("data/q_backtest.parquet")[["origin", "style_color", "q50", "ytot"]].rename(columns={"q50": "qm"}).merge(X, on=["origin", "style_color"])
def score(f, a): return np.abs(f - a).sum() / a.sum()
for lvl, base, act in [("production (quantile model)", "qm", "ytot"), ("monthly (LightGBM)", "pred", "y")]:
    fw = lambda d, w: np.expm1((1 - w) * np.log1p(d[base]) + w * np.log1p(d.q50)) if lvl.startswith("prod") else (1 - w) * d[base] + w * d.q50
    tab = pd.DataFrame({w: Q.groupby("origin").apply(lambda d: score(fw(d, w), d[act])) for w in W})
    tabt = pd.DataFrame({w: Q[Q.style_color.isin(T)].groupby("origin").apply(lambda d: score(fw(d, w), d[act])) for w in W})
    rows = []
    for o in tab.index:
        oo = pd.Period(o).ordinal; known = [x for x in tab.index if pd.Period(x).ordinal + 4 <= oo]
        w = 0.5 if len(known) < 2 else tab.loc[known].mean().idxmin()
        rows.append(dict(origin=o, w=w, roll=tab.loc[o, w], fixed50=tab.loc[o, 0.5], base=tab.loc[o, 0], roll_t=tabt.loc[o, w], fixed50_t=tabt.loc[o, 0.5], base_t=tabt.loc[o, 0]))
    R = pd.DataFrame(rows).set_index("origin"); rep = R.index >= "2025-06"
    print(f"\n== {lvl}: chosen weights", R.w.to_dict())
    print("mean WAPE 13 origins  base / fixed 0.5 / rolling:", R[["base", "fixed50", "roll"]].mean().round(4).tolist(), "| targets:", R[["base_t", "fixed50_t", "roll_t"]].mean().round(4).tolist())
    print("mean WAPE 9 rep. orig base / fixed 0.5 / rolling:", R[rep][["base", "fixed50", "roll"]].mean().round(4).tolist(), "| targets:", R[rep][["base_t", "fixed50_t", "roll_t"]].mean().round(4).tolist())
    print("hindsight best fixed weight (all 13):", tab.mean().round(4).to_dict())
