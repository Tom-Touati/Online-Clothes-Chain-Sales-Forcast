"""Does a candidate model remove the segment biases that the hand rules fix? Bias and WAPE of 4-month totals by segment
(swimwear, Homecoming/Halloween items under a year old, age groups <3m / 3-12m / 12m+), per origin and averaged, for LightGBM,
the candidate(s) and the 50/50 blend. Usage: python eda/64_segment_compare.py file1 [file2 ...] (q50 column)."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
it = pd.read_parquet("data/items.parquet").set_index("style_color"); occ = it.occasion.fillna("")
B = pd.read_parquet("data/best_backtest_preds.parquet").rename(columns={"pred": "lgb"})
F = pd.read_parquet("data/features.parquet", columns=["origin", "style_color", "h", "age_m"])
X = B.merge(F, on=["origin", "style_color", "h"])
names = []
for f in sys.argv[1:]:
    x = pd.read_parquet(f"data/{f}.parquet")[["origin", "style_color", "h", "q50"]].rename(columns={"q50": f}); X = X.merge(x, on=["origin", "style_color", "h"]); names.append(f)
    X[f"blend:{f}"] = 0.5 * X.lgb + 0.5 * X[f]; names.append(f"blend:{f}")
G = X.groupby(["origin", "style_color"]).agg(**{c: (c, "sum") for c in ["lgb", "y"] + names}, age=("age_m", "first")).reset_index()
pt = it.product_type.reindex(G.style_color).values; oc = occ.reindex(G.style_color).values
G["seg"] = np.where(np.isin(pt, ["Swimwear Top", "Swimwear Bottom"]), "swimwear",
           np.where(pd.Series(oc).str.contains("Homecoming|Halloween").values & (G.age < 12), "Hc/Hw <12m",
           np.where(G.age < 3, "<3m", np.where(G.age < 12, "3-12m", "12m+"))))
cols = ["lgb"] + names
rows = []
for (o, s), g in G.groupby(["origin", "seg"]):
    for c in cols: rows.append(dict(origin=o, seg=s, model=c, n=len(g), bias=g[c].sum() / g.y.sum() - 1, wape=(g[c] - g.y).abs().sum() / g.y.sum()))
R = pd.DataFrame(rows)
print("== mean over origins (bias / WAPE of 4-month totals)")
print(R.groupby(["seg", "model"])[["n", "bias", "wape"]].mean().round(3).unstack("model").to_string())
summer = R[R.origin.isin(["2025-05", "2025-06", "2025-07"])]
print("\n== summer origins 2025-05..07 (the deliverable's situation)")
print(summer.groupby(["seg", "model"])[["bias", "wape"]].mean().round(3).unstack("model").to_string())
