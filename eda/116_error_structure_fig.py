"""Error-structure figure (D123): six panels on the three-way combination's errors, recent origins unless noted -> eda/figs/err_structure.png
 1 aggregate bias by target month for each member and the three-way's monthly shape (all origins)
 2 bias heatmap, origin x product group (three-way, 4-month totals)
 3 log error vs member disagreement (hexbin) with the binned median and 10-90% band
 4 Black Friday: this year's item lift vs last year's (log-log), items live in both years
 5 WAPE / R3 by age band x volume quintile
 6 concentration: cumulative share of absolute error by item-origin rank"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
A = pd.read_parquet("data/archetype_items.parquet"); R = A[A.origin >= "2025-06"].copy()
b = pd.read_parquet("data/best_backtest_preds.parquet")[["origin", "style_color", "h", "pred", "y"]].rename(columns={"pred": "lgb"})
n = pd.read_parquet("data/nn_backtest_preds_S13x6.parquet")[["origin", "style_color", "h", "q50"]].rename(columns={"q50": "nn"})
X = b.merge(n, on=["origin", "style_color", "h"]); RR = []
for o in X.origin.unique():
    r = pd.read_csv(f"agent_eval/{o}/predictions_verified.csv", dtype={"month": str}); r["h"] = [(pd.Period(m, "M") - pd.Period(o, "M")).n for m in r.month]; RR.append(r.assign(origin=o))
X = X.merge(pd.concat(RR)[["origin", "style_color", "h", "pred"]].rename(columns={"pred": "rules"}), on=["origin", "style_color", "h"])
X["geo"] = np.exp((np.log(X.lgb.clip(lower=.01)) + np.log(X.nn.clip(lower=.01)) + np.log(X.rules.clip(lower=.01))) / 3)
X["tm"] = [str(pd.Period(o, "M") + h) for o, h in zip(X.origin, X.h)]
fig, ax = plt.subplots(2, 3, figsize=(22, 12)); ax = ax.ravel()
g = X.groupby("tm")[["y", "lgb", "nn", "rules", "geo"]].sum(); bias = g[["lgb", "nn", "rules", "geo"]].div(g.y, axis=0) - 1
for c, col in [("lgb", "purple"), ("nn", "red"), ("rules", "green"), ("geo", "blue")]: ax[0].plot(bias.index, bias[c], marker="o", color=col, lw=2.2 if c == "geo" else 1, label=c)
ax[0].axhline(0, color="k", lw=0.6); ax[0].set_title("1  Aggregate bias by target month (all origins, horizons pooled)"); ax[0].legend(); ax[0].tick_params(axis="x", rotation=60)
hm = R.groupby(["origin", "macro"]).apply(lambda x: x.r_q_nn.sum() / x.y.sum() - 1).unstack()
im = ax[1].imshow(hm.values, cmap="RdBu", vmin=-0.5, vmax=0.5, aspect="auto"); ax[1].set_xticks(range(hm.shape[1])); ax[1].set_xticklabels(hm.columns)
ax[1].set_yticks(range(hm.shape[0])); ax[1].set_yticklabels(hm.index)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]): ax[1].text(j, i, f"{hm.values[i, j]:+.0%}", ha="center", va="center", fontsize=8)
ax[1].set_title("2  Bias of the three-way, origin x product group (red = over, blue = under)"); plt.colorbar(im, ax=ax[1])
L = np.log(R[["rules", "qmodel", "nn", "lgb"]].clip(lower=0.5)); dis = L.std(axis=1); le = np.log((R.r_q_nn + 1) / (R.y + 1))
ax[2].hexbin(dis, le, gridsize=50, bins="log", cmap="Greys", extent=(0, 1.5, -3, 3)); bb = pd.cut(dis, np.arange(0, 1.6, 0.1))
st = le.groupby(bb).quantile([0.1, 0.5, 0.9]).unstack(); xc = [i.mid for i in st.index]
ax[2].plot(xc, st[0.5], color="blue", lw=2); ax[2].fill_between(xc, st[0.1], st[0.9], color="blue", alpha=0.15)
ax[2].set_xlabel("member disagreement (sd of log forecasts)"); ax[2].set_ylabel("log error (pred / actual)"); ax[2].axhline(0, color="k", lw=0.6)
ax[2].set_title("3  Error grows with disagreement; centre stays near zero (median, 10-90% band)")
Bf = A[A.bflift.notna() & A.bflift_ly.notna()].drop_duplicates("style_color")
ax[3].scatter(np.log(Bf.bflift_ly.clip(lower=.1)), np.log(Bf.bflift.clip(lower=.1)), s=6, alpha=0.4); ax[3].plot([-1, 4], [-1, 4], color="k", lw=0.6)
ax[3].set_xlabel("log Black Friday lift, 2024"); ax[3].set_ylabel("log Black Friday lift, 2025")
ax[3].set_title(f"4  Black Friday lift: average repeats (median ratio {np.median(Bf.bflift / Bf.bflift_ly):.2f}), item level does not (r = {np.corrcoef(np.log(Bf.bflift.clip(lower=.1)), np.log(Bf.bflift_ly.clip(lower=.1)))[0, 1]:.2f})")
R["vq"] = pd.qcut(R.level.rank(method="first"), 5, labels=["v1", "v2", "v3", "v4", "v5"])
w = R.groupby(["age", "vq"]).apply(lambda x: (x.r_q_nn - x.y).abs().sum() / (x.R3 - x.y).abs().sum()).unstack()
im2 = ax[4].imshow(w.values, cmap="viridis_r", vmin=0.3, vmax=0.8, aspect="auto"); ax[4].set_xticks(range(5)); ax[4].set_xticklabels(["smallest", "v2", "v3", "v4", "largest"])
ax[4].set_yticks(range(w.shape[0])); ax[4].set_yticklabels(w.index)
for i in range(w.shape[0]):
    for j in range(w.shape[1]): ax[4].text(j, i, f"{w.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
ax[4].set_title("5  WAPE / R3 by age x volume (lower = more value over recency)"); plt.colorbar(im2, ax=ax[4])
e = (R.r_q_nn - R.y).abs().sort_values(ascending=False); cs = e.cumsum() / e.sum(); xs = np.arange(1, len(cs) + 1) / len(cs)
ax[5].plot(xs, cs.values, lw=2); ax[5].axhline(0.5, color="grey", lw=0.6); ax[5].axvline(xs[np.searchsorted(cs.values, 0.5)], color="grey", lw=0.6)
ax[5].set_xlabel("share of item-origins (largest error first)"); ax[5].set_ylabel("share of absolute error")
ax[5].set_title(f"6  Concentration: {xs[np.searchsorted(cs.values, 0.5)]:.0%} of item-origins carry half the error, {xs[np.searchsorted(cs.values, 0.8)]:.0%} carry 80%")
fig.tight_layout(); fig.savefig("eda/figs/err_structure.png", dpi=60); print("saved")
