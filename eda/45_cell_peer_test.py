"""Out-of-sample test of the mix-matched peer method (macro category x peak position cells, peers = items 3-12 months old at the same
origin month one year earlier) against the model, for items 3-12 months old, at origins May-Aug 2025. Also 25% and 50% geometric blends."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
src = open("eda/44_mix_matched_july.py", encoding="utf-8").read(); src = src[:src.index("age25 =")]; ns = {}; exec(src, ns)
M, cols, start, it, MAC = ns["M"], ns["cols"], ns["start"], ns["it"], ns["MAC"]
Q = pd.read_parquet("data/q_backtest.parquet")
def met(f, a):
    f, a = np.asarray(f, float), np.asarray(a, float); ape = np.abs(f - a) / np.clip(a, 1, None)
    return dict(bias=round(f.sum() / a.sum() - 1, 3), wape=round(np.abs(f - a).sum() / a.sum(), 3), w20=round((ape <= .2).mean(), 3), w50=round((ape <= .5).mean(), 3))
def cell(ids, o):
    k = cols.index(o); pos = M.loc[ids, o] / M.loc[ids, cols[:k + 1]].max(axis=1)
    return pd.Series(it.product_type.reindex(ids).map(MAC).fillna("other").values, index=ids) + "|" + pd.cut(pos, [0, .35, .7, 10], labels=["<35", "35-70", ">=70"]).astype(str)
def young(o):
    age = (pd.Period(o).end_time - start.reindex(M.index)).dt.days / 30.44
    return M.index[((age >= 3) & (age < 12)).values & (M[o] >= 50).values]
rows = []
for o in ["2025-05", "2025-06", "2025-07", "2025-08"]:
    k = cols.index(o); oly = cols[k - 12]
    p = young(oly); cp = cell(p, oly); r = (M.loc[p, cols[k - 11:k - 7]].sum(axis=1) / M.loc[p, oly])
    agg = pd.DataFrame({"c": cp, "num": M.loc[p, cols[k - 11:k - 7]].sum(axis=1), "den": M.loc[p, oly]}).groupby("c").agg(num=("num", "sum"), den=("den", "sum"), n=("num", "size"))
    ratio = (agg.num / agg.den)[agg.n >= 3]; fallback = agg.num.sum() / agg.den.sum()
    g = Q[Q.origin == o].set_index("style_color"); ids = [s for s in young(o) if s in g.index]; ci = cell(pd.Index(ids), o)
    peer = M.loc[ids, o].values * ci.map(ratio).fillna(fallback).values; a = g.loc[ids, "ytot"].values; m = g.loc[ids, "q50"].values
    for nm, f in [("model", m), ("cell peers", peer), ("blend 25% peers", np.exp(.75 * np.log(m + 1) + .25 * np.log(peer + 1)) - 1), ("blend 50% peers", np.exp(.5 * np.log(m + 1) + .5 * np.log(peer + 1)) - 1)]:
        rows.append(dict(origin=o, method=nm, n=len(ids), **met(f, a)))
R = pd.DataFrame(rows); print(R.to_string(index=False)); print("\nmean over the four origins:"); print(R.groupby("method")[["bias", "wape", "w20", "w50"]].mean().round(3).to_string())
