"""Multi-origin robustness for D060-D062.
1 Peer-ratio method vs model for items 3-12 months old, at origins May-Aug 2025, peers taken from the same month one year earlier (by peak position).
2 Mature-item 50/50 blend (model + own trend x ageing fitted on earlier origins) per origin, all origins with a year of history.
3 Floors at the July 2025 origin: Aug-Nov / July ratios of 2024 Homecoming / Halloween items, floor at the 2024 median, applied to young items."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 230)
M = pd.read_parquet("data/monthly_item.parquet"); cols = list(M.columns)
it = pd.read_parquet("data/items.parquet").set_index("style_color")
start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
occ = it.occasion.fillna(""); Q = pd.read_parquet("data/q_backtest.parquet")
def met(f, a):
    f, a = np.asarray(f, float), np.asarray(a, float); ape = np.abs(f - a) / np.clip(a, 1, None)
    return dict(n=len(a), bias=round(f.sum() / a.sum() - 1, 3), wape=round(np.abs(f - a).sum() / a.sum(), 3), w20=round((ape <= .2).mean(), 3), w50=round((ape <= .5).mean(), 3))
def young(o, min_last=50):
    k = cols.index(o); age = (pd.Period(o).end_time - start.reindex(M.index)).dt.days / 30.44
    return M.index[((age >= 3) & (age < 12)).values & (M[o] >= min_last).values]
def pos(ids, o):
    k = cols.index(o); r = M.loc[ids, o] / M.loc[ids, cols[:k + 1]].max(axis=1)
    return pd.cut(r, [0, .35, .7, 10], labels=["<35%", "35-70%", ">=70%"])
print("== 1 peer-ratio vs model, items 3-12 months old")
rows = []
for o in ["2025-05", "2025-06", "2025-07", "2025-08"]:
    k = cols.index(o); oly = cols[k - 12]; nxt = lambda kk: cols[kk + 1:kk + 5]
    p = young(oly); bp = pos(p, oly)
    peer = (M.loc[p, nxt(k - 12)].sum(axis=1) / M.loc[p, oly]).groupby(bp.values).median()
    g = Q[Q.origin == o].set_index("style_color"); ids = [s for s in young(o) if s in g.index]; b = pos(pd.Index(ids), o)
    pr = M.loc[ids, o].values * peer.reindex(b.values).values; a = g.loc[ids, "ytot"].values
    rows.append(dict(origin=o, method="model", **met(g.loc[ids, "q50"], a))); rows.append(dict(origin=o, method="peer ratio", **met(pr, a)))
R = pd.DataFrame(rows); print(R.to_string(index=False))
print("mean over origins:"); print(R.groupby("method")[["bias", "wape", "w20", "w50"]].mean().round(3).to_string())

print("\n== 2 mature-item blend, per origin")
src = open("eda/31_yoy_test.py", encoding="utf-8").read(); src = src[:src.index("def met(")]; ns = {}; exec(src, ns); D = ns["D"]
rows = []
for o, g in D.groupby("origin"):
    past = D[D.oo + 4 <= pd.Period(o).ordinal]
    if len(past) < 50: continue
    age = np.exp(np.median(np.log((past.a + 5) / (past.yn + 5)))); b = np.exp(.5 * np.log(g.p50 + 1) + .5 * np.log(g.yn * age + 1)) - 1
    rows.append(dict(origin=o, ageing=round(age, 3), model_bias=met(g.p50, g.a)["bias"], blend_bias=met(b, g.a)["bias"], model_w20=met(g.p50, g.a)["w20"], blend_w20=met(b, g.a)["w20"],
                     blend_vs_actual_ratio=round(b.sum() / g.ly.sum(), 3), actual_ratio=round(g.a.sum() / g.ly.sum(), 3)))
B = pd.DataFrame(rows); print(B.to_string(index=False))
print(f"origins: {len(B)} | blend |bias| <= 10% in {int((B.blend_bias.abs() <= .10).sum())} | blend w20 >= model w20 in {int((B.blend_w20 >= B.model_w20).sum())} | mean blend bias {B.blend_bias.mean():+.3f}")

print("\n== 3 floors at the July 2025 origin (Aug-Nov / July)")
seg = lambda ids, s: occ.reindex(ids).str.contains(s).values
live = start[start < pd.Timestamp("2024-06-15")].index.intersection(M.index); live = live[M.loc[live, "2024-07"] >= 50]
g = Q[Q.origin == "2025-07"].set_index("style_color")
age = (pd.Timestamp("2025-07-31") - start.reindex(g.index)).dt.days / 30.44
for s in ["Homecoming", "Halloween"]:
    r24 = (M.loc[live, ["2024-08", "2024-09", "2024-10", "2024-11"]].sum(axis=1) / M.loc[live, "2024-07"])[seg(live, s)]
    ids = g.index[(age < 15).values & seg(g.index, s) & (M.loc[g.index, "2025-07"] >= 50).values]
    fl = np.maximum(g.loc[ids, "q50"], M.loc[ids, "2025-07"] * r24.median())
    print(f"{s}: 2024 n={len(r24)}, floor {r24.median():.2f}x July | young items n={len(ids)} | model {met(g.loc[ids, 'q50'], g.loc[ids, 'ytot'])} | with floor {met(fl, g.loc[ids, 'ytot'])}")
