"""Backtest of a year-on-year benchmark for items with a full year of own history.
YN = units in the same 4 months last year x (last 3 months / same 3 months last year). Aging factor fitted on earlier origins (ratio actual/YN).
Compared with the quantile median (P50) and geometric blends, per origin, for items live >= 15 months before the origin."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 230)
Q = pd.read_parquet("data/q_backtest.parquet"); M = pd.read_parquet("data/monthly_item.parquet"); cols = list(M.columns)
it = pd.read_parquet("data/items.parquet").set_index("style_color")
start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
occ = it.occasion.fillna("")
rows = []
for o, g in Q.groupby("origin"):
    k = cols.index(o)
    if k < 14: continue
    ids = g.style_color.values; old = (start.reindex(ids) < pd.Period(cols[k - 14]).start_time).values
    ly = M.loc[ids, cols[k - 11:k - 7]].sum(axis=1).values
    r3 = M.loc[ids, cols[k - 2:k + 1]].sum(axis=1).values; r3ly = M.loc[ids, cols[k - 14:k - 11]].sum(axis=1).values
    yn = ly * (r3 + 5) / (r3ly + 5)
    d = pd.DataFrame(dict(origin=o, sc=ids, p50=g.q50.values, yn=yn, a=g.ytot.values, old=old, ly=ly)); rows.append(d[d.old & (d.ly >= 50)])
D = pd.concat(rows); D["oo"] = D.origin.map(lambda s: pd.Period(s, freq="M").ordinal)
D["tag"] = np.where(occ.reindex(D.sc).str.contains("Halloween|Homecoming").values, "Halloween/Homecoming", "other")
print("item-origins with a full year of own history:", len(D), "| by origin:", D.groupby("origin").size().to_dict())
def met(f, a):
    ape = np.abs(f - a) / np.clip(a, 1, None); return dict(wape=np.abs(f - a).sum() / a.sum(), bias=f.sum() / a.sum() - 1, w20=(ape <= .2).mean(), w50=(ape <= .5).mean())
out = []
for o, g in D.groupby("origin"):
    past = D[D.oo + 4 <= pd.Period(o).ordinal]
    age = np.exp(np.median(np.log((past.a + 5) / (past.yn + 5)))) if len(past) > 50 else np.nan     # aging factor from earlier origins
    if np.isnan(age): continue
    yna = g.yn * age
    for nm, f in [("P50 model", g.p50), ("YoY naive x aging", yna), ("blend 25% YoY", np.exp(.75 * np.log(g.p50 + 1) + .25 * np.log(yna + 1)) - 1),
                  ("blend 50% YoY", np.exp(.5 * np.log(g.p50 + 1) + .5 * np.log(yna + 1)) - 1)]:
        out.append(dict(origin=o, model=nm, age=age, n=len(g), **met(f.values, g.a.values)))
        for t, gt in g.groupby("tag"):
            ff = f[gt.index] if hasattr(f, "index") else f
            out.append(dict(origin=o, model=nm + " | " + t, age=age, n=len(gt), **met(ff.values, gt.a.values)))
R = pd.DataFrame(out)
base = R[R.model == "P50 model"].set_index("origin").wape
T = R.groupby("model")[["n", "wape", "bias", "w20", "w50"]].mean()
T["origins_beating_P50"] = [int((R[R.model == m].set_index("origin").wape < R[R.model == (m.split(" | ")[0].replace(m.split(" | ")[0], "P50 model") + (" | " + m.split(" | ")[1] if " | " in m else ""))].set_index("origin").wape.reindex(R[R.model == m].origin)).sum()) for m in T.index]
print(f"\n{R.origin.nunique()} origins (aging factor from earlier origins only):"); print(T.round(3).to_string())
print("aging factor by origin:", R.drop_duplicates("origin").set_index("origin").age.round(3).to_dict())
J = R[R.origin.isin(["2025-06", "2025-07"])]; print("\nJune and July 2025 origins:"); print(J.groupby("model")[["n", "wape", "bias", "w20", "w50"]].mean().round(3).to_string())
