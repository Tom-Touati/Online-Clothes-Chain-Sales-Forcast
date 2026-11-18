"""How much demand is hidden by stock-outs? (No inventory data, so inferred from sales.)
A) item level: runs of zero-sales days that are improbable given the item's own recent rate (Poisson, P < 0.1%, >= 3 days)
B) size level: weeks where a core size sells nothing although the item's other sizes imply it should have sold >= 7 units (P < 0.1%)"""
import sys, duckdb, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
T = set(pd.read_csv("data/targets.csv").style_color)
it = pd.read_parquet("data/items.parquet").set_index("style_color")
d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[~d.cancelled]; d["ny_date"] = pd.to_datetime(d.ny_date)
days = pd.date_range("2024-01-01", "2026-06-30")
D = d.groupby(["style_color", "ny_date"]).qty.sum().unstack(fill_value=0).reindex(columns=days, fill_value=0)

# ---- A) item-level zero runs
flag = np.zeros(D.shape, bool); rows = []
for i, (sc, v) in enumerate(zip(D.index, D.values)):
    nz = np.flatnonzero(v > 0)
    if len(nz) == 0: continue
    z = (v == 0); z[:nz[0]] = False
    edges = np.flatnonzero(np.diff(np.concatenate([[0], z.astype(int), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):          # zero run is [a, b)
        k = b - a; pre = v[max(a - 28, nz[0]):a]; pre = pre[~flag[i, max(a - 28, nz[0]):a]]
        if len(pre) < 7: continue
        lam = pre.mean()
        if k >= 3 and lam * k > 6.9:
            flag[i, a:b] = True; rows.append((sc, days[a], days[b - 1], k, round(lam, 2), b == len(v), sc in T))
G = pd.DataFrame(rows, columns=["style_color", "start", "end", "days", "rate_before", "runs_to_end", "target"])
live = (D.values.cumsum(axis=1) > 0)
print("== A) item-level improbable zero runs")
print("runs:", len(G), "| items affected:", G.style_color.nunique(), "| flagged share of live item-days:", round(flag.sum() / live.sum(), 4),
      "| est. lost units share:", round((G.days * G.rate_before).sum() / D.values.sum(), 4))
print("run length quantiles:", G.days.quantile([.5, .9, .99]).tolist(), "| runs reaching the end of data:", int(G.runs_to_end.sum()), "| target items affected:", G[G.target].style_color.nunique())
print("by year-month of run start (lost units, k):"); print((G.assign(m=G.start.dt.to_period("Q"), lost=G.days * G.rate_before).groupby("m").lost.sum() / 1e3).round(1).to_dict())
print("targets, flagged runs in 2026:"); print(G[G.target & (G.end >= "2026-01-01")].sort_values("rate_before", ascending=False).head(15).to_string(index=False))
pd.DataFrame(flag, index=D.index, columns=days.astype(str)).to_parquet("data/oos_flags_daily.parquet"); G.to_csv("data/oos_runs.csv", index=False)

# ---- B) size-level broken runs (weekly)
con = duckdb.connect(); con.execute("SET memory_limit='1200MB'; SET threads=4")
W = con.execute(f"SELECT style_color, date_trunc('week', ny_date) wk, size, sum(qty_ordered) qty FROM read_parquet('{sys.argv[1]}') WHERE NOT cancelled AND class='ONLINE' GROUP BY ALL").fetchdf()
W["wk"] = pd.to_datetime(W.wk)
tot = W.groupby(["style_color", "wk"]).qty.sum().rename("tot").reset_index()
share = (W.groupby(["style_color", "size"]).qty.sum() / W.groupby("style_color").qty.sum()).rename("share").reset_index()
core = share[share.share >= 0.15]
X = tot.merge(core, on="style_color").merge(W, on=["style_color", "wk", "size"], how="left").fillna({"qty": 0})
X["others"] = X.tot - X.qty; X["exp"] = X.others * X.share / (1 - X.share)      # expected units of this size given the other sizes
X["broken"] = (X.qty == 0) & (X.exp >= 7)
b = X[X.broken]
print("\n== B) size-level: core size sells 0 in a week where >= 7 units were expected (online only)")
print("item-weeks with a broken core size:", b.groupby(["style_color", "wk"]).ngroups, "of", len(tot), f"({b.groupby(['style_color', 'wk']).ngroups / len(tot):.1%})",
      "| est. lost units share of online:", round(b.exp.sum() / W.qty.sum(), 4), "| items ever affected:", b.style_color.nunique())
q = b.assign(q=b.wk.dt.to_period("Q")).groupby("q").exp.sum() / tot.assign(q=tot.wk.dt.to_period("Q")).groupby("q").tot.sum(); print("lost share by quarter:", q.round(3).to_dict())
last13 = X[X.wk >= "2026-03-30"]; lb = last13[last13.broken].groupby("style_color").exp.sum(); lt = last13.drop_duplicates(["style_color", "wk"]).groupby("style_color").tot.sum()
r = (lb / lt).dropna().sort_values(ascending=False); r = r[r.index.isin(T)]
print("targets with broken core sizes in the last 13 weeks:", len(r), "| est. lost share of their online units (top 12):"); print(r.head(12).round(3).to_dict())
al = (lb / lt).dropna(); print("all items, last 13 weeks: affected", len(al), "| median lost share among affected:", round(al.median(), 3), "| share of items losing >10%:", round((al > 0.1).sum() / lt.shape[0], 3))
r.to_csv("data/target_size_gaps_last13w.csv")
