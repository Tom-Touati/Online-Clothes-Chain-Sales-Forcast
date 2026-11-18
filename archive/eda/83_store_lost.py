"""In-stock uplift, step 1 (D107): daily lost retail units per store x item from store stock-outs.
Runs as in eda/82 test A (strict: >= 3 open days, rate x days > 6.9, rate = mean of the previous 28 open days at the store, >= 7 of them),
now including runs that reach the end of the data. Losses are counted on the first 28 open days of a run only, so an item that has
left a store stops counting after four weeks.
Expected store units on a run day = s / (1 - s) x the item's sales everywhere else that day, where s = the store's share of the item's
network sales in the 28 days before the run (capped at 0.8). This follows demand changes elsewhere.
known = the date the run became improbable (the j-th run day with rate x j > 6.9): a forecast at origin O may use a run day only if
both the day and known are <= O.
Writes data/store_lost_daily.parquet (store, style_color, date, lost, known)."""
import duckdb, numpy as np, pandas as pd
con = duckdb.connect(); con.execute("SET memory_limit='1500MB'; SET threads=4")
ALL = con.execute("SELECT style_color, ny_date::DATE d, CASE WHEN class='RETAIL' THEN store ELSE '_nonretail' END st, sum(qty_ordered) q "
                  "FROM read_parquet('Dataset.parquet') WHERE NOT cancelled GROUP BY ALL").fetchdf(); ALL["d"] = pd.to_datetime(ALL.d)
days = pd.date_range(ALL.d.min(), ALL.d.max())
NET = ALL.groupby(["style_color", "d"]).q.sum().unstack(fill_value=0).reindex(columns=days, fill_value=0)
RT = ALL[ALL.st != "_nonretail"]
open_days = RT.groupby("st").d.unique()
out = []
for (st, sc), g in RT.groupby(["st", "style_color"]):
    od = pd.DatetimeIndex(np.sort(open_days[st])); s = g.set_index("d").q.reindex(od, fill_value=0).values.astype(float)
    nz = np.flatnonzero(s > 0)
    if len(nz) < 2: continue
    net = NET.loc[sc]
    z = s == 0; z[:nz[0]] = False; flag = np.zeros(len(s), bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], z.astype(int), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        pre_i = np.arange(max(a - 28, nz[0]), a); pre_i = pre_i[~flag[pre_i]]
        if len(pre_i) < 7: continue
        lam = s[pre_i].mean(); k = b - a
        if k < 3 or lam * k <= 6.9: continue
        flag[a:b] = True
        b0 = od[a] - pd.Timedelta(days=28); w = net[(net.index >= b0) & (net.index < od[a])].sum()
        if w <= 0: continue
        sh = min(s[(od >= b0) & (od < od[a])].sum() / w, 0.8)
        j = int(np.ceil(6.9 / lam)); known = od[a + max(j, 3) - 1]
        dd = od[a:min(b, a + 28)]
        out.append(pd.DataFrame({"store": st, "style_color": sc, "date": dd, "lost": sh / (1 - sh) * net.reindex(dd).values, "known": known}))
L = pd.concat(out, ignore_index=True); L.to_parquet("data/store_lost_daily.parquet", index=False)
ret = RT.q.sum(); print("run-days", len(L), "| lost", round(L.lost.sum()), f"= {L.lost.sum() / ret:.3f} of retail units, {L.lost.sum() / ALL.q.sum():.3f} of all units")
m = L.assign(m=L.date.dt.to_period("Q")).groupby("m").lost.sum() / RT.assign(m=RT.d.dt.to_period("Q")).groupby("m").q.sum()
print("lost / retail units by quarter:", m.round(3).to_dict())
