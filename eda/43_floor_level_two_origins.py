"""Floor level chosen across two origins (June and July 2025), each using only the same origin month in 2024.
Floor = q-th percentile of the 2024 item-level next-4-months / origin-month ratio for the segment; applied to young items (< 15 months)."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
M = pd.read_parquet("data/monthly_item.parquet"); cols = list(M.columns); it = pd.read_parquet("data/items.parquet").set_index("style_color")
start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
occ = it.occasion.fillna(""); Q = pd.read_parquet("data/q_backtest.parquet")
def met(f, a):
    f, a = np.asarray(f, float), np.asarray(a, float); ape = np.abs(f - a) / np.clip(a, 1, None)
    return dict(bias=f.sum() / a.sum() - 1, wape=np.abs(f - a).sum() / a.sum(), w20=(ape <= .2).mean(), w50=(ape <= .5).mean())
QS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]; rows = []
for o in ["2025-06", "2025-07"]:
    k = cols.index(o); oly = cols[k - 12]
    live = start[start < pd.Period(oly).start_time - pd.Timedelta(days=15)].index.intersection(M.index); live = live[M.loc[live, oly] >= 50]
    g = Q[Q.origin == o].set_index("style_color"); age = (pd.Period(o).end_time - start.reindex(g.index)).dt.days / 30.44
    for s in ["Homecoming", "Halloween"]:
        r = (M.loc[live, cols[k - 11:k - 7]].sum(axis=1) / M.loc[live, oly])[occ.reindex(live).str.contains(s).values]
        ids = g.index[(age < 15).values & occ.reindex(g.index).str.contains(s).values & (M.loc[g.index, o] >= 50).values]
        a = g.loc[ids, "ytot"]; rows.append(dict(origin=o, seg=s, level="model", n=len(ids), **met(g.loc[ids, "q50"], a)))
        for q in QS:
            rows.append(dict(origin=o, seg=s, level=f"p{int(q*100)}", n=len(ids), mult=r.quantile(q), **met(np.maximum(g.loc[ids, "q50"], M.loc[ids, o] * r.quantile(q)), a)))
R = pd.DataFrame(rows)
for s in ["Homecoming", "Halloween"]:
    x = R[R.seg == s].pivot_table(index="level", columns="origin", values=["bias", "wape", "w20"]).round(3)
    x["mean_wape"] = x["wape"].mean(axis=1).round(3); x["mean_abs_bias"] = x["bias"].abs().mean(axis=1).round(3); x["mean_w20"] = x["w20"].mean(axis=1).round(3)
    print(f"== {s}"); print(x.sort_values("mean_wape").to_string())
# percentile of the floors currently in use within the 2024 and 2025 June distributions
for s, cur in [("Homecoming", 6.31), ("Halloween", 6.86)]:
    for y in (2024, 2025):
        live = start[start < pd.Timestamp(f"{y}-05-15")].index.intersection(M.index); live = live[M.loc[live, f"{y}-06"] >= 50]
        r = (M.loc[live, [f"{y}-{m:02d}" for m in (7, 8, 9, 10)]].sum(axis=1) / M.loc[live, f"{y}-06"])[occ.reindex(live).str.contains(s).values]
        print(f"{s} current floor {cur}: percentile in {y} June distribution = {(r <= cur).mean():.0%} (n={len(r)}); p30 {r.quantile(.3):.2f}, p35 {r.quantile(.35):.2f}, p40 {r.quantile(.4):.2f}")
