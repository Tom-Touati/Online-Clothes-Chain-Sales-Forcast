"""Backtest: young items (3-12 months) by June position relative to their peak so far, at the summer origins. P50 bias and median ratio.
If the model under-forecast items far below peak, a correction is justified; if not, the gap to peers is composition."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
Q = pd.read_parquet("data/q_backtest.parquet"); M = pd.read_parquet("data/monthly_item.parquet"); cols = list(M.columns)
it = pd.read_parquet("data/items.parquet").set_index("style_color")
start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
MAC = {"Hoodie": "warm", "Sweater": "warm", "Sweatpants": "warm", "Jacket": "warm", "Cardigan": "warm", "Pants": "warm"}
rows = []
for o in ["2025-05", "2025-06", "2025-07", "2025-08"]:
    k = cols.index(o); g = Q[Q.origin == o].copy()
    age = (pd.Period(o).end_time - start.reindex(g.style_color)).dt.days.values / 30.44
    g = g[(age >= 3) & (age < 12)]
    pk = M.loc[g.style_color, cols[:k + 1]].max(axis=1).values; last = M.loc[g.style_color, o].values
    g["pos"] = pd.cut(last / pk, [0, .35, .7, 10], labels=["below 35% of peak", "35-70%", ">=70%"]); g["warm"] = it.product_type.reindex(g.style_color).map(MAC).fillna("other").values
    rows.append(g)
G = pd.concat(rows)
f = lambda x: pd.Series(dict(n=len(x), bias=x.q50.sum() / x.ytot.sum() - 1, median_ratio=(x.q50 / x.ytot.clip(lower=1)).median(), w20=((x.q50 - x.ytot).abs() / x.ytot.clip(lower=1) <= .2).mean()))
print("young items (3-12 months) at summer origins May-Aug 2025, P50:"); print(G.groupby("pos", observed=True).apply(f).round(3).to_string())
print("\nby origin:"); print(G.groupby(["origin", "pos"], observed=True).apply(f).round(3).to_string())
print("\nby warm (hoodies, sweaters, sweatpants, jackets, cardigans, pants) vs other:"); print(G.groupby(["pos", "warm"], observed=True).apply(f).round(3).to_string())
O = pd.read_csv("data/final_forecast_q3.csv", index_col=0)
ids = O[(O.age_m >= 3) & (O.age_m < 12)].index; pk = M.loc[ids, :"2026-06"].max(axis=1); pos = M.loc[ids, "2026-06"] / pk
x = pd.DataFrame({"type": O.loc[ids, "ptype"], "pos": pos.round(2), "jun": M.loc[ids, "2026-06"], "fc": O.loc[ids, "final_total"], "fc/jun": (O.loc[ids, "final_total"] / M.loc[ids, "2026-06"]).round(2), "seg": O.loc[ids, "seg"]})
print("\n2026 targets 3-12 months below 35% of peak:"); print(x[x.pos < .35].sort_values("type").to_string())
