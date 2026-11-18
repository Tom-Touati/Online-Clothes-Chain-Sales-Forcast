"""Is the low July/June step explained by channel mix (retail share up, new stores opening in June 2026), and what did the
June 2025 backtest say about July for the same item groups? Targets only; history = like-for-like items selling >= 50 in June."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[~d.cancelled]; d["ny_date"] = pd.to_datetime(d.ny_date); d["m"] = d.ny_date.dt.strftime("%Y-%m")
O = pd.read_csv("data/final_forecast_q3.csv", index_col=0); T = list(O.index)
C = d.groupby(["style_color", "m", "class"]).qty.sum().unstack("class", fill_value=0)
for y in (2024, 2025, 2026):
    jun = C.xs(f"{y}-06", level="m"); jun = jun[jun.sum(axis=1) >= 50]
    if y < 2026:
        jul = C.xs(f"{y}-07", level="m").reindex(jun.index, fill_value=0)
        print(f"{y} like-for-like (n={len(jun)}): June class share", (jun.sum() / jun.sum().sum()).round(3).to_dict(), "| July/June by class", (jul.sum() / jun.sum()).round(2).to_dict(), "| total", round(jul.sum().sum() / jun.sum().sum(), 2))
    else:
        tj = jun.reindex(T).fillna(0); print(f"2026 targets: June class share", (tj.sum() / tj.sum().sum()).round(3).to_dict())
tj = C.xs("2026-06", level="m").reindex(T).fillna(0)
for y in (2024, 2025):
    jun = C.xs(f"{y}-06", level="m"); jun = jun[jun.sum(axis=1) >= 50]; jul = C.xs(f"{y}-07", level="m").reindex(jun.index, fill_value=0)
    r = jul.sum() / jun.sum(); print(f"expected July/June for the 2026 targets' June channel mix with {y} class ratios: {(tj.sum() * r).sum() / tj.sum().sum():.2f}")
# retail: daily units in June 2026 by store, and new-store share of June retail
st = pd.read_parquet("data/daily_otype.parquet"); st["ny_date"] = pd.to_datetime(st.ny_date); st = st[~st.cancelled & (st["class"] == "RETAIL")]
j = st[(st.ny_date >= "2026-05-01")].groupby([pd.Grouper(key="ny_date", freq="W-SUN"), "otype"]).qty.sum().unstack(fill_value=0)
new = ["NYFIFTH", "FLLINCOLN", "STONEBRIAR", "CHOAKBROOK", "HOUGALLERIA"]
print("\nretail weekly units, May-June 2026: all stores", j.sum(axis=1).astype(int).tolist(), "| stores opened since mid-April", j[[c for c in new if c in j]].sum(axis=1).astype(int).tolist())
r25 = st[(st.ny_date >= "2025-06-01") & (st.ny_date < "2025-08-01")].groupby(st.ny_date.dt.month).qty.sum(); print("retail all items Jun/Jul 2025:", r25.astype(int).to_dict(), "-> July/June", round(r25[7] / r25[6], 2))
# June 2026 vs a 'run-rate' base: what share of June target units came in the last week of June (Summer Sale started 28 June)?
dj = d[d.style_color.isin(T) & (d.ny_date >= "2026-06-01")].groupby("ny_date").qty.sum()
dj5 = d[d.style_color.isin(T) & (d.ny_date >= "2025-06-01") & (d.ny_date < "2025-07-01")].groupby("ny_date").qty.sum()
print("targets June 2026 daily units: first half avg", int(dj[:"2026-06-15"].mean()), "| second half avg", int(dj["2026-06-16":].mean()), "| 28-30 June avg", int(dj["2026-06-28":].mean()))
