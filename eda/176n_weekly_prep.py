"""D176 (overnight M5 goal, directions 5, 7, 8): weekly item series and weekly known-future inputs, for the neural multi-horizon
models (server), the weekly-origin and the augmentation experiments.
  data/weekly_item.parquet   unique_id (style_color), ds (week start Monday), y (units, all channels, non-cancelled)
  data/weekly_futr.parquet   ds plus known-future inputs per week: days of each of 16 event windows, national normal temperature
                             (site-mix weighted 2014-2023 normals of that calendar month), site discount and deep-discount days
                             (realised campaign calendar, standing in for the plan), week-of-year sin / cos
  data/weekly_futr_item.parquet  unique_id, ds, t_norm_item (normal temperature where the item sells, 12-month mix to 2026-06)
  data/static_item.parquet   unique_id plus static inputs: product type one-hot (top 15 + other), the 14 LLM scores (0-1)."""
import sys, glob, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
d = pd.read_parquet("data/daily_item_chan.parquet", columns=["style_color", "ny_date", "class", "cancelled", "qty", "disc_w"]); d = d[~d.cancelled]; d["ny_date"] = pd.to_datetime(d.ny_date)
items = np.sort(pd.read_parquet("data/items.parquet").style_color.values)
d = d[d.style_color.isin(items)]; d["ds"] = d.ny_date - pd.to_timedelta(d.ny_date.dt.weekday, unit="D")
W = d.groupby(["style_color", "ds"]).qty.sum().reset_index().rename(columns={"style_color": "unique_id", "qty": "y"})
weeks = pd.date_range(W.ds.min(), "2026-06-29", freq="W-MON"); first = W.groupby("unique_id").ds.min()
full = pd.MultiIndex.from_product([items, weeks], names=["unique_id", "ds"]).to_frame(index=False).merge(W, how="left").fillna({"y": 0})
full = full[full.ds >= full.unique_id.map(first)]; full.to_parquet("data/weekly_item.parquet", index=False)
E = pd.read_parquet("data/events_month.parquet")
from datetime import date, timedelta
ev = {}
src = open("eda/161_events.py", encoding="utf-8").read(); exec(src[src.index("EASTER ="):src.index("months = pd.period_range")])
fut = pd.DataFrame({"ds": pd.date_range(weeks.min(), "2026-12-28", freq="W-MON")})
for k, ds in days.items():
    s = pd.Series(1, index=pd.to_datetime(sorted(set(ds)))); wk = s.index - pd.to_timedelta(s.index.weekday, unit="D")
    fut[f"ev_{k}"] = fut.ds.map(pd.Series(1, index=wk).groupby(level=0).sum()).fillna(0)
Wt = pd.read_parquet("data/weather_state_month.parquet"); S = pd.read_parquet("data/state_item_month.parquet")
sw = S.groupby("state").units.sum(); sw = sw / sw.sum()
norm = Wt[Wt.state.isin(sw.index)].groupby(["cm", "state"]).tmean_norm.first().unstack().reindex(columns=sw.index).fillna(Wt.tmean_norm.mean())
fut["t_norm"] = fut.ds.dt.month.map(pd.Series(norm.values @ sw.values, index=norm.index))
dd = d[d["class"] == "ONLINE"].groupby("ny_date").agg(q=("qty", "sum"), dw=("disc_w", "sum")); dd["disc"] = dd.dw / dd.q; dd["deep"] = (dd.disc >= 0.585).astype(int)
dd["ds"] = dd.index - pd.to_timedelta(dd.index.weekday, unit="D"); wkd = dd.groupby("ds").agg(disc=("dw", "sum"), q=("q", "sum"), deep=("deep", "sum")); wkd["disc"] = wkd.disc / wkd.q
fut["site_disc"] = fut.ds.map(wkd.disc); fut["site_deep"] = fut.ds.map(wkd.deep)
fut["site_disc"] = fut.site_disc.fillna(fut.site_disc.mean()); fut["site_deep"] = fut.site_deep.fillna(0)
woy = fut.ds.dt.isocalendar().week.astype(int); fut["woy_sin"] = np.sin(2 * np.pi * woy / 52); fut["woy_cos"] = np.cos(2 * np.pi * woy / 52)
fut.to_parquet("data/weekly_futr.parquet", index=False)
u12 = S[S.m >= "2025-07"].groupby(["it", "state"]).units.sum().unstack(fill_value=0).reindex(columns=sw.index, fill_value=0)
mix = (u12.values + 20 * sw.values) / (u12.values.sum(1, keepdims=True) + 20); tm = mix @ norm.values.T              # items x calendar months
TI = pd.DataFrame(tm, index=items[u12.index.values], columns=range(1, 13)).reindex(items).fillna(pd.Series(norm.values @ sw.values, index=range(1, 13)))
fi = fut[["ds"]].assign(cm=fut.ds.dt.month).merge(TI.stack().rename("t_norm_item").reset_index().rename(columns={"level_0": "unique_id", "level_1": "cm"}), on="cm")
fi[["unique_id", "ds", "t_norm_item"]].to_parquet("data/weekly_futr_item.parquet", index=False)
it = pd.read_parquet("data/items.parquet").set_index("style_color").reindex(items); pt = it.product_type.fillna("na"); top = pt.value_counts().head(15).index
st = pd.get_dummies(pt.where(pt.isin(top), "other"), prefix="pt").astype(float)
L = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("data/llm/scores_part*.csv"))]).drop_duplicates("style_color").set_index("style_color").reindex(items) / 10
st = pd.concat([st.reset_index(drop=True), L.reset_index(drop=True).add_prefix("llm_").fillna(0.5)], axis=1); st.insert(0, "unique_id", items)
st.to_parquet("data/static_item.parquet", index=False)
print("weekly rows", len(full), "| series", full.unique_id.nunique(), "| weeks", len(weeks), "| futr cols", fut.shape[1] - 1, "| static cols", st.shape[1] - 1)
