"""Retail discount/stores, FSM opening, July-24 web jump decomposition."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "research")
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
from rules_v1 import load
it, di, st = load()
ds = pd.read_parquet("data/daily_site.parquet").set_index("date")
print("retail_discount monthly"); print(ds.retail_discount.resample("ME").mean().round(3).tail(14).to_string())
print("weekly retail disc 2025"); print(ds.loc["2025-02-01":, ["retail", "retail_discount"]].resample("W").mean().round(2))
f = st[st.store == "FSM"].set_index("date").units
print("FSM daily"); print(f.to_string())
# July-24 web: LFL per day by week vs June-24 (items live >30d before Jun, selling in June)
d = di[(di.date >= "2024-06-01") & (di.date <= "2024-08-31")]
ok = d[d.m == pd.Period("2024-06")].groupby("style_color").units.sum()
ok = ok[(ok > 0) & (it.launch.reindex(ok.index) < pd.Timestamp("2024-05-01"))].index
w = d[d.style_color.isin(ok)].groupby("date").web_units.sum()
jun = w["2024-06"].mean()
wk = w.resample("W").mean() / jun
print("LFL web per day vs June-24 by week"); print(pd.concat([wk.round(2), ds.web_discount.resample("W").mean().round(2)], axis=1).loc["2024-06":"2024-08"])
print("Jul 1-21 / Jun", round(w["2024-07-01":"2024-07-21"].mean() / jun, 3), " Jul 8-21 / Jun", round(w["2024-07-08":"2024-07-21"].mean() / jun, 3),
      " Jul 22-31 / Jun", round(w["2024-07-22":"2024-07-31"].mean() / jun, 3))
print("June disc", round(ds.loc["2024-06", "web_discount"].mean(), 3), " Jul 8-21 disc", round(ds.loc["2024-07-08":"2024-07-21", "web_discount"].mean(), 3))
# web units per day vs discount, 2024-08 .. 2025-03 by month (regime after the shift)
m = ds.loc["2024-03":].resample("ME").agg({"web_units": "mean", "web_discount": "mean"}).round(3)
print(m)
print("daily web, last 14d of March-25"); print(ds.loc["2025-03-18":, ["web_units", "web_discount", "retail", "units"]].round(2))
