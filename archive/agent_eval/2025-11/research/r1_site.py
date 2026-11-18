"""Site / channel / store levels per month and weekly, Nov-24..Mar-25 vs 2025."""
import pandas as pd, numpy as np
D = "data/"
s = pd.read_parquet(D + "daily_site.parquet")
s["other"] = s.units - s.web_units - s.retail
s["m"] = s.date.dt.to_period("M")
g = s.groupby("m")
pm = pd.DataFrame({"days": g.date.nunique(), "units": g.units.sum(), "web": g.web_units.sum(),
                   "retail": g.retail.sum(), "other": g.other.sum(), "canc": g.cancelled_units.sum(),
                   "wdisc": g.web_discount.mean()})
for c in ["units", "web", "retail", "other", "canc"]:
    pm[c + "_pd"] = (pm[c] / pm.days).round(0)
pm["web_yoy"] = (pm.web_pd / pm.web_pd.shift(12)).round(2)
pm["oth_yoy"] = (pm.other_pd / pm.other_pd.shift(12)).round(2)
pm["ret_yoy"] = (pm.retail_pd / pm.retail_pd.shift(12)).round(2)
print(pm[["days", "units_pd", "web_pd", "retail_pd", "other_pd", "canc_pd", "wdisc", "web_yoy", "oth_yoy", "ret_yoy"]].tail(26).to_string())
# daily Nov 2025 and Nov-24..Mar-25 weekly
s = s.set_index("date")
print("\nDaily Nov-25 web / other / retail / canc")
print(s.loc["2025-11-01":, ["web_units", "other", "retail", "cancelled_units", "web_discount"]].round(2).to_string())
w = s.loc["2024-10-27":"2025-03-31", ["web_units", "other", "retail", "web_discount"]].resample("W-SAT").mean().round(1)
print("\nWeekly LY (per day)"); print(w.to_string())
ch = pd.read_parquet(D + "daily_channel.parquet"); ch["m"] = ch.date.dt.to_period("M")
t = ch.pivot_table(index="channel", columns="m", values="units", aggfunc="sum").fillna(0)
print("\nChannel monthly"); print(t.iloc[:, -17:].round(0).astype(int).to_string())
st = pd.read_parquet(D + "daily_store.parquet"); st["m"] = st.date.dt.to_period("M")
t = st.groupby(["store", "m"]).agg(u=("units", "sum"), d=("date", "nunique"))
t = (t.u / t.d).unstack().round(0)
print("\nStore per day"); print(t.iloc[:, -17:].to_string())
