import pandas as pd, numpy as np
s = pd.read_parquet("data/daily_site.parquet")
s["m"] = s.date.dt.to_period("M")
g = s.groupby("m").agg(days=("date","nunique"), units=("units","sum"), web=("web_units","sum"), online=("online","sum"),
                       retail=("retail","sum"), drop=("drop","sum"), wd=("web_discount","mean"), rd=("retail_discount","mean"))
g["other"] = g.units - g.web - g.retail
for c in ["units","web","retail","other","drop"]: g[c+"_pd"] = (g[c]/g.days).round(0)
print(g[["days","units_pd","web_pd","other_pd","retail_pd","drop_pd","wd","rd"]].to_string())
ch = pd.read_parquet("data/daily_channel.parquet"); ch["m"]=ch.date.dt.to_period("M")
p = ch[ch.m>=pd.Period("2025-01","M")].pivot_table(index="m", columns="channel", values="units", aggfunc="sum").fillna(0)
p = p.div(s.groupby("m").date.nunique().reindex(p.index), axis=0).round(0)
print(p.T.to_string())
st = pd.read_parquet("data/daily_store.parquet"); st["m"]=st.date.dt.to_period("M")
q = st[st.m>=pd.Period("2025-01","M")].pivot_table(index="m", columns="store", values="units", aggfunc="sum").fillna(0)
print(q.T.to_string())
