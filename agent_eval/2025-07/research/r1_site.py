import pandas as pd, numpy as np
s = pd.read_parquet("data/daily_site.parquet"); s["m"]=s.date.dt.to_period("M")
s["other"]=s.units-s.web_units-s.retail
g=s.groupby("m").agg(days=("date","nunique"),units=("units","sum"),web=("web_units","sum"),ret=("retail","sum"),oth=("other","sum"),drop=("drop","sum"),disc=("web_discount","mean"),canc=("cancelled_units","sum"))
for c in ["units","web","ret","oth","drop","canc"]: g[c+"_pd"]=(g[c]/g.days).round(0)
print(g[[c for c in g if c.endswith("_pd")]+["disc"]].to_string())
s["wk"]=s.date.dt.to_period("W-SAT")
w=s[s.date>="2025-06-01"].groupby("wk").agg(units=("units","mean"),web=("web_units","mean"),ret=("retail","mean"),oth=("other","mean"),disc=("web_discount","mean"))
print(w.round(2).to_string())
print(s[s.date>="2025-07-01"][["date","units","web_units","retail","other","web_discount"]].round(3).to_string())
# 2024 Jul-Nov weekly
w=s[(s.date>="2024-06-01")&(s.date<"2024-12-15")].groupby("wk").agg(units=("units","mean"),web=("web_units","mean"),ret=("retail","mean"),oth=("other","mean"),disc=("web_discount","mean"))
print(w.round(2).to_string())
c=pd.read_parquet("data/daily_channel.parquet"); c["m"]=c.date.dt.to_period("M")
print(c[c.date>="2025-03-01"].pivot_table(index="channel",columns="m",values="units",aggfunc="sum").fillna(0).astype(int).to_string())
st=pd.read_parquet("data/daily_store.parquet"); st["m"]=st.date.dt.to_period("M")
x=st.groupby(["store","m"]).agg(u=("units","sum"),d=("date","nunique"))
print((x.u/x.d).round(0).unstack().iloc[:, -14:].to_string())
print(st[st.date>="2025-07-01"].groupby("store").date.agg(["min","max","nunique"]))
