import pandas as pd, numpy as np
s = pd.read_parquet("data/daily_site.parquet"); s["m"]=s.date.dt.to_period("M")
s["other"]=s.units-s.web_units-s.retail
g=s.groupby("m").agg(days=("date","nunique"),units=("units","sum"),web=("web_units","sum"),ret=("retail","sum"),oth=("other","sum"),drop=("drop","sum"),wd=("web_discount","mean"))
for c in ["units","web","ret","oth","drop"]: g[c]=(g[c]/g.days).round(0)
print(g.tail(20).to_string())
# weekly web Jul-Dec 2024 and Jul-Aug 2025
s["wk"]=s.date.dt.to_period("W-SAT")
w=s.groupby("wk").agg(web=("web_units","mean"),ret=("retail","mean"),oth=("other","mean"),disc=("web_discount","mean"))
print(w[(w.index.start_time>="2024-07-01")&(w.index.start_time<"2025-01-05")].round(0).to_string())
print(w[w.index.start_time>="2025-06-25"].round(2).to_string())
print(s[s.date>="2025-08-15"][["date","web_units","retail","other","web_discount"]].to_string())
ch=pd.read_parquet("data/daily_channel.parquet"); ch["m"]=ch.date.dt.to_period("M")
p=ch.pivot_table(index="channel",columns="m",values="units",aggfunc="sum").fillna(0)
print(p.iloc[:, -15:].astype(int).to_string())
st=pd.read_parquet("data/daily_store.parquet"); st["m"]=st.date.dt.to_period("M")
q=st.groupby(["store","m"]).units.sum()/st.groupby(["store","m"]).date.nunique()
print(q.unstack().iloc[:, -15:].round(0).to_string())
print(st[st.date>="2025-08-20"].groupby("store").date.max())
