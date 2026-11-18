import pandas as pd, numpy as np
s = pd.read_parquet("data/daily_site.parquet"); s["m"]=s.date.dt.to_period("M")
s["other"]=s.units-s.web_units-s.retail
g=s.groupby("m").agg(days=("date","size"),units=("units","sum"),web=("web_units","sum"),ret=("retail","sum"),oth=("other","sum"),drop=("drop","sum"),disc=("web_discount","mean"),rdisc=("retail_discount","mean"),canc=("cancelled_units","sum"))
for c in ["units","web","ret","oth","drop"]: g[c+"/d"]=(g[c]/g.days).round(0)
print(g[["days","units/d","web/d","ret/d","oth/d","drop/d","disc","rdisc"]].to_string())
s["wk"]=s.date.dt.to_period("W")
w=s[s.date>="2025-02-01"].groupby("wk").agg(web=("web_units","mean"),ret=("retail","mean"),oth=("other","mean"),disc=("web_discount","mean"))
print(w.round(2).to_string())
print(s[s.date>="2025-04-01"][["date","units","web_units","retail","other","web_discount"]].round(3).to_string())
ch=pd.read_parquet("data/daily_channel.parquet"); ch["m"]=ch.date.dt.to_period("M")
print(ch[ch.date>="2024-10-01"].pivot_table(index="channel",columns="m",values="units",aggfunc="sum").fillna(0).astype(int).to_string())
st=pd.read_parquet("data/daily_store.parquet"); st["m"]=st.date.dt.to_period("M")
t=st[st.date>="2024-01-01"].groupby(["store","m"]).agg(u=("units","sum"),d=("date","nunique")); t["pd"]=t.u/t.d
print(t.pd.unstack("m").round(0).to_string())
