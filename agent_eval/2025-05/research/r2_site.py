import pandas as pd, numpy as np
s=pd.read_parquet("data/daily_site.parquet"); s["m"]=s.date.dt.to_period("M")
s["other"]=s.units-s.web_units-s.retail
g=s.groupby("m").agg(days=("date","nunique"),units=("units","sum"),web=("web_units","sum"),ret=("retail","sum"),oth=("other","sum"),wd=("web_discount","mean"),rd=("retail_discount","mean"))
for c in ["units","web","ret","oth"]: g[c+"/d"]=(g[c]/g.days).round(0)
print(g[["units/d","web/d","ret/d","oth/d","wd","rd"]].round(3).to_string())
# weekly May 2025
w=s[s.date>="2025-04-01"].set_index("date")[["web_units","retail","other","web_discount"]].resample("W").mean().round(2)
print(w)
st=pd.read_parquet("data/daily_store.parquet"); st["m"]=st.date.dt.to_period("M")
p=st.groupby(["store","m"]).units.sum().unstack().fillna(0)
d=st.groupby("m").date.nunique()
print((p/d).loc[:,"2024-03":].round(0).astype(int).to_string())
ch=pd.read_parquet("data/daily_channel.parquet"); ch["m"]=ch.date.dt.to_period("M")
print(ch[ch.m>="2024-04"].groupby(["channel","m"]).units.sum().unstack().fillna(0).astype(int).to_string())
