import pandas as pd, numpy as np
s = pd.read_parquet("data/daily_site.parquet"); s["m"]=s.date.dt.to_period("M")
s["other"]=s.units-s.web_units-s.retail
g=s.groupby("m").agg(n=("date","nunique"),units=("units","sum"),web=("web_units","sum"),ret=("retail","sum"),oth=("other","sum"),canc=("cancelled_units","sum"),wd=("web_discount","mean"),rd=("retail_discount","mean"))
for c in ["units","web","ret","oth","canc"]: g[c+"_pd"]=(g[c]/g.n).round(0)
g=g[[c for c in g.columns if c.endswith("_pd")]+["wd","rd"]]
g["web_yoy"]=(g.web_pd/g.web_pd.shift(12)).round(3); g["oth_yoy"]=(g.oth_pd/g.oth_pd.shift(12)).round(3)
print(g.tail(26).to_string())
# daily Dec-25 and Dec-24 web
for y in [2024,2025]:
    d=s[(s.date>=f"{y}-11-20")&(s.date<=f"{y+1 if y==2024 else y}-{'01-31' if y==2024 else '12-31'}")]
    w=d.set_index("date").web_units.resample("W-SAT").mean().round(0)
    print(y, w.to_string())
d=s[s.date>="2025-12-01"].set_index("date")[["web_units","retail","other","web_discount"]]
print(d.to_string())
ch=pd.read_parquet("data/daily_channel.parquet"); ch["m"]=ch.date.dt.to_period("M")
p=ch.pivot_table(index="channel",columns="m",values="units",aggfunc="sum").fillna(0)
print(p.iloc[:, -16:].round(0).to_string())
st=pd.read_parquet("data/daily_store.parquet"); st["m"]=st.date.dt.to_period("M")
q=st.pivot_table(index="store",columns="m",values="units",aggfunc="sum").fillna(0)
n=st.groupby("m").date.nunique()
print((q/n).iloc[:,-15:].round(0).to_string())
