# site / channel / store levels: monthly per day 2024-2025, Oct-25 weekly, YoY
import pandas as pd, numpy as np
D="data/"
s=pd.read_parquet(D+"daily_site.parquet"); s["m"]=s.date.dt.to_period("M")
s["other"]=s.units-s.web_units-s.retail
g=s.groupby("m")[["units","web_units","retail","other","cancelled_units"]].mean().round(0)
g["web_disc"]=s.groupby("m").web_discount.mean().round(3)
g["yoy_web"]=(g.web_units/g.web_units.shift(12)).round(3)
g["yoy_oth"]=(g.other/g.other.shift(12)).round(3)
g["yoy_ret"]=(g.retail/g.retail.shift(12)).round(3)
print(g.to_string())
s["wk"]=s.date.dt.to_period("W-SAT")
w=s[s.date>="2024-09-01"].groupby("wk")[["web_units","retail","other"]].mean().round(0)
w["disc"]=s[s.date>="2024-09-01"].groupby("wk").web_discount.mean().round(3)
print(w[(w.index.start_time<"2025-03-01")|(w.index.start_time>="2025-09-01")].to_string())
ch=pd.read_parquet(D+"daily_channel.parquet"); ch["m"]=ch.date.dt.to_period("M")
c=ch[ch.m>="2024-08"].pivot_table(index="channel",columns="m",values="units",aggfunc="sum").fillna(0)
print(c.loc[c.sum(axis=1).sort_values(ascending=False).index[:12]].astype(int).to_string())
st=pd.read_parquet(D+"daily_store.parquet"); st["m"]=st.date.dt.to_period("M")
x=st.groupby(["store","m"]).units.sum()/st.groupby(["store","m"]).date.nunique()
print(x.unstack().loc[:, "2024-08":].round(0).to_string())
print(st.groupby("store").date.agg(["min","max"]))
