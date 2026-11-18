import sys; sys.path.insert(0,".")
import pandas as pd, numpy as np, rules as R
it,di,st=R.load()
items=pd.read_csv("data/to_forecast.csv").style_color.tolist()
o=di.date.max(); B=o.to_period("M"); T=[B+h for h in range(1,5)]
f=R.seasonal_factors(it,di,st,B,T)
for k in ["web","defl_w","ret"]: print(k, {str(a):round(b,3) for a,b in f[k].items()})
F=pd.read_csv("predictions.csv").pivot(index="style_color",columns="month",values="pred")
mi=pd.read_parquet("data/monthly_item.parquet"); may=mi[mi.month=="2025-05"].set_index("style_color").units.reindex(F.index).fillna(0)
t=it.reindex(F.index)
g=F.groupby(t.grp).sum(); g.insert(0,"May act",may.groupby(t.grp).sum()); print(g.round(0))
age=t.launch.dt.to_period("M").astype(str).where(t.launch>="2025-01-01","old")
g=F.groupby(age.values).sum(); g.insert(0,"May act",may.groupby(age.values).sum()); print(g.round(0))
for tag in ["4th of July","Graduation","Homecoming","Halloween"]:
    m=t.occasion.fillna("").str.contains(tag,regex=False)
    print(tag, round(may[m].sum()), F[m].sum().round(0).to_dict())
print("zero items", (F.sum(axis=1)==0).sum(), "of", len(F))
