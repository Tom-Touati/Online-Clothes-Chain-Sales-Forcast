import pandas as pd, sys
sys.path.insert(0,"."); import rules as R
it,di,st=R.load(); it["occ"]=it.occasion.fillna("")
p=pd.read_csv("predictions.csv").pivot(index="style_color",columns="month",values="pred")
jul=di[di.m==pd.Period("2025-07")].groupby("style_color").units.sum().reindex(p.index).fillna(0)
p.insert(0,"Jul-act",jul)
print((p.sum()/1000).round(1).to_string()); print("4m total", round(p.iloc[:,1:].sum().sum()/1e3,1))
days=[31,31,30,31,30]; print("per day", (p.sum()/days).round(0).to_dict())
g=p.groupby(it.grp.reindex(p.index)).sum(); print((g/1000).round(1).to_string())
for t in ["Halloween","Homecoming","New Year's Eve","4th of July","Graduation","Valentine"]:
    m=it.occ.reindex(p.index).str.contains(t); print(t, int(m.sum()), (p[m].sum()/1000).round(1).to_dict())
ty=p.groupby(it.product_type.reindex(p.index)).sum(); ty=ty[ty["Jul-act"]>15000]
print((ty.div(ty["Jul-act"],axis=0)).round(2).to_string())
age=(pd.Timestamp("2025-07-01")-it.launch.reindex(p.index)).dt.days/30.44
a=p.groupby(pd.cut(age,[-5,0,1,2,3,99])).sum(); print((a/1000).round(1).to_string())
