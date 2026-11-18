# forecast summary by group / tag, vs Oct actual; variant comparison at this origin
import pandas as pd, numpy as np, sys, copy
sys.path.insert(0,"."); import rules
it,di,st=rules.load()
items=pd.read_csv("data/to_forecast.csv").style_color.tolist()
o=di.date.max()
mi=pd.read_parquet("data/monthly_item.parquet")
oct_=mi[mi.month=="2025-10"].set_index("style_color").units.reindex(items).fillna(0)
sep_=mi[mi.month=="2025-09"].set_index("style_color").units.reindex(items).fillna(0)
def fc(over={}):
    P0=copy.deepcopy(rules.P); rules.P.update(over)
    try: return rules.forecast(o,items,it,di,st)
    finally: rules.P.clear(); rules.P.update(P0)
f=fc()
print("Oct actual",oct_.sum(),"Sep",sep_.sum()); print((f.sum()/1000).round(1).to_dict(), "4m", round(f.sum().sum()/1e3,1))
print("per day", {c:round(f[c].sum()/pd.Period(c).days_in_month) for c in f.columns})
g=it.grp.reindex(items)
t=f.groupby(g).sum(); t["Oct"]=oct_.groupby(g).sum(); print((t/1000).round(1).to_string()); print((t.div(t.sum())).round(3).to_string())
occ=it.occasion.reindex(items).fillna("")
for tag in ["Halloween","New Year's Eve","Homecoming","Valentine"]:
    h=occ.str.contains(tag,regex=False); print(tag,h.sum(),"Oct",oct_[h].sum(),(f[h].sum()/1000).round(1).to_dict())
for k,v in {"no spike cap":{"SPIKE_X":1e9},"amp .8 fixed":{"ADAPT_AMP":False,"SEAS_POW":0.8},"no tier":{"TIER_IDX":{}},"drift1.01":{"DRIFT_ADJ":1.01}}.items():
    f2=fc(v); print(k,(f2.sum()/1000).round(1).to_dict(), "WINTER Nov", round(f2[g=="WINTER"]["2025-11"].sum()/1e3,1))
    if k=="no spike cap":
        d=(f.sum(axis=1)-f2.sum(axis=1)).sort_values(); print(pd.concat([d.head(6),d.tail(4)]).round(0).to_string())
# factor dump
B=pd.Period("2025-10","M"); fa=rules.seasonal_factors(it,di,st,B,[B+h for h in range(1,5)])
print("amp",round(fa["seas_pow"],3),"web",fa["web"].round(3).tolist(),"other",fa["other"].round(3).tolist(),"ret",fa["ret"].round(3).tolist())
# young items
age=(B.start_time-it.launch.reindex(items)).dt.days/30.44
ab=pd.cut(age,[-99,0,1,2,99],labels=["<0","0-1","1-2","2+"])
y=f.groupby(ab).sum(); y["Oct"]=oct_.groupby(ab).sum(); print((y/1000).round(1).to_string())
