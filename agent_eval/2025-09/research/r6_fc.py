import pandas as pd, numpy as np, sys
sys.path.insert(0,"."); import rules as R
it,di,st=R.load()
items=pd.read_csv("data/to_forecast.csv").style_color.tolist()
o=di.date.max(); B=o.to_period("M")
def show(over,label):
    P0=dict(R.P); R.P.update(over)
    f=R.seasonal_factors(it,di,st,B,[B+h for h in range(1,5)])
    F=R.forecast(o,items,it,di,st)
    R.P.clear(); R.P.update(P0)
    g=it.grp.reindex(items).values
    print(label,"pow",round(f["seas_pow"],3),"anc",f["anc"].round(3).to_dict(),"ret",f["ret"].round(3).values,"web",f["web"].round(3).values,"other",f["other"].round(3).values)
    t=F.groupby(g).sum(); t.loc["TOTAL"]=t.sum(); print((t/1000).round(1).to_string())
    for tag in ["Halloween","New Year's Eve","Homecoming"]:
        h=it.occasion.reindex(items).fillna("").str.contains(tag,regex=False).values
        print(" ",tag,h.sum(),(F[h].sum()/1000).round(1).values)
    return F
sep=di[(di.m==B)&di.style_color.isin(items)]
print("Sep actual panel",sep.units.sum(), pd.Series(sep.groupby(it.grp.reindex(sep.style_color).values).units.sum()/1000).round(1).to_dict())
show({},"v8")
show(dict(ADAPT_AMP=False,RET_POW=0.6),"v7 settings (new plan, new occ)")
