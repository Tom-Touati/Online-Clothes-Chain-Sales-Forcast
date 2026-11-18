import pandas as pd, numpy as np
import sys; sys.path.insert(0,".")
import rules as R
it,di,st=R.load()
sb=pd.read_csv("feedback/scoreboard.csv")
# overall per h
sb["ae"]=(sb.pred-sb.actual).abs()
print(sb.groupby("h").apply(lambda g: pd.Series(dict(wape=g.ae.sum()/g.actual.sum(), bias=g.pred.sum()/g.actual.sum()-1))).round(3))
# 4-month totals for complete origins
tot=sb.groupby(["origin","style_color"]).agg(pred=("pred","sum"),actual=("actual","sum"),n=("h","count")).reset_index()
tot=tot[tot.n==4]
print(tot.groupby("origin").apply(lambda g: pd.Series(dict(wape=(g.pred-g.actual).abs().sum()/g.actual.sum(), bias=g.pred.sum()/g.actual.sum()-1,
   w20=((g.pred-g.actual).abs()<=0.2*g.actual).mean(), w50=((g.pred-g.actual).abs()<=0.5*g.actual).mean()))).round(3))
d=sb[(sb.origin=="2025-11")].copy()
d=d.join(it[["product_type","grp","seg","occasion","tier","launch"]],on="style_color")
B=pd.Period("2025-11")
d["age"]=((B.start_time-d.launch).dt.days/30.44)
d["ageb"]=pd.cut(d.age,[-99,0,1,2,4,8,14,99],labels=["new_in_B","1","2","2-4","4-8","8-14","14+"])
lm=di[di.m==B].groupby("style_color")[["units","web_units","retail","other"]].sum()
d=d.join(lm,on="style_color")
d["webshare"]=pd.cut(d.web_units/d.units.clip(lower=1),[-0.01,0.5,0.8,0.95,1.01])
d["size"]=pd.qcut(d.pred.rank(method="first"),5,labels=False)
T=d.pred.sum()/d.actual.sum()
def rep(col, mn=8):
    g=d.groupby(col).agg(n=("pred","size"),pred=("pred","sum"),act=("actual","sum"))
    g["rel"]=(g.pred/g.act/T); g["ratio"]=g.pred/g.act
    g["wape"]=d.groupby(col).apply(lambda x:(x.pred-x.actual).abs().sum()/x.actual.sum())
    g=g[g.n>=mn].round(3); print(g.sort_values("act",ascending=False).head(25).to_string()); print()
print("overall ratio",round(T,3))
for c in ["grp","ageb","webshare","size","product_type","tier"]: rep(c)
tags=["Halloween","New Year's Eve","Valentine's","Homecoming","Graduation","4th of July","St. Patrick","Christmas"]
for t in tags:
    m=d.occasion.fillna("").str.contains(t,regex=False)
    if m.sum(): print(t, m.sum(), round(d[m].pred.sum()/d[m].actual.sum()/T,3))
# list all occasion tags frequency among to_forecast
fc=pd.read_csv("data/to_forecast.csv").style_color
oc=it.occasion.reindex(fc).fillna("").str.split("; ").explode().value_counts()
print(oc.head(30).to_string())
print(it.tier.reindex(fc).value_counts().to_string())
