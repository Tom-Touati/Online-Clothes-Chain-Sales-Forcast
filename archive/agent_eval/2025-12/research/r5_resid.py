import pandas as pd, numpy as np, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"research"); sys.argv=["x"]
exec(open("research/r4_bt.py").read().split("base=run()")[0])
F=run(occ={},tier={},key="seg")
occ=it.occasion.reindex(items).fillna(""); tr=it.tier.reindex(items).fillna("")
def res(mask,name):
    if mask.sum()<4: return
    r=[A.loc[mask,c].sum()/F.loc[mask,c].sum() for c in A.columns]
    print(f"{name:26s} n={mask.sum():4d} "+" ".join(f"{x:.2f}" for x in r)+f" | 4m {A[mask].values.sum()/F[mask].values.sum():.2f}")
print("SEG-oracle residual actual/pred Jan Feb Mar Apr")
for t in ["Halloween","New Year's Eve","Valentine's","Homecoming","Graduation","4th of July","St. Patrick"]:
    res(occ.str.contains(t,regex=False),t)
for t in ["Party","Evening","Basics","Resort-Festival","Casual","Social Daywear","Loungewear"]:
    res(tr.str.contains(t,regex=False)&~occ.str.contains("Halloween|New Year|Valentine"),"tier "+t)
# scoreboard: size quintile and age by h (act/pred relative to origin-month total)
sb=pd.read_csv("feedback/scoreboard.csv")
sb["tot"]=sb.groupby(["origin","month"]).pred.transform("sum")/sb.groupby(["origin","month"]).actual.transform("sum")
sb["q"]=sb.groupby(["origin","month"]).pred.transform(lambda x: pd.qcut(x.rank(method="first"),5,labels=False))
g=sb.groupby(["h","q"]).apply(lambda x: (x.actual.sum()/(x.pred/x.tot).sum())).unstack().round(2)
print("scoreboard act/pred rel by h x size quintile\n",g)
g=sb.groupby(["origin","q"]).apply(lambda x: (x.actual.sum()/(x.pred/x.tot).sum())).unstack().round(2)
print(g)
sb=sb.join(it.launch,on="style_color")
sb["age"]=(pd.PeriodIndex(sb.origin,freq="M").start_time-sb.launch).dt.days/30.44
sb["ab"]=pd.cut(sb.age,[-99,0,1,2,4,99],labels=["inB","B-1","B-2","2-4","4+"])
g=sb.groupby(["h","ab"]).apply(lambda x: (x.actual.sum()/(x.pred/x.tot).sum())).unstack().round(2)
print("scoreboard act/pred rel by h x age (age at start of base month)\n",g)
print(sb.groupby(["h","ab"]).size().unstack())
