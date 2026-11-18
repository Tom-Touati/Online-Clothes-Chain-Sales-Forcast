import pandas as pd, numpy as np, sys
sys.path.insert(0,"prev"); import rules as R
it,di,st=R.load()
sb=pd.read_csv("feedback/scoreboard.csv")
sb=sb.join(it[["grp","seg","product_type","occasion","launch"]],on="style_color")
mi=pd.read_parquet("data/monthly_item.parquet")
def rel(df,key):
    tot=df.actual.sum()/df.pred.sum()
    g=df.groupby(key).agg(n=("pred","size"),pred=("pred","sum"),act=("actual","sum"))
    g["rel"]=(g.act/g.pred/tot).round(2); g["ratio"]=(g.act/g.pred).round(2)
    g[["pred","act"]]=g[["pred","act"]].round(0); return g
for o in ["2025-08","2025-07","2025-06"]:
    d=sb[(sb.origin==o)&(sb.month=="2025-09")].copy()
    print("=== origin",o,"Sep  ratio",round(d.actual.sum()/d.pred.sum(),3))
    print(rel(d,"grp").to_string())
    B=pd.Period(o)
    d["age"]=pd.cut((B.start_time-d.launch).dt.days/30.44,[-99,0,1,2,4,12,999],labels=["B","B-1","B-2","2-4","4-12","12+"])
    print(rel(d,"age").to_string())
    if o=="2025-08":
        t=rel(d,"product_type"); print(t[t.n>=8].sort_values("rel").to_string())
        for tag in ["Halloween","Homecoming","New Year's Eve","4th of July","Graduation","Valentine's"]:
            x=d[d.occasion.fillna("").str.contains(tag,regex=False)]
            print(tag,len(x),round(x.actual.sum()/x.pred.sum()/(d.actual.sum()/d.pred.sum()),2))
        # channel mix in Aug
        aug=mi[mi.month=="2025-08"].set_index("style_color")
        d["rsh"]=(aug.retail/aug.units).reindex(d.style_color).values
        d["rb"]=pd.cut(d.rsh,[-.01,.05,.15,.3,.5,1.01])
        print(rel(d,"rb").to_string())
        d["q"]=pd.qcut(d.pred.rank(method="first"),5,labels=False)
        print(rel(d,"q").to_string())
# channel-level panel moves Aug->Sep
for col in ["units","online","retail","drop"]:
    a=mi[mi.month=="2025-08"].set_index("style_color")[col]; b=mi[mi.month=="2025-09"].set_index("style_color")[col]
    ok=a.index[(it.launch.reindex(a.index)<"2025-07-01")&(a>0)]
    print(col,"LFL Sep/Aug per day",round(b.reindex(ok).fillna(0).sum()/a[ok].sum()*31/30,3))
