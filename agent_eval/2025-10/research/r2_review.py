# scoreboard review: Oct-25 by segment (origin Sep h=1, plus Jul/Aug origins), cancellations
import pandas as pd, numpy as np
D="data/"
s=pd.read_parquet(D+"daily_site.parquet")
pass
sb=pd.read_csv("feedback/scoreboard.csv")
it=pd.read_csv(D+"items.csv",parse_dates=["published_date","first_sale"]).set_index("style_color")
at=pd.read_csv(D+"item_attributes.csv").set_index("style_color")
import importlib.util,sys
sys.path.insert(0,".")
import rules
itr,di,st=rules.load()
sb["grp"]=sb.style_color.map(itr.grp); sb["type"]=sb.style_color.map(it.product_type)
sb["occ"]=sb.style_color.map(it.occasion).fillna("")
sb["launch"]=sb.style_color.map(itr.launch)
sb["age"]=((pd.to_datetime(sb.origin+"-01")-sb.launch).dt.days/30.44)
sb["ageb"]=pd.cut(sb.age,[-99,0,1,2,4,8,99],labels=["<0","0-1","1-2","2-4","4-8","8+"])
mi=pd.read_parquet(D+"monthly_item.parquet")
def rel(df,key,minn=8):
    g=df.groupby(key).agg(n=("pred","size"),pred=("pred","sum"),act=("actual","sum"))
    tot=df.actual.sum()/df.pred.sum()
    g["a/p"]=(g.act/g.pred).round(3); g["rel"]=(g["a/p"]/tot).round(3)
    return g[g.n>=minn].sort_values("pred",ascending=False)
for o in ["2025-09","2025-08","2025-07"]:
    x=sb[(sb.origin==o)&(sb.month=="2025-10")].copy()
    print("\n=== origin",o,"Oct  a/p total",round(x.actual.sum()/x.pred.sum(),3), "WAPE", round((x.pred-x.actual).abs().sum()/x.actual.sum(),3))
    print(rel(x,"grp").to_string())
    if o=="2025-09":
        print(rel(x,"ageb").to_string())
        print(rel(x,"type",15).head(25).to_string())
        # channel share in base month
        m=mi[mi.month==o].set_index("style_color")
        x["rsh"]=x.style_color.map(m["retail"]/m["units"]).fillna(0)
        x["osh"]=x.style_color.map(m["drop"]/m["units"]).fillna(0)
        x["rb"]=pd.cut(x.rsh,[-1,0.05,0.25,0.5,1.1]); x["ob"]=pd.cut(x.osh,[-1,0.05,0.25,0.5,1.1])
        print(rel(x,"rb").to_string()); print(rel(x,"ob").to_string())
        x["q"]=pd.qcut(x.pred.rank(method="first"),5,labels=False)
        print(rel(x,"q").to_string())
        for tag in ["Halloween","Homecoming","New Year's Eve","Graduation","Valentine","4th of July","Christmas","Holiday","Party","Date Night","Vacation","Festival","Casual"]:
            y=x[x.occ.str.contains(tag,regex=False)]
            if len(y): print(f"{tag:16s} n={len(y):4d} pred={y.pred.sum():9.0f} act={y.actual.sum():9.0f} rel={y.actual.sum()/y.pred.sum()/(x.actual.sum()/x.pred.sum()):.3f}")
# all tags present
tags=it.occasion.dropna().str.split("; ").explode().value_counts()
print(tags.head(40).to_string())
