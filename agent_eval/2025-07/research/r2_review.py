import pandas as pd, numpy as np, sys
sys.path.insert(0,"."); import rules as R
it, di, st = R.load()
sb = pd.read_csv("feedback/scoreboard.csv")
mi = pd.read_parquet("data/monthly_item.parquet")
it["occ"]=it.occasion.fillna("")
def rel(df, key):
    tot = df.actual.sum()/df.pred.sum()
    g = df.groupby(key).agg(n=("pred","size"),pred=("pred","sum"),act=("actual","sum"))
    g["a/p"]=(g.act/g.pred).round(2); g["rel"]=(g["a/p"]/tot).round(2)
    return g
for o in ["2025-06","2025-05","2025-04"]:
    d = sb[(sb.origin==o)&(sb.month=="2025-07")].copy()
    d = d.join(it[["grp","seg","product_type","launch","occ"]], on="style_color")
    B = pd.Period(o); d["age"]=((B.start_time-d.launch).dt.days/30.44).clip(-1,30)
    d["ageb"]=pd.cut(d.age,[-2,0,1,2,3,6,12,99])
    d["sizeq"]=pd.qcut(d.pred.rank(method="first"),5,labels=False)
    # channel mix in base month
    m = mi[mi.month==o].set_index("style_color")
    d["ret_sh"]=(m.retail/m.units).reindex(d.style_color).values
    d["retb"]=pd.cut(d.ret_sh.fillna(0),[-.01,.1,.25,.4,1.01])
    d["oth_sh"]=((m.units-m.online-m.retail+ (m.online- 0))/m.units).reindex(d.style_color).values
    tot=d.actual.sum()/d.pred.sum()
    print(f"\n==== origin {o} July: pred {d.pred.sum():.0f} act {d.actual.sum():.0f} a/p {tot:.3f}  scaled WAPE {(abs(d.pred*tot-d.actual).sum()/d.actual.sum()):.3f}")
    for k in ["grp","ageb","sizeq","retb"]: print(rel(d,k).to_string())
    if o=="2025-06":
        t=rel(d,"product_type"); print(t[t.n>=8].sort_values("rel").to_string())
        for tag in ["4th of July","Graduation","Homecoming","Halloween","Valentine","New Year","Back to School","Festival","Vacation","Beach"]:
            x=d[d.occ.str.contains(tag)]
            if len(x): print(tag,len(x),round(x.actual.sum()/x.pred.sum()/tot,2))
# all-origin by h
sb["ratio"]=1
print(sb.groupby(["origin","h"]).apply(lambda x: pd.Series({"bias":x.pred.sum()/x.actual.sum()-1,"wape":abs(x.pred-x.actual).sum()/x.actual.sum()})).round(3).unstack())
