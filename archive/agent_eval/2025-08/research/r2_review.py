import pandas as pd, numpy as np
sb = pd.read_csv("feedback/scoreboard.csv")
it = pd.read_csv("data/items.csv", parse_dates=["published_date","first_sale"]).set_index("style_color")
it["launch"]=it[["published_date","first_sale"]].min(axis=1)
mi = pd.read_parquet("data/monthly_item.parquet")
W=["Hoodie","Sweater","Sweatpants","Cardigan","Jacket","Coat","Sweatshirt","Leggings","Scarf","Leg Warmers","Tights","Vest","Knitted Top","Beanie","Gloves"]
S=["Shorts","Short Jeans","Skort","Crop Top","Tank Top","Swimwear Top","Swimwear Bottom","Romper","Bikini","Anklet","Belly Chain","Bracelet"]
it["grp"]=np.where(it.product_type.isin(W),"WINTER",np.where(it.product_type.isin(S),"SUMMER","CORE"))
a=sb[sb.month=="2025-08"].copy()
a=a.join(it[["product_type","grp","occasion","launch"]],on="style_color")
def rep(df,key,minn=8):
    tot=df.actual.sum()/df.pred.sum()
    g=df.groupby(key).agg(n=("pred","size"),pred=("pred","sum"),act=("actual","sum"))
    g["ratio"]=g.act/g.pred; g["rel"]=g.ratio/tot
    return g[g.n>=minn].round(3)
for o in ["2025-07","2025-06","2025-05"]:
    d=a[a.origin==o]
    print("origin",o,"tot ratio",round(d.actual.sum()/d.pred.sum(),3), "WAPE", round((d.pred-d.actual).abs().sum()/d.actual.sum(),3))
    print(rep(d,"grp").to_string())
d=a[a.origin=="2025-07"].copy()
print(rep(d,"product_type",10).sort_values("rel").to_string())
d["age"]=((pd.Timestamp("2025-08-01")-d.launch).dt.days/30.44).clip(-1,24).round().clip(upper=13)
print(rep(d,"age",3).to_string())
# channel mix in July
m7=mi[mi.month=="2025-07"].set_index("style_color")
d["rshare"]=(m7.retail/m7.units).reindex(d.style_color).values
d["rb"]=pd.cut(d.rshare,[-0.01,0.1,0.25,0.4,1.0])
print(rep(d,"rb").to_string())
d["q"]=pd.qcut(d.pred.rank(method="first"),5,labels=False)
print(rep(d,"q").to_string())
for tag in ["Halloween","Homecoming","New Year's Eve","4th of July","Graduation","Valentine's Day","Christmas","Thanksgiving","Back to School"]:
    for o in ["2025-07","2025-06"]:
        dd=a[a.origin==o]; h=dd.occasion.fillna("").str.contains(tag,regex=False)
        if h.sum()>0: print(tag,o,h.sum(), round((dd[h].actual.sum()/dd[h].pred.sum())/(dd.actual.sum()/dd.pred.sum()),3))
# all tags present
print(it.occasion.dropna().str.split("; ").explode().value_counts().head(25))
