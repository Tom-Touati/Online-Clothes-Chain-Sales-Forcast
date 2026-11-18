import pandas as pd, numpy as np
sb=pd.read_csv("feedback/scoreboard.csv")
it=pd.read_csv("data/items.csv")
at=pd.read_csv("data/item_attributes.csv")
print(at.columns.tolist())
sb=sb.merge(it[["style_color","product_type","occasion"]],on="style_color",how="left")
sb["key"]=sb.origin+">"+sb.month
tot=sb.groupby("key").apply(lambda g:g.actual.sum()/g.pred.sum())
def rel(col,minn=10):
    g=sb.groupby([col,"key"]).agg(n=("pred","size"),p=("pred","sum"),a=("actual","sum"))
    g["r"]=g.a/g.p/g.index.get_level_values("key").map(tot).values
    t=g.r.unstack().round(2); n=g.n.unstack().max(axis=1)
    return t[n>=minn]
print(rel("product_type",12).to_string())
x=sb.assign(tag=sb.occasion.fillna("").str.split("; ")).explode("tag")
x=x[x.tag!=""]
g=x.groupby(["tag","key"]).agg(n=("pred","size"),p=("pred","sum"),a=("actual","sum"))
g["r"]=g.a/g.p/g.index.get_level_values("key").map(tot).values
print(g.r.unstack().round(2).to_string())
