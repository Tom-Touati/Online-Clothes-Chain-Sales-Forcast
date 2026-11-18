import pandas as pd, numpy as np
sb = pd.read_csv("feedback/scoreboard.csv")
it = pd.read_csv("data/items.csv", parse_dates=["published_date","first_sale"]).set_index("style_color")
it["launch"] = it[["published_date","first_sale"]].min(axis=1)
print(sb.groupby(["origin","month","h"]).agg(n=("pred","size"),pred=("pred","sum"),act=("actual","sum")))
mi = pd.read_parquet("data/monthly_item.parquet")
mi["web"]=0
for o in sorted(sb.origin.unique()):
    s = sb[(sb.origin==o)&(sb.month=="2025-04")].copy()
    if s.empty: continue
    s["launch"]=it.launch.reindex(s.style_color).values
    s["lm"]=s.launch.dt.to_period("M").astype(str)
    s["pt"]=it.product_type.reindex(s.style_color).values
    print("\n=== origin",o,"Apr-25: pred",s.pred.sum(),"act",s.actual.sum(),"WAPE",(s.pred-s.actual).abs().sum()/s.actual.sum())
    g=s.groupby(s.lm.where(s.lm>="2024-12","old")).agg(n=("pred","size"),pred=("pred","sum"),act=("actual","sum")); g["r"]=g.act/g.pred; print(g)
    g=s.groupby("pt").agg(n=("pred","size"),pred=("pred","sum"),act=("actual","sum")); g["r"]=g.act/g.pred; print(g[g.n>=5].sort_values("pred",ascending=False).head(20))
    s["q"]=pd.qcut(s.pred.rank(method="first"),5,labels=False)
    g=s.groupby("q").agg(pred=("pred","sum"),act=("actual","sum")); g["r"]=g.act/g.pred; print(g)
    # channel mix in base month
    bm = (pd.Period(o[:7])).strftime("%Y-%m") if len(o)>=7 else None
