import pandas as pd, numpy as np
sb = pd.read_csv("feedback/scoreboard.csv")
it = pd.read_csv("data/items.csv", parse_dates=["published_date","first_sale"]).set_index("style_color")
at = pd.read_csv("data/item_attributes.csv").set_index("style_color")
mi = pd.read_parquet("data/monthly_item.parquet")
print(sb.origin.unique())
def summ(df, key):
    t = df.groupby(["style_color"]+([key] if key else [])).agg(pred=("pred","sum"), actual=("actual","sum")).reset_index()
    t["ae"] = (t.pred-t.actual).abs()
    t["w20"] = (t.ae <= 0.2*t.actual); t["w50"] = (t.ae <= 0.5*t.actual)
    if not key: t["k"]="ALL"; key="k"
    g = t.groupby(key).agg(n=("pred","size"), pred=("pred","sum"), act=("actual","sum"), ae=("ae","sum"), w20=("w20","mean"), w50=("w50","mean"))
    g["wape"]=g.ae/g.act; g["a/p"]=g.act/g.pred
    return g.drop(columns="ae").round(3)
for o in ["2025-06","2025-10","2025-11","2025-12","2026-01","2026-02"]:
    d = sb[sb.origin==o]
    if d.month.nunique()==4: print(o, summ(d,None).to_string(header=False))
d = sb[sb.origin=="2026-02"].copy()
o = pd.Timestamp("2026-02-28")
L = it[["published_date","first_sale"]].min(axis=1)
d["age"] = pd.cut(((o - pd.to_datetime(L.reindex(d.style_color).values)).days/30.44).values, [-99,1,2,4,8,14,99], labels=["<1","1-2","2-4","4-8","8-14","14+"])
d["type"] = it.product_type.reindex(d.style_color).values
d["tier"] = at.occasion_tier.reindex(d.style_color).fillna("none").values
# channel shares base month
b = mi[mi.month=="2026-02"].set_index("style_color")
sh = (b.retail/b.units).reindex(d.style_color)
d["ret_sh"] = pd.cut(sh.fillna(0).values, [-1,0.001,0.2,0.5,1.01], labels=["0","<20","20-50",">50"])
osh = (b["drop"]/b.units).reindex(d.style_color)
d["drop_sh"] = pd.cut(osh.fillna(0).values, [-1,0.001,0.1,0.3,1.01], labels=["0","<10","10-30",">30"])
bu = b.units.reindex(d.style_color).fillna(0).values
d["size"] = pd.cut(bu, [-1,30,100,300,1000,1e9], labels=["<30","30-100","100-300","300-1k","1k+"])
for k in ["age","tier","ret_sh","drop_sh","size"]:
    print(summ(d,k).to_string())
g = summ(d,"type"); print(g[g.n>=15].sort_values("a/p").to_string())
# monthly per type
d["mon"]=d.month
pm = d.groupby(["type","mon"])[["pred","actual"]].sum().unstack()
r = (pm["actual"]/pm["pred"]).round(2); r["n"]=d.groupby("type").style_color.nunique()
print(r[r.n>=15].to_string())
# tags
occ = it.occasion.reindex(d.style_color).fillna("").values
tags = sorted({t for s in it.occasion.dropna() for t in s.split("; ")})
rows=[]
for tg in tags:
    m = pd.Series(occ).str.contains(tg, regex=False).values
    if m.sum()<4*8: continue
    x = d[m]; rows.append((tg, m.sum()//4, x.actual.sum()/x.pred.sum()))
print(pd.DataFrame(rows, columns=["tag","n","a/p"]).round(3).to_string())
