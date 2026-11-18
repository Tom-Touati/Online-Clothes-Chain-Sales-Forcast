import pandas as pd, numpy as np
sb=pd.read_csv("feedback/scoreboard.csv")
it=pd.read_csv("data/items.csv",parse_dates=["published_date","first_sale"])
it["launch"]=it[["published_date","first_sale"]].min(axis=1)
mi=pd.read_parquet("data/monthly_item.parquet")
print(sb.head())
s=sb[sb.origin=="2025-04"].merge(it[["style_color","product_type","occasion","launch"]],on="style_color",how="left")
s["age"]=((pd.Timestamp("2025-04-01")-s.launch).dt.days/30.44)
s["agec"]=pd.cut(s.age,[-99,0,1,2,3,6,12,999],labels=["Apr","Mar","Feb","Jan","Oct-Dec","6-12","12+"])
def agg(g): return pd.Series(dict(n=len(g),pred=g.pred.sum(),act=g.actual.sum(),ratio=g.actual.sum()/g.pred.sum(),wape=(g.pred-g.actual).abs().sum()/g.actual.sum()))
print(s.groupby("agec").apply(agg).round(3))
print(s.groupby("product_type").apply(agg).query("n>=8").sort_values("act",ascending=False).round(3))
occ=s.assign(tag=s.occasion.fillna("").str.split("; ")).explode("tag")
print(occ.groupby("tag").apply(agg).query("n>=8").round(3))
# size quintile
s["q"]=pd.qcut(s.pred.rank(method="first"),5,labels=False)
print(s.groupby("q").apply(agg).round(3))
# channel mix: share web in April
m=mi[mi.month=="2025-04"].set_index("style_color")
s["websh"]=s.style_color.map(m.online/m.units)
s["retsh"]=s.style_color.map(m.retail/m.units)
s["rq"]=pd.cut(s.retsh,[-0.01,0.1,0.25,0.4,1.01])
print(s.groupby("rq").apply(agg).round(3))
# h analysis across origins for May
print(sb[sb.month=="2025-05"].groupby("origin").apply(agg).round(3))
# items with zero pred but actual
print("zero pred", s[s.pred==0].actual.sum(), (s.pred==0).sum())
print("in20",(abs(s.pred-s.actual)<=0.2*s.actual).mean(),"in50",(abs(s.pred-s.actual)<=0.5*s.actual).mean())
