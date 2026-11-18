# Oct-14 spike concentration; TY vs LY group moves; factor dump at the Oct origin; tag profiles from Oct-24
import pandas as pd, numpy as np, sys
sys.path.insert(0,"."); import rules
it,di,st=rules.load()
items=pd.read_csv("data/to_forecast.csv").style_color.tolist()
o=di[(di.date>="2025-10-07")&(di.date<="2025-10-21")]
p=o.pivot_table(index="style_color",columns="date",values="web_units",aggfunc="sum").fillna(0)
d14=p[pd.Timestamp("2025-10-14")]; oth=p.drop(columns=[pd.Timestamp("2025-10-14")]).mean(axis=1)
print("Oct14 web",d14.sum(),"other-day mean",oth.sum().round(0))
r=(d14/oth.clip(lower=1)); top=d14.sort_values(ascending=False).head(15)
print(pd.DataFrame({"d14":top,"avg":oth[top.index].round(1),"ratio":r[top.index].round(2)}).to_string())
print("share of Oct14 units in items with ratio>5:", d14[r>5].sum()/d14.sum())
# discount on Oct14 per item
dd=di[di.date=="2025-10-14"].set_index("style_color"); print("Oct14 web disc", (dd.web_disc_sum.sum()/dd.web_units.sum()).round(3))
# group moves TY vs LY rel. to panel
def cum_rel(ms):
    Uq=di[di.m.isin(ms)].pivot_table(index="style_color",columns="m",values="units",aggfunc="sum").reindex(columns=ms).fillna(0)
    g,_=rules.lfl_chain(Uq,it.launch,ms,it.grp); a,_=rules.lfl_chain(Uq,it.launch,ms,pd.Series("ALL",index=Uq.index))
    return (g[ms[-1]]/a.loc["ALL",ms[-1]]).round(3)
P=pd.Period
for a in ["2025-06","2025-07","2025-08","2025-09"]:
    a=P(a,"M"); print(a,"->",a+1,"TY",cum_rel([a,a+1]).to_dict(),"LY",cum_rel([a-12,a-11]).to_dict())
print("LY Oct->Nov/Dec/Jan/Feb cum:",{str(t):cum_rel([P("2024-10","M")]+[P("2024-10","M")+k for k in range(1,i+1)]).to_dict() for i,t in enumerate(["2024-11","2024-12","2025-01","2025-02"],1)})
B=P("2025-10","M"); targets=[B+h for h in range(1,5)]
f=rules.seasonal_factors(it,di,st,B,targets)
print("seas_pow",round(f["seas_pow"],3),"anc",f["anc"].round(3).to_dict())
print("drift",f["drift"].round(3).to_dict()); print("web",f["web"].round(3).to_dict()); print("other",f["other"].round(3).to_dict()); print("ret",f["ret"].round(3).to_dict(),"ret_adj",round(f["ret_adj"],3))
print("grp_rel\n",f["grp_rel"].round(3).to_string())
# tag profiles from Oct-24 base: tagged vs same-group untagged LFL (items live before Sep-24, sold in Oct-24)
mi=pd.read_parquet("data/monthly_item.parquet")
M=mi.pivot_table(index="style_color",columns="month",values="units",aggfunc="sum").fillna(0)
ms=["2024-10","2024-11","2024-12","2025-01","2025-02","2025-03"]
ok=M.index[(M["2024-10"]>0)&(it.launch.reindex(M.index)<"2024-09-01")]
X=M.loc[ok,ms]; occ=it.occasion.reindex(ok).fillna(""); grp=it.grp.reindex(ok)
for tag in ["Halloween","Homecoming","New Year's Eve","Valentine's Day","Graduation","4th of July","St. Patrick's Day"]:
    has=occ.str.contains(tag,regex=False); rows=[]
    for g in X[has].groupby(grp[has]).groups:
        t=X[has&(grp==g)].sum(); u=X[(~has)&(grp==g)].sum(); rows.append((t,u))
    tt=sum(r[0] for r in rows); uu=sum(r[1] for r in rows)
    # untagged same-group mix weighted by tagged base
    w=X[has].groupby(grp[has])["2024-10"].sum()
    uidx=sum(w[g]*(X[(~has)&(grp==g)].sum()/X[(~has)&(grp==g)]["2024-10"].sum()) for g in w.index)/w.sum()
    print(f"{tag:16s} n={has.sum():3d}", ((tt/tt['2024-10'])/uidx).round(2).to_dict())
