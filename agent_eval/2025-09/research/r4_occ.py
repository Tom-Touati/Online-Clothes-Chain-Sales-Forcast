import pandas as pd, numpy as np, sys
sys.path.insert(0,"research"); import rules_v7 as R
it,di,st=R.load()
tags=it.occasion.dropna().str.split("; ").explode().value_counts()
fc=pd.read_csv("data/to_forecast.csv").style_color
tf=it.occasion.reindex(fc).dropna().str.split("; ").explode().value_counts()
print(pd.concat([tags,tf],axis=1,keys=["all","panel"]).fillna(0).astype(int).to_string())
mi=pd.read_parquet("data/monthly_item.parquet"); mi["m"]=pd.PeriodIndex(mi.month,freq="M")
U=mi.pivot_table(index="style_color",columns="m",values="units",aggfunc="sum").fillna(0)
def prof(base,months,tag,excl=None):
    b=pd.Period(base); ok=(it.launch.reindex(U.index)<b.start_time-pd.Timedelta(days=30))&(U[b]>0)
    occ=it.occasion.reindex(U.index).fillna("")
    has=occ.str.contains(tag,regex=False)
    if excl: has&=~occ.str.contains(excl,regex=False)
    seg=it.grp.reindex(U.index)
    out=[]
    for m in months:
        m=pd.Period(m)
        r_t=U[ok&has][m].sum()/U[ok&has][b].sum()
        # same-group untagged control, weighted by tagged items' group mix of base units
        w=U[ok&has].groupby(seg[ok&has])[b].sum(); w=w/w.sum()
        ctl=(U[ok&~has].groupby(seg[ok&~has])[m].sum()/U[ok&~has].groupby(seg[ok&~has])[b].sum()).reindex(w.index)
        out.append(round(r_t/(w*ctl).sum(),2))
    return int((ok&has).sum()),out
LYm=["2024-10","2024-11","2024-12","2025-01","2025-02"]
for tag,ex in [("Halloween",None),("Homecoming",None),("New Year's Eve","Homecoming"),("New Year's Eve",None),("Valentine's",None),("Valentine's","Homecoming"),("4th of July",None),("Graduation",None),("Christmas",None),("Holiday",None),("Thanksgiving",None),("Birthday",None),("Winter",None),("Ski",None),("Party",None),("Vacation",None),("Spring Break",None),("Festival",None),("Concert",None)]:
    if not it.occasion.fillna("").str.contains(tag,regex=False).any(): continue
    print(tag,"excl",ex,"LY Sep-24 base",prof("2024-09",LYm,tag,ex),"| TY Aug->Sep",prof("2025-08",["2025-09"],tag,ex), "| LY Aug->Sep",prof("2024-08",["2024-09"],tag,ex))
