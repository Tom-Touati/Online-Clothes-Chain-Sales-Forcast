import pandas as pd, numpy as np, sys
sys.path.insert(0,"research"); import rules_v7 as R
it,di,st=R.load()
o=di.date.max(); B=o.to_period("M"); T=[B+h for h in range(1,5)]
f=R.seasonal_factors(it,di,st,B,T)
for k in ["web","defl_w","drift","other","ret","anc"]:
    print(k, f[k].round(3).to_dict() if hasattr(f[k],"to_dict") else f[k])
print("ret_adj",f["ret_adj"])
print(f["grp_rel"].round(3).to_string())
# group LFL moves TY vs LY, monthly, rel to panel
mi=pd.read_parquet("data/monthly_item.parquet"); mi["m"]=pd.PeriodIndex(mi.month,freq="M")
U=mi.pivot_table(index="style_color",columns="m",values="units",aggfunc="sum").fillna(0)
def mv(a,b):
    ok=(it.launch.reindex(U.index)<a.start_time-pd.Timedelta(days=30))&(U[a]>0)
    g=U[ok].groupby(it.grp.reindex(U.index)[ok]); r=g[b].sum()/g[a].sum(); tot=U[ok][b].sum()/U[ok][a].sum()
    return (r/tot).round(3)
for a in ["2024-06","2024-07","2024-08","2024-09","2024-10","2024-11","2024-12","2025-01","2025-06","2025-07","2025-08"]:
    a=pd.Period(a); print(a,"->",a+1, mv(a,a+1).to_dict())
# group share of panel units
sh=U.groupby(it.grp.reindex(U.index)).sum(); sh=sh/sh.sum()
print(sh.loc[:,pd.Period("2024-06"):].round(3).to_string())
# store status end of Sep
s=st[st.date>="2025-09-20"]; print(s.groupby("store").date.max())
