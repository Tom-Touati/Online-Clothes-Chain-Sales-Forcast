import pandas as pd, numpy as np
import sys; sys.path.insert(0,".")
import rules as R
it,di,st=R.load()
s=pd.read_parquet("data/daily_site.parquet"); s["m"]=s.date.dt.to_period("M"); s["other"]=s.units-s.web_units-s.retail
spd=s.groupby("m")[["web_units","retail","other"]].sum().div(s.groupby("m").date.nunique(),axis=0)
M=sorted(di.m.unique())
sb=pd.read_csv("feedback/scoreboard.csv")
rows=[]
for a in M[1:-1]:
    for h in range(1,5):
        b=a+h
        if b>M[-1]: continue
        X=di[di.m.isin([a,b])].pivot_table(index="style_color",columns="m",values=["web_units","retail","other"],aggfunc="sum").fillna(0)
        ok=(it.launch.reindex(X.index)<a.start_time-pd.Timedelta(days=30))&(X[("web_units",a)]+X[("retail",a)]+X[("other",a)]>0)
        r={"a":str(a),"h":h}
        for c in ["web_units","other","retail"]:
            pan=X.loc[ok,(c,b)].sum()/X.loc[ok,(c,a)].sum()*a.days_in_month/b.days_in_month
            r[c[:3]]=pan/(spd.loc[b,c]/spd.loc[a,c])
        # scoreboard-item drift (items in scoreboard list for origin a, all-channel)
        L=sb[(sb.origin==str(a))].style_color.unique()
        if len(L):
            Y=di[di.style_color.isin(L)&di.m.isin([a,b])].groupby("m").units.sum()
            tot=s.groupby("m").units.sum()
            r["sb_all"]=(Y[b]/Y[a])/(tot[b]/tot[a])
        rows.append(r)
d=pd.DataFrame(rows).round(3)
print(d[d.h==1].to_string()); print(d[d.h==4].to_string())
