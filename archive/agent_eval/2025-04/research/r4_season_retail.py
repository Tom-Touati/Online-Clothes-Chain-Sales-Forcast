import pandas as pd, numpy as np, sys
sys.path.insert(0,"."); import rules as R
it, di, st = R.load()
sb = pd.read_csv("feedback/scoreboard.csv")
for o in ["2025-02","2025-03"]:
    s = sb[(sb.origin==o)&(sb.month=="2025-04")].copy()
    s["grp"]=it.grp.reindex(s.style_color).values; tot=s.actual.sum()/s.pred.sum()
    g=s.groupby("grp")[["pred","actual"]].sum(); g["rel"]=g.actual/g.pred/tot; print(o, g.round(2).to_dict())
    s["occ"]=it.occasion.reindex(s.style_color).fillna("").values
    for tag in ["Vacation","Festival","Beach","Graduation","Prom","Party","Date Night","Casual","Wedding Guest","Mother","4th of July","Memorial"]:
        q=s[s.occ.str.contains(tag,case=False)]
        if len(q)>=5: print("   ",tag,len(q),round(q.actual.sum()/q.pred.sum()/tot,2))
# LFL by group: this year vs last year, per month
def lfl_grp(a,b):
    a,b=pd.Period(a),pd.Period(b)
    X=di[di.m.isin([a,b])].pivot_table(index="style_color",columns="m",values="units",aggfunc="sum").reindex(columns=[a,b]).fillna(0)
    ok=(it.launch.reindex(X.index)<a.start_time-pd.Timedelta(days=30))&(X[a]>0); X=X[ok]
    g=X.groupby(it.grp.reindex(X.index)).sum(); r=g[b]/g[a]; return (r/(X[b].sum()/X[a].sum())).round(2).to_dict()
for a,b in [("2024-02","2024-03"),("2024-03","2024-04"),("2024-04","2024-05"),("2025-02","2025-03"),("2025-03","2025-04")]:
    print("grp rel LFL",a,b,lfl_grp(a,b))
# occasion tags present in to_forecast
tf=pd.read_csv("data/to_forecast.csv").style_color
print(it.occasion.reindex(tf).fillna("").str.split("; ").explode().value_counts().head(25).to_dict())
# retail: day-of-week and last weeks
r=st[st.date>="2025-03-01"].groupby("date").units.sum().to_frame("u"); r["dow"]=r.index.dayofweek; r["wk"]=r.index.to_period("W")
print(r.pivot_table(index="wk",columns="dow",values="u").round(0).to_string())
print(st[st.date>="2025-04-21"].pivot_table(index="date",columns="store",values="units",aggfunc="sum").to_string())
