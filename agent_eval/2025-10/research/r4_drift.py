# monthly panel-vs-site drift (all channels and web), TY vs LY: LFL per-day move / site per-day move
import pandas as pd, numpy as np, sys
sys.path.insert(0,"."); import rules
it,di,st=rules.load()
s=pd.read_parquet("data/daily_site.parquet"); s["m"]=s.date.dt.to_period("M")
sp=s.groupby("m")[["units","web_units"]].mean()
def drift(a,col,scol):
    b=a+1
    X=di[di.m.isin([a,b])].pivot_table(index="style_color",columns="m",values=col,aggfunc="sum").reindex(columns=[a,b]).fillna(0)
    ok=(it.launch.reindex(X.index)<a.start_time-pd.Timedelta(days=30))&(X[a]>0)
    lfl=X[ok][b].sum()/X[ok][a].sum()*a.days_in_month/b.days_in_month
    return lfl/(sp.loc[b,scol]/sp.loc[a,scol])
rows=[]
for a in pd.period_range("2024-02","2025-09",freq="M"):
    rows.append((str(a+1),round(drift(a,"units","units"),3),round(drift(a,"web_units","web_units"),3)))
d=pd.DataFrame(rows,columns=["to","all","web"]).set_index("to"); print(d.to_string())
# share of site units from items launched within 60 days of the month start
mi=pd.read_parquet("data/monthly_item.parquet"); mi["launch"]=mi.style_color.map(it.launch)
mi["age"]=(pd.PeriodIndex(mi.month,freq="M").start_time-mi.launch).dt.days
x=mi.groupby("month").apply(lambda g: pd.Series({"new60":g.units[g.age<60].sum()/g.units.sum(),"n_new":(g.age.between(-31,0)&(g.units>0)).sum()}))
print(x.loc["2024-06":].round(3).T.to_string())
