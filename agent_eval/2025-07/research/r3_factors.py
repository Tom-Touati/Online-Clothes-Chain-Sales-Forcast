import pandas as pd, numpy as np, sys
sys.path.insert(0,"."); import rules as R
it, di, st = R.load()
c=pd.read_parquet("data/daily_channel.parquet"); c["m"]=c.date.dt.to_period("M")
print(c[(c.date>="2024-05-01")&(c.date<"2025-03-01")].pivot_table(index="channel",columns="m",values="units",aggfunc="sum").fillna(0).astype(int).to_string())
# Nordstrom daily July 2025 & 2024
for y in ["2024","2025"]:
    x=c[(c.channel.isin(["Nordstrom","Tiktok","Pacsun"]))&(c.date>=f"{y}-07-01")&(c.date<=f"{y}-08-31")]
    x=x.assign(wk=x.date.dt.to_period("W-SAT")).pivot_table(index="wk",columns="channel",values="units",aggfunc="sum")
    print(x.to_string())
B=pd.Period("2025-07"); T=[B+h for h in range(1,5)]
f=R.seasonal_factors(it,di,st,B,T)
for k in ["web","defl_w","ret"]: print(k, f[k].round(3).to_dict())
print("ret_adj",f["ret_adj"],"anc",f["anc"].round(3).to_dict())
print(f["grp_rel"].round(3).to_string())
# panel drift: LFL web/day vs site web/day by month pair, 2024 and 2025
ms=pd.period_range("2024-02","2025-07",freq="M")
W=di.pivot_table(index="style_color",columns="m",values="web_units",aggfunc="sum").reindex(columns=ms).fillna(0)
U=di.pivot_table(index="style_color",columns="m",values="units",aggfunc="sum").reindex(columns=ms).fillna(0)
site=di.groupby("m").web_units.sum()
s=pd.read_parquet("data/daily_site.parquet"); s["m"]=s.date.dt.to_period("M"); sw=s.groupby("m").web_units.sum(); su=s.groupby("m").units.sum()
rows=[]
for a,b in zip(ms[:-1],ms[1:]):
    ok=(it.launch.reindex(W.index)<a.start_time-pd.Timedelta(days=30))&(U[a]>0)
    rows.append((str(b), round(W[ok][b].sum()/W[ok][a].sum()/(sw[b]/sw[a]),3), round(U[ok][b].sum()/U[ok][a].sum()/(su[b]/su[a]),3), round(sw[b]/sw[a],3)))
print(pd.DataFrame(rows,columns=["m","web_drift","units_drift","site_web_mom"]).to_string())
