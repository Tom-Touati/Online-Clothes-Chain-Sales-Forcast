import pandas as pd, numpy as np, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"."); import rules as R
it,di,st=R.load(); B=pd.Period("2025-12"); T=[B+h for h in range(1,5)]
f=R.seasonal_factors(it,di,st,B,T)
for k in ["web","defl_w","drift","other","ret"]: print(k, f[k].round(3).to_dict())
print("seas_pow",round(f["seas_pow"],3),"anc",f["anc"].round(3).to_dict(),"ret_adj",f["ret_adj"])
print(f["grp_rel"].round(3))
R.P["RET_MATURE"]=False; f2=R.seasonal_factors(it,di,st,B,T); print("ret flagship-v10", f2["ret"].round(3).to_dict()); R.P["RET_MATURE"]=True
p=pd.read_csv("predictions.csv"); items=pd.read_csv("data/to_forecast.csv").style_color
dec=di[(di.m==B)&di.style_color.isin(items)].units.sum()
print("Dec actual these items",dec, "per day",round(dec/31))
tot=p.groupby("month").pred.sum(); print((tot/1000).round(1).to_dict(), "per day", (tot/np.array([31,28,31,30])).round(0).to_dict(), "4m",round(tot.sum()))
p=p.join(it[["grp","occasion","tier","product_type"]],on="style_color")
print((p.pivot_table(index="grp",columns="month",values="pred",aggfunc="sum")/1000).round(1))
dd=di[(di.m==B)].groupby("style_color").units.sum()
p["dec"]=p.style_color.map(dd).fillna(0)
for t in ["Halloween","New Year's Eve","Valentine's","Graduation","St. Patrick"]:
    m=p.occasion.fillna("").str.contains(t,regex=False)
    print(t, m.sum()//4, "Dec",int(p[m].dec.sum()/4), (p[m].groupby("month").pred.sum()/1000).round(1).to_dict())
# ratio Jan/Dec per day by type, top types
g=p.groupby(["product_type","month"]).pred.sum().unstack(); g["dec"]=p[p.month=="2026-01"].groupby("product_type").dec.sum()
g=g.sort_values("dec",ascending=False).head(12)
print((g.div(g.dec,axis=0)).round(2))
