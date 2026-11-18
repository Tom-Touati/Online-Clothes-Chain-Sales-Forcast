import pandas as pd, numpy as np, sys
sys.path.insert(0,"."); import rules as R
it, di, st = R.load()
def lfl_pd(col, a, b, d=di):
    a,b=pd.Period(a),pd.Period(b)
    X=d[d.m.isin([a,b])].pivot_table(index="style_color",columns="m",values=col,aggfunc="sum").reindex(columns=[a,b]).fillna(0)
    ok=(it.launch.reindex(X.index)<a.start_time-pd.Timedelta(days=30))&(X[a]>0)
    return X[ok][b].sum()/X[ok][a].sum()*a.days_in_month/b.days_in_month
for a,b in [("2024-02","2024-03"),("2024-03","2024-04"),("2024-04","2024-05"),("2024-05","2024-06"),("2024-06","2024-07"),("2024-07","2024-08"),("2024-08","2024-09"),("2025-01","2025-02"),("2025-02","2025-03"),("2025-03","2025-04")]:
    print(a,b," ".join(f"{c}={lfl_pd(c,a,b):.3f}" for c in ["units","web_units","retail","other"]))
# factors used at origin 2025-03-31
o=pd.Timestamp("2025-03-31"); B=o.to_period("M"); T=[B+h for h in range(1,5)]
f=R.seasonal_factors(it,di[di.date<=o],st[st.date<=o],B,T)
print("prev origin web",f["web"].round(3).to_dict(),"ret",f["ret"].round(3).to_dict())
o=pd.Timestamp("2025-04-30"); B=o.to_period("M"); T=[B+h for h in range(1,5)]
f=R.seasonal_factors(it,di,st,B,T)
print("now web",f["web"].round(3).to_dict(),"ret",f["ret"].round(3).to_dict(), "a_feb",{k:round(v,3) for k,v in f["a_feb"].items()},"a_pre",{k:round(v,3) for k,v in f["a_pre"].items()},"cur",f["cur_pd"],f["cur_ret"],f["ret_adj"])
print(f["grp_rel"].round(2))
# weekly LFL web/day around Jul-Aug 2024 for items live & selling in June-24
jun=pd.Period("2024-06")
d=di[(di.date>="2024-06-01")&(di.date<="2024-09-30")]
ok=d[d.m==jun].groupby("style_color").units.sum(); ok=ok[(ok>0)&(it.launch.reindex(ok.index)<jun.start_time-pd.Timedelta(days=30))].index
x=d[d.style_color.isin(ok)].groupby("date").agg(w=("web_units","sum"))
s=pd.read_parquet("data/daily_site.parquet").set_index("date")
x["disc"]=s.web_discount
x["wk"]=x.index.to_period("W")
print((x.groupby("wk").agg(w=("w","mean"),disc=("disc","mean"))).assign(rel=lambda q:q.w/x.loc["2024-06","w"].mean()).round(3).to_string())
