import pandas as pd, numpy as np, sys
sys.path.insert(0,"."); import rules as R
it,di,st=R.load()
sb=pd.read_csv("feedback/scoreboard.csv")
for o in ["2025-07","2025-08"]:
    B=pd.Period(o); o_=B.end_time.normalize(); t=B+1
    d=di[di.date<=o_]
    f=R.seasonal_factors(it,d,st[st.date<=o_],B,[B+h for h in range(1,5)])
    items=sb[sb.origin==o].style_color.unique()
    X=di[di.m.isin([B,t])&di.style_color.isin(items)]
    ok=items[(it.launch.reindex(items)<B.start_time-pd.Timedelta(days=30)).values]
    out={}
    for col in ["web_units","retail","other","units"]:
        p=X[X.style_color.isin(ok)].pivot_table(index="style_color",columns="m",values=col,aggfunc="sum").fillna(0)
        out[col]=round(p[t].sum()/p[B].sum()*B.days_in_month/t.days_in_month,3)
    print(o,"actual LFL per-day",out,"| assumed web",round(f["web"][t-12],3),"ret",round(f["ret"][t-12],3),"other",round(f["other"][t-12],3),"pow",round(f["seas_pow"],2),"anc",f["anc"].round(2).to_dict())
    # forecast components: ramp & total
    F=R.forecast(o_,list(items),it,di,st)
    a=di[(di.m==t)&di.style_color.isin(items)].units.sum(); b=di[(di.m==B)&di.style_color.isin(items)].units.sum()
    print("  pred",round(F[str(t)].sum()),"act",a,"base month",b, "pred/base per day",round(F[str(t)].sum()/b*B.days_in_month/t.days_in_month,3),"act/base",round(a/b*B.days_in_month/t.days_in_month,3))
