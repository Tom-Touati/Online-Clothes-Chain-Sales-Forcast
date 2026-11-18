import sys; sys.path.insert(0,".")
import pandas as pd, numpy as np, rules as R
it,di,st=R.load()
it["dress"]=np.where(it.product_type=="Dress","DRESS","x")
it["j4"]=np.where(it.occasion.fillna("").str.contains("4th of July"),"J4","x")
it["grad"]=np.where(it.occasion.fillna("").str.contains("Graduation"),"GRAD","x")
it["homec"]=np.where(it.occasion.fillna("").str.contains("Homecoming"),"HOMEC","x")
ms=list(pd.period_range("2024-02","2025-05",freq="M"))
U=di.pivot_table(index="style_color",columns="m",values="units",aggfunc="sum").reindex(columns=ms).fillna(0)
def step(key):
    rows={}
    for a,b in zip(ms[:-1],ms[1:]):
        ok=(it.launch.reindex(U.index)<a.start_time-pd.Timedelta(days=30))&(U[a]>0)
        X=U[ok]; k=key.reindex(X.index)
        r=X.groupby(k)[b].sum()/X.groupby(k)[a].sum()
        allr=X[b].sum()/X[a].sum()
        rows[str(b)]=(r/allr).round(2)
        rows[str(b)]["ALL/day"]=round(allr*a.days_in_month/b.days_in_month,3)
    return pd.DataFrame(rows)
print(step(it.grp).to_string())
for c in ["dress","j4","grad","homec"]:
    print(step(it[c]).drop("x",errors="ignore").to_string())
