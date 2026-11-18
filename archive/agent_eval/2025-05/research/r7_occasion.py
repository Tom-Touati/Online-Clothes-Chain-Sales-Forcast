import sys; sys.path.insert(0,".")
import pandas as pd, numpy as np, rules as R
it,di,st=R.load()
ms=list(pd.period_range("2024-04","2024-10",freq="M"))
U=di.pivot_table(index="style_color",columns="m",values="units",aggfunc="sum").reindex(columns=ms).fillna(0)
B=pd.Period("2024-05")
ok=(it.launch.reindex(U.index)<B.start_time-pd.Timedelta(days=30))&(U[B]>0)
X=U[ok].copy(); X["grp"]=it.grp.reindex(X.index); X["seg"]=it.seg.reindex(X.index)
# expected per item = item's May units x segment(cum from May) move  (segment = product type x sleeve)
cols=ms[2:]
segmove=X.groupby("seg")[cols].sum().div(X.groupby("seg")[B].sum(),axis=0)
occ=it.occasion.fillna("").reindex(X.index)
for tag in ["4th of July","Graduation","Homecoming","Halloween","Valentine's Day","New Year's Eve"]:
    m=occ.str.contains(tag,regex=False)
    Y=X[m]
    exp=segmove.reindex(Y.seg).values*Y[[B]].values
    act=Y[cols].sum().values; e=exp.sum(axis=0)
    print(f"{tag:16s} n={m.sum():3d} mayU={Y[B].sum():6.0f}", dict(zip([str(c) for c in cols],np.round(act/e,2))))
# current panel tag counts by group
items=pd.read_csv("data/to_forecast.csv").style_color
t=it.reindex(items)
for tag in ["4th of July","Graduation","Homecoming","Halloween"]:
    m=t.occasion.fillna("").str.contains(tag,regex=False)
    print(tag, m.sum(), t[m].product_type.value_counts().head(5).to_dict())
