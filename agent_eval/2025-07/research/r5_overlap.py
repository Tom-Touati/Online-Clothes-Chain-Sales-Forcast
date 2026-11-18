import pandas as pd, numpy as np, sys
sys.path.insert(0,"."); import rules as R
it, di, st = R.load(); it["occ"]=it.occasion.fillna("")
ms=pd.period_range("2024-01","2025-07",freq="M")
U=di.pivot_table(index="style_color",columns="m",values="units",aggfunc="sum").reindex(columns=ms).fillna(0)
B=pd.Period("2024-07"); T=[B+h for h in range(1,5)]
ok=(it.launch.reindex(U.index)<B.start_time-pd.Timedelta(days=30))&(U[B]>0)
X=U[ok]; seg=it.seg.reindex(X.index); occ=it.occ.reindex(X.index)
def prof(mask,label):
    r={}
    for m in T:
        r_seg=(X[~mask&(occ=="")].groupby(seg)[m].sum()/X[~mask&(occ=="")].groupby(seg)[B].sum())
        w=X[mask].groupby(seg[mask])[B].sum()
        r[str(m)]=round(X[mask][m].sum()/(w*r_seg.reindex(w.index).fillna(1)).sum(),2)
    print(f"{label:35s} n={mask.sum():3d} units={X[mask][B].sum():6.0f}", r, dict(seg[mask].value_counts().head(3)))
H=occ.str.contains("Homecoming"); N=occ.str.contains("New Year"); V=occ.str.contains("Valentine"); HW=occ.str.contains("Halloween")
prof(H,"Homecoming"); prof(N&~H,"NYE not Homecoming"); prof(V&~H&~N,"Valentine only (no HC/NYE)"); prof(HW,"Halloween"); prof(HW&~V&~N&~H,"Halloween only")
prof(occ=="", "untagged (control)")
# TY panel counts
P=pd.read_csv("data/to_forecast.csv").style_color
o=it.occ.reindex(P)
for t in ["Homecoming","New Year","Valentine","Halloween","4th of July","Graduation"]: print(t,o.str.contains(t).sum(), "NYE&~HC" if t=="New Year" else "", (o.str.contains("New Year")&~o.str.contains("Homecoming")).sum() if t=="New Year" else "")
