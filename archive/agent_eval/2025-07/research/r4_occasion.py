import pandas as pd, numpy as np, sys
sys.path.insert(0,"."); import rules as R
it, di, st = R.load()
it["occ"]=it.occasion.fillna("")
tags=pd.Series([t.strip() for o in it.occ for t in o.split(";") if t.strip()]).value_counts()
print(tags.head(25).to_string())
ms=pd.period_range("2024-01","2025-07",freq="M")
U=di.pivot_table(index="style_color",columns="m",values="units",aggfunc="sum").reindex(columns=ms).fillna(0)
def prof(base, months, tag):
    ok=(it.launch.reindex(U.index)<base.start_time-pd.Timedelta(days=30))&(U[base]>0)
    X=U[ok]; seg=it.seg.reindex(X.index); has=it.occ.reindex(X.index).str.contains(tag,regex=False)
    if has.sum()<3: return None
    # tagged items vs untagged items of the same segments (weighted by tagged base units)
    out={"n":int(has.sum())}
    for m in months:
        r_seg=(X[~has].groupby(seg[~has])[m].sum()/X[~has].groupby(seg[~has])[base].sum())
        w=X[has].groupby(seg[has])[base].sum()
        exp=(w*r_seg.reindex(w.index).fillna(X[~has][m].sum()/X[~has][base].sum())).sum()
        out[str(m)]=round(X[has][m].sum()/exp,2)
    return out
B=pd.Period("2024-07"); T=[B+h for h in range(1,5)]
for tag in tags.index[:22]:
    p=prof(B,T,tag)
    q=prof(pd.Period("2024-06"),[pd.Period("2024-07")],tag); q2=prof(pd.Period("2025-06"),[pd.Period("2025-07")],tag)
    if p: print(f"{tag:22s}", p, "Jun->Jul LY", q and q["2024-07"], "TY", q2 and q2["2025-07"], "n TY", q2 and q2["n"])
# season groups: rel to panel, cumulative Jan->Jul, this year vs last year; and Jul-24 -> Nov-24 vs Jan-25
def grel(a,b):
    ok=(it.launch.reindex(U.index)<a.start_time-pd.Timedelta(days=30))&(U[a]>0)
    X=U[ok]; g=it.grp.reindex(X.index)
    return ((X.groupby(g)[b].sum()/X.groupby(g)[a].sum())/(X[b].sum()/X[a].sum())).round(3)
print("Jan->Jul 2024", grel(pd.Period("2024-01"),pd.Period("2024-07")).to_dict())
print("Jan->Jul 2025", grel(pd.Period("2025-01"),pd.Period("2025-07")).to_dict())
print("Mar->Jul 2024", grel(pd.Period("2024-03"),pd.Period("2024-07")).to_dict())
print("Mar->Jul 2025", grel(pd.Period("2025-03"),pd.Period("2025-07")).to_dict())
print("Nov-24->Jul-25", grel(pd.Period("2024-11"),pd.Period("2025-07")).to_dict())
print("Jul-24->Nov-24", grel(pd.Period("2024-07"),pd.Period("2024-11")).to_dict())
print("Jun->Jul 2024", grel(pd.Period("2024-06"),pd.Period("2024-07")).to_dict(), "2025", grel(pd.Period("2025-06"),pd.Period("2025-07")).to_dict())
# group shares of panel units per month (all items)
g=U.groupby(it.grp.reindex(U.index)).sum(); print((g/g.sum()).round(3).T.to_string())
