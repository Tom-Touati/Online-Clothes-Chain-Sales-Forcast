"""Same-season allocation test: Dec-24 base -> Jan..Apr-25, group x month oracle totals (LY factors for a
Dec-24 origin need Dec-23 data, which does not exist).  Item logic only: base blend, flash cap, ramp, tags."""
import pandas as pd, numpy as np, sys, copy
sys.path.insert(0,"."); import rules as R
it,di,st=R.load()
o=pd.Timestamp("2024-12-31"); B=o.to_period("M"); T=[B+h for h in range(1,5)]
dB=di[di.m==B].groupby("style_color").units.sum()
items=dB[dB>0].index.intersection(it.index[it.launch<=o]).tolist()
A=di[di.m.isin(T)].pivot_table(index="style_color",columns="m",values="units",aggfunc="sum").reindex(index=items,columns=T).fillna(0)
A.columns=[str(c) for c in A.columns]
one=lambda idx: pd.Series(1.0,index=idx)
ly=[t-12 for t in T]
fac=dict(web=pd.Series(1.0,index=ly),ret=pd.Series(1.0,index=ly),other=pd.Series(1.0,index=ly),
         grp_rel=pd.DataFrame(1.0,index=["WINTER","SUMMER","CORE"],columns=ly),seg_rel=pd.DataFrame(index=[],columns=ly,dtype=float),
         seg_n=pd.Series(dtype=float),seas_pow=1.0,anc=pd.Series(1.0,index=["WINTER","SUMMER","CORE"]))
grp=it.grp.reindex(items); seg=it.seg.reindex(items)
def run(p=None, occ=None, tier=None, key="grp"):
    P0,O0,T0=dict(R.P),R.OCCASION,R.TIER_IDX
    if p: R.P.update(p)
    if occ is not None: R.OCCASION=occ
    if tier is not None: R.TIER_IDX=tier
    F=R.forecast(o,items,it,di,st,factors=fac)
    R.P.clear(); R.P.update(P0); R.OCCASION,R.TIER_IDX=O0,T0
    k=grp if key=="grp" else seg.where(seg.map(seg.value_counts())>=15, grp)
    for c in F.columns:   # oracle: segment x month totals
        s=A[c].groupby(k).sum()/F[c].groupby(k).sum()
        F[c]=F[c]*k.map(s).fillna(1)
    return F
def score(F,tag=""):
    p,a=F.sum(1),A.sum(1)
    e=(p-a).abs()
    print(f"{tag:28s} WAPE {e.sum()/a.sum():.4f} w20 {(e<=0.2*a).mean():.3f} w50 {(e<=0.5*a).mean():.3f} mWAPE {np.mean([(F[c]-A[c]).abs().sum()/A[c].sum() for c in A.columns]):.4f}")
base=run(); score(base,"v10 logic (grp oracle)")
score(run(occ={},tier={}),"no tag/tier idx")
score(run(key="seg"),"v10 seg oracle")
for cap in [0.8,1.5,99]: score(run({"SPIKE_CAP":cap}),f"SPIKE_CAP {cap}")
for r in [(1.3,1.2,1.05),(1.8,1.5,1.2),(2.0,1.6,1.25)]: score(run({"RAMP0":r[0],"RAMP1":r[1],"RAMP2":r[2]}),f"ramp {r}")
for w in [0.2,0.6]: score(run({"W14":w}),f"W14 {w}")
for w in [0.0,0.35]: score(run({"WLONG":w}),f"WLONG {w}")
for pw in [0.05,-0.05]: score(run({"POW":pw}),f"POW {pw}")
# residuals (no tag / tier idx) by attribute: actual / pred, per month and 4-mo
F=run(occ={},tier={})
occ=it.occasion.reindex(items).fillna(""); tr=it.tier.reindex(items).fillna("")
launch=it.launch.reindex(items); age=(B.start_time-launch).dt.days/30.44
def res(mask,name):
    if mask.sum()<4: return
    r=[A.loc[mask,c].sum()/F.loc[mask,c].sum() for c in A.columns]
    print(f"{name:26s} n={mask.sum():4d} "+" ".join(f"{x:.2f}" for x in r)+f" | 4m {A[mask].values.sum()/F[mask].values.sum():.2f}")
print("residual actual/pred (no tag idx, grp oracle): Jan Feb Mar Apr")
for t in ["Halloween","New Year's Eve","Valentine's","Homecoming","Graduation","4th of July","St. Patrick"]:
    res(occ.str.contains(t,regex=False),t)
for t in ["Party","Evening","Basics","Resort-Festival","Casual","Social Daywear","Loungewear"]:
    res(tr.str.contains(t,regex=False)&~occ.str.contains("Halloween|New Year|Valentine"),"tier "+t)
for lo,hi,n in [(-99,0,"launched Dec"),(0,1,"launched Nov"),(1,2,"launched Oct"),(2,4,"2-4m"),(4,8,"4-8m"),(8,99,"8m+")]:
    res((age>lo)&(age<=hi) if lo>-99 else age<=0,n)
sz=pd.qcut(F.sum(1).rank(method="first"),5,labels=False)
for q in range(5): res(sz==q,f"size q{q}")
lm=di[di.m==B].groupby("style_color")[["units","web_units"]].sum().reindex(items).fillna(0)
ws=lm.web_units/lm.units.clip(lower=1)
for lo,hi in [(-0.01,0.5),(0.5,0.8),(0.8,0.95),(0.95,1.01)]: res((ws>lo)&(ws<=hi),f"webshare {lo}-{hi}")
for t in it.product_type.reindex(items).value_counts().index[:22]: res(it.product_type.reindex(items)==t,"type "+t)
