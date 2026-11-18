"""Full-rules backtest on 2025 origins (scoreboard item lists, site plan dict = actuals for past months)."""
import pandas as pd, numpy as np, sys, warnings, time; warnings.filterwarnings("ignore")
sys.path.insert(0,"."); import rules as R
it,di,st=R.load()
sb=pd.read_csv("feedback/scoreboard.csv")
act=di.groupby(["style_color","m"]).units.sum()
last=di.m.max()
V={"v10-like":dict(RAMP0_NEW=1.6,POW_H=(0,0,0,0),AGE_FADE=1.0,RET_MATURE=False),
   "v11":{},
   "v11 no POW_H":dict(POW_H=(0,0,0,0)),
   "v11 no AGE_FADE":dict(AGE_FADE=1.0),
   "v11 RAMP0 1.6":dict(RAMP0_NEW=1.6),
   "v11 no RET_MATURE":dict(RET_MATURE=False),
   "v11 POW_H x1.5":dict(POW_H=(0,0.045,0.09,0.12)),
   "v11 AGE_FADE .9":dict(AGE_FADE=0.9)}
P0=dict(R.P); rows=[]
for org in sorted(sb.origin.unique()):
    o=pd.Period(org,"M").end_time.normalize(); B=pd.Period(org,"M")
    items=sb[sb.origin==org].style_color.unique().tolist()
    T=[B+h for h in range(1,5) if B+h<=last]
    A=pd.DataFrame({str(t):[act.get((i,t),0) for i in items] for t in T},index=items)
    for name,p in V.items():
        R.P.clear(); R.P.update(P0); R.P.update(p)
        F=R.forecast(o,items,it,di,st)[[str(t) for t in T]]
        pt,at=F.sum(1),A.sum(1); e=(pt-at).abs()
        rows.append(dict(origin=org,v=name,nm=len(T),wape=e.sum()/at.sum(),bias=pt.sum()/at.sum()-1,
                         w20=(e<=0.2*at).mean(),w50=(e<=0.5*at).mean(),
                         mw=np.mean([(F[c]-A[c]).abs().sum()/A[c].sum() for c in A.columns])))
R.P.clear(); R.P.update(P0)
d=pd.DataFrame(rows)
print(d.pivot(index="origin",columns="v",values="wape").round(4).to_string())
full=d[d.nm==4]
print("\n4-month origins (Feb-Aug-25) mean:"); print(full.groupby("v")[["wape","bias","w20","w50","mw"]].mean().round(4).to_string())
print("\nall origins mean (Sep-Nov partial):"); print(d.groupby("v")[["wape","bias","w20","w50","mw"]].mean().round(4).to_string())
