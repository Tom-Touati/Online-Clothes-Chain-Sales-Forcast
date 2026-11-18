"""Split-half backtest of rules.forecast: seasonal/level factors measured on half of the STYLES
(same calendar months, in-sample in time), forecasts scored on the other half. Swap halves, pool."""
import pandas as pd, numpy as np, sys
sys.path.insert(0,'.')
import rules
from rules import *
it,di,st=load()
M=di.pivot_table(index='style_color',columns='m',values='units',aggfunc='sum').fillna(0)
style=pd.Series(it.index.str.split('_').str[0],index=it.index)
halfA=set(it.index[style.map(lambda s: hash(s)%2==0)]) if False else set(it.index[style.map(lambda s: sum(map(ord,s))%2==0)])
def shift(f,k):
    g=dict(f)
    for key in ['seg_rel','grp_rel']: g[key]=f[key].rename(columns=lambda c:c-k)
    for key in ['web','ret']: g[key]=f[key].rename(index=lambda c:c-k)
    return g
def score(p,a):
    e=(p-a).abs()
    return dict(wape=e.sum()/a.sum(),bias=p.sum()/a.sum()-1,w20=(e<=0.2*a).mean(),w50=(e<=0.5*a).mean())
def run(origin, flat=False, **over):
    for k,v in over.items(): setattr(rules,k,v)
    o=pd.Timestamp(origin); B=o.to_period('M'); T=[B+h for h in range(1,5)]
    items=[i for i in it.index if it.launch[i]<B.start_time and i in M.index]
    P=[]; 
    for fit,ev in [(halfA,None),(None,halfA)]:
        src=halfA if fit else set(it.index)-halfA
        tgt=[i for i in items if (i not in src)]
        # factors measured in the target months themselves (B+12 trick then shift back)
        f=rules.seasonal_factors(it,di,st,B+12,[t+12 for t in T],items_for_factors=src,anchor_pre=0.0)
        f=shift(f,12)
        if flat:
            for key in ['seg_rel','grp_rel']: f[key]=f[key]*0+1
        f['web']=f['web']; P.append(rules.forecast(o,tgt,it,di,st,factors=f))
    P=pd.concat(P); A=M.loc[P.index,T]; A.columns=P.columns
    r=score(P.sum(axis=1),A.sum(axis=1)); r['mwape']=(P-A).abs().values.sum()/A.values.sum()
    return r
if __name__=='__main__':
    origins=['2024-02-29','2024-05-31','2024-06-30','2024-07-31','2024-08-31','2024-09-30','2024-10-31']
    for lbl,kw in [('v1',{}),('v1 no type seasonality',{'flat':True})]:
        rs=pd.DataFrame([run(o,**kw) for o in origins],index=origins)
        print(lbl); print(rs.round(3)); print('mean',rs.mean().round(3).to_dict())
