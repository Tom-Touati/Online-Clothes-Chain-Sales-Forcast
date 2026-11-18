"""Out-of-time check: origin 2025-01-31, factors from Jan->Feb 2024 (truly past), score Feb 2025 (1 month)."""
import pandas as pd, numpy as np, sys
sys.path.insert(0,'.')
import rules
from rules import *
it,di,st=load()
o=pd.Timestamp('2025-01-31'); B=o.to_period('M')
items=[i for i in it.index if it.launch[i]<B.start_time and i in set(di.style_color)]
A=di[(di.m==B+1)].groupby('style_color').units.sum().reindex(items).fillna(0)
d=di[di.date<=o]
f=seasonal_factors(it,d,st[st.date<=o],B,[B+h for h in range(1,5)])
def sc(p,a,lbl):
    e=(p-a).abs(); print(lbl.ljust(34),'wape %.3f bias %+.3f w20 %.3f w50 %.3f'%(e.sum()/a.sum(),p.sum()/a.sum()-1,(e<=.2*a).mean(),(e<=.5*a).mean()))
P=forecast(o,items,it,di,st,factors=f)['2025-02']
sc(P,A,'rules (Jan24->Feb24 factors)'); sc(P*A.sum()/P.sum(),A,'  same, level-scaled')
g=dict(f); g['seg_rel']=f['seg_rel']*0+1; g['grp_rel']=f['grp_rel']*0+1
P0=forecast(o,items,it,di,st,factors=g)['2025-02']
sc(P0*A.sum()/P0.sum(),A,'no type seasonality, level-scaled')
print('web level used',f['web'].round(3).to_dict())
for r in [(1,1),(1.25,1.1),(1.5,1.2)]:
    rules.RAMP1,rules.RAMP2=r
    P=forecast(o,items,it,di,st,factors=f)['2025-02']; sc(P*A.sum()/P.sum(),A,f'ramp {r}, level-scaled')
rules.RAMP1,rules.RAMP2=1.25,1.1
for k in [2,5,10,20]:
    rules.SHRINK_K=k; P=forecast(o,items,it,di,st,factors=f)['2025-02']; sc(P*A.sum()/P.sum(),A,f'shrink K={k}, level-scaled')
rules.SHRINK_K=5
for w in [(0.4,0.2),(0.2,0.2),(0.4,0.0),(0.6,0.1)]:
    rules.W14,rules.WLONG=w; P=forecast(o,items,it,di,st,factors=f)['2025-02']; sc(P*A.sum()/P.sum(),A,f'W14,WLONG={w}, level-scaled')
