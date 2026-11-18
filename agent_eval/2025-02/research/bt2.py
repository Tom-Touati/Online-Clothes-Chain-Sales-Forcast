import pandas as pd, numpy as np, sys
sys.path.insert(0,'research')
from bt import *
def fc2(origin, w14=0.4, wlong=0.0, ramp=(1.5,1.2), oracle=False, minfloor=0):
    o=pd.Timestamp(origin); om=o.to_period('M')
    items=launch[launch<om.start_time].index.intersection(M.index)
    Dd=D.loc[items,:o]; L=launch[items]
    live=((o-L).dt.days+1).clip(lower=1)
    rate=lambda n: Dd.iloc[:,-n:].sum(axis=1)/np.minimum(live,n)
    rm=M.loc[items,om]/om.days_in_month
    b=(1-w14)*rm+w14*rate(14)
    b=b*rm.sum()/b.sum()
    if wlong>0:
        bl=rate(91); bl=bl*rm.sum()/bl.sum()
        b=(1-wlong)*b+wlong*bl
    age=(om.start_time-L).dt.days/30.44
    y1=(age<1).values; y2=((age>=1)&(age<2)).values
    out={}
    for h in range(1,5):
        tm=om+h
        f=np.where(y1, ramp[0]*(ramp[1] if h>=2 else 1), np.where(y2, ramp[1],1.0))
        out[str(tm)]=b.values*tm.days_in_month*f
    P=pd.DataFrame(out,index=items)
    A=M.loc[items,[om+h for h in range(1,5)]]; A.columns=P.columns
    if oracle:  # leave-one-out type factor on 4-month totals
        pt=it.product_type.reindex(items)
        p=P.sum(axis=1); a=A.sum(axis=1)
        gp=p.groupby(pt).transform('sum')-p; ga=a.groupby(pt).transform('sum')-a
        n=pt.groupby(pt).transform('size')
        ov=a.sum()/p.sum()
        fac=np.where(n>=4, (ga+5*ov*gp.mean()/gp.mean())/(gp+5), ov)  # crude
        fac=np.where(n>=4,(ga/gp.replace(0,np.nan)).fillna(ov),ov)
        fac=(n*fac+3*ov)/(n+3)
        P=P.mul(fac,axis=0)
    return P,A
origins=['2024-05-31','2024-06-30','2024-07-31','2024-08-31','2024-09-30','2024-10-31']
def run(lbl,**kw):
    res=[score(*fc2(o,**kw),scale=True) for o in origins]
    print(lbl.ljust(28),'wape',[r['wape'] for r in res],'mean %.3f w20 %.3f w50 %.3f mwape %.3f'%(np.mean([r['wape'] for r in res]),np.mean([r['w20'] for r in res]),np.mean([r['w50'] for r in res]),np.mean([r['mwape'] for r in res])))
run('base m1 noramp',w14=0,ramp=(1,1))
run('w14=.4 ramp1.5/1.2')
run('w14=.4 ramp1.3/1.1',ramp=(1.3,1.1))
run('w14=.4 ramp1.7/1.3',ramp=(1.7,1.3))
run('w14=.6',w14=0.6)
run('w14=.4 wlong=.2',wlong=0.2)
run('w14=.4 wlong=.4',wlong=0.4)
run('oracle type LOO',oracle=True)
