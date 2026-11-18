import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
d='data/'
it=pd.read_csv(d+'items.csv',parse_dates=['published_date','first_sale','last_sale']).set_index('style_color')
launch=it[['published_date','first_sale']].min(axis=1)
di=pd.read_parquet(d+'daily_item.parquet')
di['m']=di.date.dt.to_period('M')
M=di.pivot_table(index='style_color',columns='m',values='units',aggfunc='sum').fillna(0)
D=di.pivot_table(index='style_color',columns='date',values='units',aggfunc='sum').fillna(0)
def fc(origin, base='m1', ramp=(1.0,1.0), m1w=None):
    o=pd.Timestamp(origin); om=o.to_period('M')
    items=launch[launch<om.start_time].index.intersection(M.index)
    Dd=D.loc[items, :o]
    L=launch[items]
    days_live=((o-L).dt.days+1).clip(lower=1)
    def rate(n):  # units/day over last n days (only days live)
        span=np.minimum(days_live,n)
        return Dd.iloc[:,-n:].sum(axis=1)/span
    if base=='m1': b=M.loc[items,om]/om.days_in_month
    elif base=='m2': b=(M.loc[items,om]+M.loc[items,om-1])/np.minimum(days_live,om.days_in_month+(om-1).days_in_month)
    elif base=='d14': b=rate(14)
    elif base=='d28': b=rate(28)
    elif base=='blend': b=0.5*rate(28)+0.3*rate(56)+0.2*rate(14)
    elif base=='blend2': b=0.6*(M.loc[items,om]/om.days_in_month)+0.4*rate(14)
    # ramp for young: age at origin month start, months
    age=(om.start_time-L).dt.days/30.44
    out={}
    for h in range(1,5):
        tm=om+h
        f=np.ones(len(items))
        # first full month = om for items with age<1 ; ramp[0] applied from h>=1, ramp[1] additionally from h>=2
        y1=(age<1).values; y2=((age>=1)&(age<2)).values
        f=np.where(y1, ramp[0]*(ramp[1] if h>=2 else 1), f)
        f=np.where(y2, ramp[1], f)
        out[str(tm)]=b.values*tm.days_in_month*f
    P=pd.DataFrame(out,index=items)
    A=M.loc[items,[om+h for h in range(1,5)]]; A.columns=P.columns
    return P,A
def score(P,A,scale=False):
    p=P.sum(axis=1); a=A.sum(axis=1)
    if scale: p=p*a.sum()/p.sum()
    wape=(p-a).abs().sum()/a.sum(); bias=p.sum()/a.sum()-1
    r=p/a.replace(0,np.nan)
    w20=((p-a).abs()<=0.2*a).mean(); w50=((p-a).abs()<=0.5*a).mean()
    mw=(P.mul(p/P.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)-A).abs().values.sum()/A.values.sum() if scale else (P-A).abs().values.sum()/A.values.sum()
    return dict(wape=round(wape,3),bias=round(bias,3),w20=round(w20,3),w50=round(w50,3),mwape=round(mw,3))
if __name__=='__main__':
    origins=['2024-05-31','2024-06-30','2024-07-31','2024-08-31','2024-09-30','2024-10-31']
    for base in ['m1','m2','d14','d28','blend','blend2']:
        res=[score(*fc(o,base),scale=True) for o in origins]
        print(base.ljust(7),'scaled wape by origin',[r['wape'] for r in res],'mean',round(np.mean([r['wape'] for r in res]),3),'w50',round(np.mean([r['w50'] for r in res]),3))
    for ramp in [(1,1),(1.2,1.0),(1.3,1.1),(1.5,1.2),(1.5,1.0)]:
        res=[score(*fc(o,'m1',ramp),scale=True) for o in origins]
        print(ramp,'scaled wape',[r['wape'] for r in res],'mean',round(np.mean([r['wape'] for r in res]),3))
