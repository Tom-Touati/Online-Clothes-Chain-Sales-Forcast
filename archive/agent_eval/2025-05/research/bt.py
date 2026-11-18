"""Honest backtest: origins Jan..Apr-25, data up to origin, score per-item totals over complete months <= May-25."""
import sys, copy; sys.path.insert(0,".")
import pandas as pd, numpy as np, rules as R
it,di,st=R.load()
sb=pd.read_csv("feedback/scoreboard.csv")
mi=pd.read_parquet("data/monthly_item.parquet")
A=mi.pivot_table(index="style_color",columns="month",values="units",aggfunc="sum").fillna(0)
ORIG=["2025-01-31","2025-02-28","2025-03-31","2025-04-30"]
def items_for(o):
    key=o[:7]
    s=sb[sb.origin==key].style_color.unique()
    if len(s): return list(s)
    d=di[(di.date<=o)&(di.date>pd.Timestamp(o)-pd.Timedelta(days=60))]
    return list(d.groupby("style_color").units.sum().loc[lambda x:x>0].index)
def run(name, p=None, occ=None, site=None):
    P0=dict(R.P); O0=dict(R.OCCASION); S0=dict(R.SITE_WEB_IDX)
    if p: R.P.update(p)
    if occ is not None: R.OCCASION.clear(); R.OCCASION.update(occ)
    if site is not None: R.SITE_WEB_IDX.clear(); R.SITE_WEB_IDX.update(site)
    res=[]
    for o in ORIG:
        items=items_for(o); F=R.forecast(o,items,it,di,st)
        cols=[c for c in F.columns if c<="2025-05"]
        pt=F[cols].sum(axis=1); at=A.reindex(items)[cols].fillna(0).sum(axis=1)
        w=(pt-at).abs().sum()/at.sum(); b=pt.sum()/at.sum()-1
        m=at>0; i50=((pt-at).abs()[m]<=0.5*at[m]).mean(); i20=((pt-at).abs()[m]<=0.2*at[m]).mean()
        res.append((w,b,i20,i50))
    R.P.clear(); R.P.update(P0); R.OCCASION.clear(); R.OCCASION.update(O0); R.SITE_WEB_IDX.clear(); R.SITE_WEB_IDX.update(S0)
    r=np.array(res)
    print(f"{name:28s} meanWAPE {r[:,0].mean():.4f} | "+" ".join(f"{w:.3f}/{b:+.0%}" for w,b,_,_ in res)+f" | in20 {r[:,2].mean():.3f} in50 {r[:,3].mean():.3f}")
if __name__=="__main__":
    J4={5:1.15,6:1.3,7:1.2,8:1.15,9:1.1}
    OCC={"4th of July":J4,"Graduation":{4:1.1,5:1.2,6:1.0,7:1.0,8:0.95,9:0.95},
         "Homecoming":{5:1.2,6:0.95,7:1.05,8:1.3,9:1.9,10:1.9},"Halloween":{9:1.15,10:1.6}}
    run("v3")
    run("occ new",occ=OCC)
    run("anchor .5",{"SEAS_ANCHOR":0.5})
    run("anchor 1",{"SEAS_ANCHOR":1.0})
    run("other .95",{"OTHER_MONTHLY":0.95})
    run("other 1.0",{"OTHER_MONTHLY":1.0})
    run("ret pow .5",{"RET_POW":0.5})
    run("wdefl .5",{"W_DEFL":0.5})
    run("wdefl 1",{"W_DEFL":1.0})
    run("pow .05",{"POW":0.05})
    run("pow .1",{"POW":0.1})
    run("W14 .6",{"W14":0.6})
    run("ramp0 2.0",{"RAMP0":2.0})
