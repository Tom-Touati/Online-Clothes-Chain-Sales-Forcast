# backtest harness: origins Feb..Sep-25 with the scoreboard item lists; per-item totals of complete months <= Oct-25
# oracle site level (SITE_WEB_PD / OTHER_PD hold actuals for past months). usage: python research/bt.py
import pandas as pd, numpy as np, sys, copy
sys.path.insert(0,"."); import rules
it,di,st=rules.load()
mi=pd.read_parquet("data/monthly_item.parquet")
A=mi.pivot_table(index="style_color",columns="month",values="units",aggfunc="sum").fillna(0)
sb=pd.read_csv("feedback/scoreboard.csv")
ORIG=["2025-02","2025-03","2025-04","2025-05","2025-06","2025-07","2025-08","2025-09"]
def run(over=None,orig=ORIG,verbose=False):
    P0=copy.deepcopy(rules.P); rules.P.update(over or {})
    res=[];grp_rel=[]
    try:
        for o in orig:
            items=sb[sb.origin==o].style_color.unique().tolist()
            oe=pd.Period(o,"M").end_time.normalize()
            f=rules.forecast(oe,items,it,di,st)
            ms=[c for c in f.columns if c<="2025-10"]
            p=f[ms].sum(axis=1); a=A.reindex(index=items,columns=ms).fillna(0).sum(axis=1)
            w=(p-a).abs().sum()/a.sum(); b=p.sum()/a.sum()-1
            r=a/p.replace(0,np.nan); h20=((p-a).abs()<=0.2*a).mean(); h50=((p-a).abs()<=0.5*a).mean()
            g=it.grp.reindex(items); wr=(a[g=="WINTER"].sum()/p[g=="WINTER"].sum())/(a.sum()/p.sum())
            # h=1 monthly
            m1=ms[0]; p1=f[m1]; a1=A.reindex(index=items)[m1].fillna(0)
            res.append(dict(o=o,wape=w,bias=b,h20=h20,h50=h50,win=wr,w1=(p1-a1).abs().sum()/a1.sum(),b1=p1.sum()/a1.sum()-1))
    finally:
        rules.P.clear(); rules.P.update(P0)
    R=pd.DataFrame(res).set_index("o")
    return R
if __name__=="__main__":
    R=run(); print(R.round(3).to_string()); print(R.mean().round(4).to_string())
