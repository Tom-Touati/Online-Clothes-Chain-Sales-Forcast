import pandas as pd, numpy as np, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"."); import rules as R
it,di,st=R.load(); items=pd.read_csv("data/to_forecast.csv").style_color.tolist(); o=di.date.max()
base=R.forecast(o,items,it,di,st).sum()
P0=dict(R.P)
def var(**k):
    R.P.update(k); r=R.forecast(o,items,it,di,st).sum(); R.P.clear(); R.P.update(P0); return r
nr=var(RAMP0_NEW=1.0,RAMP1=1.0,RAMP2=1.0,RAMP_H2=1.0,AGE_FADE=1.0)
O0,T0=R.OCCASION,R.TIER_IDX; R.OCCASION,R.TIER_IDX={}, {}
no=R.forecast(o,items,it,di,st).sum(); R.OCCASION,R.TIER_IDX=O0,T0
print(pd.DataFrame({"v11":base,"no ramp/fade":nr,"no occ/tier":no}).round(0))
launch=it.launch.reindex(items); B=pd.Period("2025-12"); age=(B.start_time-launch).dt.days/30.44
dec=di[di.m==B].groupby("style_color").units.sum().reindex(items).fillna(0)
print(pd.cut(age,[-99,0,1,2,4,99]).value_counts().sort_index(), dec.groupby(pd.cut(age,[-99,0,1,2,4,99])).sum())
# LY: share of Jan-25 site units from items live before Dec-24 and selling in Dec-24
for a,b in [("2024-12","2025-01"),("2024-12","2025-02"),("2024-12","2025-04"),("2025-11","2025-12")]:
    a,b=pd.Period(a),pd.Period(b)
    sold=di[(di.m==a)&(di.units>0)].style_color.unique()
    sh=di[(di.m==b)&di.style_color.isin(sold)].units.sum()/di[di.m==b].units.sum()
    print(a,b,"share of target-month units from base-month sellers",round(sh,3))
