import pandas as pd, numpy as np, sys
sys.path.insert(0,'research')
from bt2 import *
di2=di.copy()
rows=[]
for o in origins:
    P,A=fc2(o,wlong=0.2)
    om=pd.Timestamp(o).to_period('M')
    x=di2[(di2.m==om)&di2.style_color.isin(P.index)].groupby('style_color')[['web_units','web_disc_sum','units','retail']].sum()
    x['disc']=x.web_disc_sum/x.web_units.replace(0,np.nan)
    p=P.sum(axis=1); a=A.sum(axis=1)
    x=x.reindex(P.index); x['p']=p*a.sum()/p.sum(); x['a']=a; x['o']=o
    # site discount in base month
    rows.append(x)
X=pd.concat(rows)
X['db']=pd.cut(X.disc,[0,0.3,0.45,0.55,0.65,0.75,1])
print(X.groupby('db').agg(n=('a','size'),p=('p','sum'),a=('a','sum')).assign(r=lambda t:t.a/t.p))
X['rel']=X.disc-X.groupby('o').disc.transform('median')
X['rb']=pd.cut(X.rel,[-1,-0.15,-0.05,0.05,0.15,1])
print(X.groupby('rb').agg(n=('a','size'),p=('p','sum'),a=('a','sum')).assign(r=lambda t:t.a/t.p))
X['vb']=pd.qcut(X.p,5)
print(X.groupby('vb').agg(n=('a','size'),p=('p','sum'),a=('a','sum')).assign(r=lambda t:t.a/t.p))
