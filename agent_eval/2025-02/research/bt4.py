import pandas as pd, numpy as np, sys
sys.path.insert(0,'research')
import bt2
from bt2 import *
def run2(lbl, c=0, pw=1.0, oracle=False):
    res=[]
    for o in origins:
        P,A=fc2(o,wlong=0.2,ramp=(1.4,1.15),oracle=oracle)
        tot=P.values.sum()
        P2=(P+c*30.4)**1 if pw==1 else P.clip(lower=0)**pw
        P2=P2*tot/P2.values.sum()
        res.append(score(P2,A,scale=True))
    print(lbl.ljust(20),'mean wape %.3f w20 %.3f w50 %.3f'%(np.mean([r['wape'] for r in res]),np.mean([r['w20'] for r in res]),np.mean([r['w50'] for r in res])))
for c in [0,0.5,1,2]: run2(f'add c={c}/day',c=c)
for pw in [0.95,0.9,0.85,0.8]: run2(f'power {pw}',pw=pw)
for pw in [1,0.9,0.85]: run2(f'oracle power {pw}',pw=pw,oracle=True)
