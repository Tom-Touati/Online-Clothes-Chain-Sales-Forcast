import pandas as pd, numpy as np, sys
sys.path.insert(0,'research'); sys.path.insert(0,'.')
from bt_rules import *
import rules
origins=['2024-02-29','2024-05-31','2024-06-30','2024-07-31','2024-08-31','2024-09-30','2024-10-31']
default=dict(W14=0.4,WLONG=0.2,RAMP1=1.4,RAMP2=1.15,SHRINK_K=5,CAP=(0.15,4.0),CH_K=20)
variants={'v1':{}, 'K=2':{'SHRINK_K':2}, 'K=10':{'SHRINK_K':10}, 'ramp 1.2/1.1':{'RAMP1':1.2,'RAMP2':1.1},
 'ramp 1.6/1.25':{'RAMP1':1.6,'RAMP2':1.25}, 'W14=.2':{'W14':0.2}, 'WLONG=0':{'WLONG':0.0}, 'WLONG=.35':{'WLONG':0.35},
 'cap 0.1-6':{'CAP':(0.1,6.0)}}
for nm,v in variants.items():
    for k,val in {**default,**v}.items(): setattr(rules,k,val)
    rs=pd.DataFrame([run(o) for o in origins],index=origins)
    print(nm.ljust(14),'feb-origin wape %.3f | mean wape %.3f bias %+.3f w20 %.3f w50 %.3f mwape %.3f'%(rs.wape.iloc[0],rs.wape.mean(),rs.bias.mean(),rs.w20.mean(),rs.w50.mean(),rs.mwape.mean()))
