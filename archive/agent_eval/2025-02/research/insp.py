import pandas as pd, numpy as np, sys
sys.path.insert(0,'.')
from rules import *
it,di,st=load()
B=pd.Period('2025-02'); T=[B+h for h in range(1,5)]
f=seasonal_factors(it,di,st,B,T)
print('web',f['web'].round(3).to_dict()); print('ret',f['ret'].round(3).to_dict())
print(f['grp_rel'].round(2)); 
x=f['seg_rel'].round(2); x['n']=f['seg_n']; print(x.to_string())
print(it.grp.value_counts())
