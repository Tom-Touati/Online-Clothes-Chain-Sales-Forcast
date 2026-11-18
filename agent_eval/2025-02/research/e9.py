import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',200); pd.set_option('display.max_columns',40)
d='data/'
it=pd.read_csv(d+'items.csv',parse_dates=['published_date','first_sale','last_sale']).set_index('style_color')
tf=pd.read_csv(d+'to_forecast.csv').style_color
launch=it[['published_date','first_sale']].min(axis=1)
di=pd.read_parquet(d+'daily_item.parquet'); di['m']=di.date.dt.to_period('M').astype(str)
W=di.pivot_table(index='style_color',columns='m',values='web_units',aggfunc='sum').fillna(0)
T=di.pivot_table(index='style_color',columns='m',values='units',aggfunc='sum').fillna(0)
coh=launch.reindex(W.index)<'2024-01-15'
pt=it.product_type.reindex(W.index)
ms=['2024-03','2024-04','2024-05','2024-06']
tot=W[coh][['2024-02']+ms].sum(); allf=tot[ms]/tot['2024-02']
print('overall web cohort factors', allf.round(3).to_dict(), 'n', coh.sum())
g=W[coh].groupby(pt[coh]).sum()
F=g[ms].div(g['2024-02'],axis=0).div(allf,axis=1)
F['n']=coh.groupby(pt).sum(); F['feb24']=g['2024-02']
tfp=it.loc[tf]
F2=pd.DataFrame({'n_tf':tfp.product_type.value_counts(),'feb25_units':T.loc[tf,'2025-02'].groupby(tfp.product_type).sum()})
out=F2.join(F,how='outer').sort_values('feb25_units',ascending=False)
print(out.round(2).to_string())
