import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',200)
d='data/'
it=pd.read_csv(d+'items.csv',parse_dates=['published_date','first_sale','last_sale']).set_index('style_color')
launch=it[['published_date','first_sale']].min(axis=1)
di=pd.read_parquet(d+'daily_item.parquet'); di['m']=di.date.dt.to_period('M').astype(str)
W=di.pivot_table(index='style_color',columns='m',values='web_units',aggfunc='sum').fillna(0)
T=di.pivot_table(index='style_color',columns='m',values='units',aggfunc='sum').fillna(0)
old=launch.reindex(W.index)<'2024-01-01'
print('items live before 2024:',old.sum())
for nm,X in [('web',W),('total',T)]:
    o=X[old]; base=o['2024-02']
    print(nm,'old cohort index vs Feb24:', (o.sum()/base.sum()).round(2).to_dict())
# product type groups
it['pt']=it.product_type
g=it.pt.reindex(W.index)[old]
o=W[old]
tab=o.groupby(g).sum()
res=pd.DataFrame({'n':o.groupby(g).size(),'feb':tab['2024-02']})
for mo in ['2024-01','2024-03','2024-04','2024-05','2024-06','2024-07','2024-08','2025-01','2025-02']:
    res[mo]=(tab[mo]/tab['2024-02']).round(2)
print(res.sort_values('feb',ascending=False))
# YoY same items
both=(T['2024-02']>0)&(T['2025-02']>0)
print('YoY Feb items',both.sum(), T.loc[both,'2025-02'].sum()/T.loc[both,'2024-02'].sum(), 'web', W.loc[both,'2025-02'].sum()/W.loc[both,'2024-02'].sum())
both=(T['2024-01']>0)&(T['2025-01']>0)
print('YoY Jan items',both.sum(), T.loc[both,'2025-01'].sum()/T.loc[both,'2024-01'].sum())
