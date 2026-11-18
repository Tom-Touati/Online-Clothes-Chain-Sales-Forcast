import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',200); pd.set_option('display.max_columns',40)
d='data/'
it=pd.read_csv(d+'items.csv',parse_dates=['published_date','first_sale','last_sale']).set_index('style_color')
at=pd.read_csv(d+'item_attributes.csv').set_index('style_color')
tf=pd.read_csv(d+'to_forecast.csv').style_color
launch=it[['published_date','first_sale']].min(axis=1)
di=pd.read_parquet(d+'daily_item.parquet'); di['m']=di.date.dt.to_period('M').astype(str)
X=di.pivot_table(index='style_color',columns='m',values='units',aggfunc='sum').fillna(0)
L=launch.reindex(X.index)
seq=['2024-02','2024-03','2024-04','2024-05','2024-06']
def chain(key):
    res={}
    for a,b in zip(seq[:-1],seq[1:]):
        ok=(L<pd.Period(a).start_time-pd.Timedelta(days=30))&(X[a]>0)
        g=X[ok].groupby(key.reindex(X.index)[ok])
        res[b]=g[b].sum()/g[a].sum()
        res['n'+b]=g.size()
    R=pd.DataFrame(res)
    C=R[[c for c in R if not c.startswith('n')]].cumprod(axis=1)
    C['n_mar']=R['n2024-03']; C['n_jun']=R['n2024-06']
    return C
allc=chain(pd.Series('ALL',index=X.index)); print(allc.round(3))
base=allc.iloc[0,:4]
for col,src in [('sleeve',at),('fabric_weight',at),('length',at)]:
    C=chain(src[col].fillna('NA'))
    C.iloc[:,:4]=C.iloc[:,:4].div(base.values,axis=1)
    C['n_tf']=src[col].reindex(tf).fillna('NA').value_counts()
    C['feb25']=X.loc[tf,'2025-02'].groupby(src[col].reindex(tf).fillna('NA').values).sum()
    print(C.sort_values('feb25',ascending=False).round(2).to_string())
