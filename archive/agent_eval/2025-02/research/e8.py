import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',200); pd.set_option('display.max_columns',40)
d='data/'
it=pd.read_csv(d+'items.csv',parse_dates=['published_date','first_sale','last_sale']).set_index('style_color')
launch=it[['published_date','first_sale']].min(axis=1)
m=pd.read_parquet(d+'monthly_item.parquet')
M=m.pivot_table(index='style_color',columns='month',values='units',aggfunc='sum').fillna(0)
occ=it.occasion.fillna('').str.split('; ')
tags=['Valentine\'s Day','St. Patrick\'s Day','Graduation','4th of July','Halloween','New Year\'s Eve','Homecoming']
L=launch.reindex(M.index)
# monthly LFL ratio (items live >=30 days before prev month start, selling prev month) per tag vs all
def chain(mask):
    ms=M.columns.tolist(); out=[]
    for a,b in zip(ms[:-1],ms[1:]):
        ok=mask&(L<pd.Period(a).start_time-pd.Timedelta(days=30))&(M[a]>0)
        out.append((M.loc[ok,b].sum()/M.loc[ok,a].sum()) if ok.sum()>=2 else np.nan)
    return pd.Series(out,index=ms[1:])
rows={'ALL':chain(pd.Series(True,index=M.index))}
for t in tags:
    mk=occ.reindex(M.index).apply(lambda x: t in x)
    rows[t+f' n={mk.sum()}']=chain(mk)
R=pd.DataFrame(rows).T
print(R.round(2))
print('relative to ALL'); print((R/R.loc['ALL']).round(2))
# monthly share profile for tagged items (raw units)
for t in tags:
    mk=occ.reindex(M.index).apply(lambda x: t in x)
    print(t, M[mk].sum().astype(int).to_dict())
