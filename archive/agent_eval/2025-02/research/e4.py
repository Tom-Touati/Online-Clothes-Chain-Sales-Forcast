import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',200)
d='data/'
it=pd.read_csv(d+'items.csv',parse_dates=['published_date','first_sale','last_sale']).set_index('style_color')
di=pd.read_parquet(d+'daily_item.parquet')
# launch date = earliest of published / first_sale (first_sale censored at 2024-01-01)
launch=it[['published_date','first_sale']].min(axis=1)
m=pd.read_parquet(d+'monthly_item.parquet')
P=m.pivot_table(index='style_color',columns='month',values='units',aggfunc='sum').fillna(0)
months=P.columns.tolist()
L=launch.reindex(P.index)
def agem(mo):  # age in months at start of month
    st=pd.Period(mo).start_time
    return ((st-L).dt.days/30.44)
rows=[]
for a,b in zip(months[:-1],months[1:]):
    ag=agem(a)  # age at start of month a; item must be fully live in a
    for lo,hi in [(0,1),(1,2),(2,3),(3,4),(4,6),(6,9),(9,12),(12,99)]:
        ok=(ag>=lo)&(ag<hi)&(P[a]>0)
        if ok.sum()>=3:
            rows.append((b,f'{lo}-{hi}',ok.sum(),P.loc[ok,b].sum()/P.loc[ok,a].sum()))
r=pd.DataFrame(rows,columns=['month','age','n','ratio'])
R=r.pivot(index='month',columns='age',values='ratio')[['0-1','1-2','2-3','3-4','4-6','6-9','9-12','12-99']]
print(R.round(2)); print(r.pivot(index='month',columns='age',values='n'))
print('relative to 12-99 group:'); print(R.div(R['12-99'],axis=0).round(2)); print(R.div(R['12-99'],axis=0).median().round(2))
