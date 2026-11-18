import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',200)
d='data/'
it=pd.read_csv(d+'items.csv',parse_dates=['published_date','first_sale','last_sale']).set_index('style_color')
launch=it[['published_date','first_sale']].min(axis=1)
di=pd.read_parquet(d+'daily_item.parquet'); di['m']=di.date.dt.to_period('M').astype(str)
di['oth']=di['units']-di['web_units']-di['retail']-di['drop']  # tiktok etc
L=launch
for col in ['units','web_units','retail','drop','oth']:
    X=di.pivot_table(index='style_color',columns='m',values=col,aggfunc='sum').fillna(0)
    T=di.pivot_table(index='style_color',columns='m',values='units',aggfunc='sum').fillna(0)
    ms=X.columns.tolist(); out=[]
    for a,b in zip(ms[:-1],ms[1:]):
        ok=(L.reindex(X.index)<pd.Period(a).start_time-pd.Timedelta(days=60))&(T[a]>0)
        out.append(X.loc[ok,b].sum()/max(X.loc[ok,a].sum(),1))
    print(col.ljust(9),' '.join(f'{b[2:]}:{v:.2f}' for b,v in zip(ms[1:],out)))
st=pd.read_parquet(d+'daily_store.parquet'); st['m']=st.date.dt.to_period('M')
print(st.pivot_table(index='m',columns='store',values='units',aggfunc='sum').fillna(0).astype(int))
# per-day by store per month
print((st.groupby(['m','store']).units.sum()/st.groupby(['m','store']).date.nunique()).unstack().round(0))
