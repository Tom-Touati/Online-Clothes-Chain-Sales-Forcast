import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',200)
d='data/'
it=pd.read_csv(d+'items.csv',parse_dates=['published_date','first_sale','last_sale'])
tf=pd.read_csv(d+'to_forecast.csv')
m=pd.read_parquet(d+'monthly_item.parquet')
print('tf not in items', set(tf.style_color)-set(it.style_color))
x=it[~it.style_color.isin(tf.style_color)]; print(x[['style_color','product_type','published_date','first_sale','last_sale']])
t=it[it.style_color.isin(tf.style_color)]
print(t.last_sale.value_counts().sort_index().tail(10)); print((t.last_sale<'2025-02-01').sum())
print('first sale vs pub', (t.first_sale-t.published_date).dt.days.describe())
# like-for-like chain index: items with units>0 in both months and published >= 60d before month start of prev month
P=m.pivot_table(index='style_color',columns='month',values='units',aggfunc='sum').fillna(0)
months=P.columns.tolist()
pub=it.set_index('style_color').published_date
rows=[]
for a,b in zip(months[:-1],months[1:]):
    st=pd.Period(a).start_time
    ok=pub.reindex(P.index)<st-pd.Timedelta(days=90)
    ok&=(P[a]>0)
    rows.append((b, ok.sum(), P.loc[ok,b].sum()/P.loc[ok,a].sum()))
r=pd.DataFrame(rows,columns=['month','n','ratio']); r['chain']=r.ratio.cumprod(); print(r)
# site daily total
s=pd.read_parquet(d+'daily_site.parquet'); s['m']=s.date.dt.to_period('M')
print(s.groupby('m')[['units','online','retail','drop','web_units','cancelled_units']].sum())
print(s.groupby('m')[['web_discount','retail_discount']].mean())
