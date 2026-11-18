import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',200)
d='data/'
it=pd.read_csv(d+'items.csv',parse_dates=['published_date','first_sale','last_sale'])
s=pd.read_parquet(d+'daily_site.parquet').set_index('date')
w=s[['units','online','retail','drop','web_units','cancelled_units']].resample('W-SUN').sum()
w['disc']=s.web_discount.resample('W-SUN').mean()
print(w.round(3).to_string())
ch=pd.read_parquet(d+'daily_channel.parquet'); ch['m']=ch.date.dt.to_period('M')
print(ch.pivot_table(index='m',columns='channel',values='units',aggfunc='sum').fillna(0).astype(int))
