import pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
s=pd.read_parquet('data/daily_site.parquet').set_index('date')
s['dow']=s.index.day_name().str[:3]
print(s.loc['2025-01-20':,['dow','units','web_units','retail','drop','web_discount','retail_discount']].round(3).to_string())
print(s.loc['2024-02-01':'2024-03-20',['dow','units','web_units','retail','drop','web_discount']].round(3).to_string())
