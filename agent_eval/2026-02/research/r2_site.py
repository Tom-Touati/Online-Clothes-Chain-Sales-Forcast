# Site / channel per month and Feb aligned YoY; group LFL moves TY vs LY
import pandas as pd, numpy as np, sys
sys.path.insert(0, '.')
import rules
s = pd.read_parquet('data/daily_site.parquet'); s['m'] = s.date.dt.to_period('M')
s['other'] = s.units - s.web_units - s.retail
g = s.groupby('m').agg(days=('date','nunique'), web=('web_units','sum'), oth=('other','sum'), ret=('retail','sum'),
                      wd=('web_discount','mean'), rd=('retail_discount','mean'))
for c in ['web','oth','ret']: g[c] = g[c]/g.days
g['webYoY'] = g.web/g.web.shift(12); g['othYoY'] = g.oth/g.oth.shift(12); g['retYoY'] = g.ret/g.ret.shift(12)
print(g[g.index>='2024-12'].round(3).to_string())
# weekly web Jan-Jun last year and TY Jan-Feb, aligned 364 days
s = s.set_index('date')
for a,b in [('2026-01-25','2026-02-07'),('2026-02-01','2026-02-14'),('2026-02-08','2026-02-21'),('2026-02-15','2026-02-28'),('2026-02-22','2026-02-28')]:
    a,b = pd.Timestamp(a), pd.Timestamp(b)
    ty = s.loc[a:b]; ly = s.loc[a-pd.Timedelta(days=364):b-pd.Timedelta(days=364)]
    print(a.date(), b.date(), 'web TY/d', round(ty.web_units.mean()), 'LY', round(ly.web_units.mean()), 'YoY', round(ty.web_units.mean()/ly.web_units.mean(),3),
          'disc', round(ty.web_discount.mean(),3), round(ly.web_discount.mean(),3), 'oth YoY', round(ty.other.mean()/ly.other.mean(),3), 'oth', round(ty.other.mean()))
# LY weekly web from Feb to end Jun
w = s.loc['2025-02-01':'2025-07-05'].resample('W-SAT').agg({'web_units':'mean','web_discount':'mean','other':'mean','retail':'mean'})
print(w.round(3).to_string())
print(s.loc['2026-02-01':].round(3)[['units','web_units','other','retail','web_discount']].to_string())
ch = pd.read_parquet('data/daily_channel.parquet'); ch['m'] = ch.date.dt.to_period('M')
c = ch.groupby(['m','channel']).units.sum().unstack().fillna(0)
c = c.div(ch.groupby('m').date.nunique(), axis=0)
print(c[c.index>='2025-01'].round(0).T.to_string())
