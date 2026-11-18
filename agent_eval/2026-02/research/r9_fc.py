# Forecast summary: totals per month, by group, vs Feb actual of the same items; site share
import pandas as pd, sys
sys.path.insert(0, '.')
import rules
it, di, st = rules.load()
p = pd.read_csv('predictions.csv'); p['grp'] = p.style_color.map(it.grp)
feb = di[di.m == '2026-02'].groupby('style_color').units.sum().reindex(p.style_color.unique()).fillna(0)
print('items', p.style_color.nunique(), 'Feb actual', feb.sum(), 'per day', round(feb.sum()/28), 'zero-pred items', (p.groupby('style_color').pred.sum()==0).sum())
t = p.groupby('month').pred.sum(); days = pd.PeriodIndex(t.index, freq='M').days_in_month
print(pd.DataFrame({'units': t.round(0), 'per_day': (t/days).round(0)}).to_string(), '\n4-month', round(t.sum()))
print(p.pivot_table(index='grp', columns='month', values='pred', aggfunc='sum').round(0).to_string())
print(feb.groupby(it.grp.reindex(feb.index)).sum())
plan = {m: rules.SITE_WEB_PD[m] + rules.OTHER_PD[m] for m in t.index}
print('items / (web+other plan per day + retail ~2.5k)', {m: round(t[m]/d/(plan[m]+2500),3) for m, d in zip(t.index, days)})
