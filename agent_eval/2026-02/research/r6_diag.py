# h=1 segment act/pred (level-neutral) over the last origins; share of site units from young items TY vs LY
import pandas as pd, numpy as np, sys
sys.path.insert(0, '.')
import rules
it, di, st = rules.load()
sb = pd.read_csv('feedback/scoreboard.csv'); sb = sb[sb.h==1].join(it[['grp','product_type','tier']], on='style_color')
out = {}
for o, g in sb[sb.origin>='2025-08'].groupby('origin'):
    lvl = g.actual.sum()/g.pred.sum()
    r = g.groupby('product_type').apply(lambda x: x.actual.sum()/x.pred.sum()/lvl, include_groups=False)
    r2 = g.groupby('grp').apply(lambda x: x.actual.sum()/x.pred.sum()/lvl, include_groups=False)
    out[o] = pd.concat([r2, r[['Jeans','Pants','Hoodie','Sweatpants','Dress','Top','Tank Top','Shorts','Skirt','Sweater']]])
print(pd.DataFrame(out).round(2).to_string())
# young-item share of units by month
d = di.groupby(['style_color','m']).units.sum().reset_index()
d['age'] = (d.m.dt.start_time - it.launch.reindex(d.style_color).values).dt.days
s = d.assign(y60=d.units.where(d.age < 60, 0), y120=d.units.where(d.age<120,0)).groupby('m')[['units','y60','y120']].sum()
s['sh60'] = s.y60/s.units; s['sh120'] = s.y120/s.units
print(s[s.index>='2024-11'][['sh60','sh120']].round(3).T.to_string())
n = it.launch.dt.to_period('M').value_counts().sort_index(); print(n[n.index>='2024-10'].to_string())
