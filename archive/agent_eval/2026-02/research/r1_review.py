# Review: Feb-26 (origin 2026-01, h=1) and older-origin Feb, by segment; 4-month totals by origin
import pandas as pd, numpy as np, sys
sys.path.insert(0, '.')
import rules
it, di, st = rules.load()
sb = pd.read_csv('feedback/scoreboard.csv')
sb = sb.join(it[['grp','product_type','occasion','tier','launch']], on='style_color')
def summ(g):
    return pd.Series({'n': len(g), 'pred': g.pred.sum(), 'act': g.actual.sum(), 'ap': g.actual.sum()/g.pred.sum(),
                      'wape': (g.pred-g.actual).abs().sum()/g.actual.sum()})
f = sb[sb.month=='2026-02']
print(f.groupby('origin').apply(summ).round(3))
f1 = f[f.origin=='2026-01'].copy()
lvl = f1.actual.sum()/f1.pred.sum()
print('level', lvl)
f1['age'] = ((pd.Timestamp('2026-01-01')-f1.launch).dt.days/30.44)
f1['ageb'] = pd.cut(f1.age, [-99,0,1,2,4,12,999])
for c in ['grp','ageb','tier']:
    r = f1.groupby(c).apply(summ); r['ap'] /= lvl; print(r.round(3))
r = f1.groupby('product_type').apply(summ); r['ap'] /= lvl; print(r[r.n>=15].sort_values('ap').round(3))
for tag in ["4th of July","Graduation","Homecoming","Halloween","New Year's Eve","Valentine's","St. Patrick"]:
    g = f1[f1.occasion.fillna('').str.contains(tag, regex=False)]
    print(tag, len(g), round(g.actual.sum()/g.pred.sum()/lvl,3))
# channel shares in base month (Jan)
jan = di[di.m=='2026-01'].groupby('style_color')[['units','web_units','retail','other']].sum()
f1 = f1.join(jan, on='style_color')
f1['rsh'] = f1.retail/f1.units; f1['osh'] = f1.other/f1.units
for c in ['rsh','osh']:
    r = f1.groupby(pd.cut(f1[c],[-0.01,0,0.1,0.25,0.5,1.01])).apply(summ); r['ap']/=lvl; print(c); print(r.round(3))
# 4-month totals by origin
s4 = sb.groupby(['origin','style_color']).agg(pred=('pred','sum'),act=('actual','sum'),nm=('month','nunique')).reset_index()
s4 = s4[s4.nm==4]
o = s4.groupby('origin').apply(lambda g: pd.Series({'wape':(g.pred-g.act).abs().sum()/g.act.sum(),'bias':g.pred.sum()/g.act.sum()-1,
      'w20':((g.pred-g.act).abs()<=0.2*g.act).mean(),'w50':((g.pred-g.act).abs()<=0.5*g.act).mean()}))
print(o.round(3))
