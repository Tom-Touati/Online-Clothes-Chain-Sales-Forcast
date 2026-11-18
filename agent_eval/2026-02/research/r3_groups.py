# Group LFL moves rel. to panel, TY vs LY (monthly), and the factors the rules derive from the Feb-26 base
import pandas as pd, numpy as np, sys
sys.path.insert(0, '.')
import rules
it, di, st = rules.load()
rules.SITE_WEB_PD['2026-02'] = 13827; rules.OTHER_PD['2026-02'] = 4184
def rel(ms):
    U = di[di.m.isin(ms)].pivot_table(index='style_color', columns='m', values='units', aggfunc='sum').reindex(columns=ms).fillna(0)
    g,_ = rules.lfl_chain(U, it.launch, ms, it.grp); a,_ = rules.lfl_chain(U, it.launch, ms, pd.Series('ALL', index=U.index))
    return (g[ms[-1]]/a.loc['ALL', ms[-1]]).round(3), round(a.loc['ALL', ms[-1]],3)
P = pd.Period
for y in [2025, 2026]:
    for a,b in [(P(f'{y-1}-11','M'),P(f'{y-1}-12','M')),(P(f'{y-1}-12','M'),P(f'{y}-01','M')),(P(f'{y}-01','M'),P(f'{y}-02','M'))]:
        r, pa = rel([a,b]); print(a,b, dict(r), 'panel', pa)
for a,b in [('2025-02','2025-03'),('2025-03','2025-04'),('2025-04','2025-05'),('2025-05','2025-06')]:
    r, pa = rel([P(a,'M'),P(b,'M')]); print(a,b, dict(r), 'panel', pa)
B = P('2026-02','M'); targets=[B+h for h in range(1,5)]
f = rules.seasonal_factors(it, di, st, B, targets)
print('seas_pow', f['seas_pow'], 'anc', dict(f['anc'].round(3)))
print('grp_rel'); print(f['grp_rel'].round(3))
print('drift', dict(f['drift'].round(3))); print('web', dict(f['web'].round(3))); print('other', dict(f['other'].round(3))); print('ret', dict(f['ret'].round(3)), f['ret_adj'])
