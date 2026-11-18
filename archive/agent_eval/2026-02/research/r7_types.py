# Oracle-site backtest residuals by product type (level-neutral act/pred of all available months) per origin
import pandas as pd, numpy as np, sys
sys.path.insert(0, '.')
import rules
it, di, st = rules.load()
mi = pd.read_parquet('data/monthly_item.parquet').set_index(['style_color','month']).units
sb = pd.read_csv('feedback/scoreboard.csv'); lists = sb.groupby('origin').style_color.unique()
res = {}; resh1 = {}
for o in lists.index:
    B = pd.Period(o,'M'); items = list(lists[o])
    F = rules.forecast(B.end_time.normalize(), items, it, di, st)
    cols = [c for c in F.columns if c <= '2026-02']
    A = pd.DataFrame({c: mi.reindex(pd.MultiIndex.from_product([items,[c]])).fillna(0).values for c in cols}, index=items)
    for R, cc in [(res, cols), (resh1, cols[:1])]:
        p, a = F[cc].sum(axis=1), A[cc].sum(axis=1); lvl = a.sum()/p.sum()
        t = it.product_type.reindex(items).fillna('NA'); g = it.grp.reindex(items)
        R[o] = pd.concat([(a.groupby(g).sum()/p.groupby(g).sum()/lvl), (a.groupby(t).sum()/p.groupby(t).sum()/lvl)])
for name, R in [('all months', res), ('h=1', resh1)]:
    D = pd.DataFrame(R)
    keep = ['CORE','SUMMER','WINTER','Skirt','Pants','Jeans','Dress','Top','Tank Top','Shorts','Skort','Hoodie','Sweatpants','Sweater','T-Shirt','Corset']
    print(name); print(D.reindex(keep).round(2).to_string())
