# Tag / tier residuals vs a segment x month oracle, from a Jan-25 and a Feb-25 base (items launched > 30 days
# before the base month, selling in it).  index = sum actual / sum (base x segment ratio)
import pandas as pd, numpy as np, sys
sys.path.insert(0, '.')
import rules
it, di, st = rules.load()
M = di.groupby(['style_color','m']).units.sum().unstack().fillna(0)
occ = it.occasion.fillna(''); tier = it.tier.fillna('')
tags = list(rules.OCCASION); tiers = list(rules.TIER_IDX)
for base in ['2024-12','2025-12','2026-01']:
    B = pd.Period(base,'M'); tg = [t for t in [B+k for k in range(1, 3)] if t <= pd.Period('2026-02','M')]
    ok = (it.launch.reindex(M.index) < B.start_time - pd.Timedelta(days=30)) & (M[B] > 0)
    X = M[ok]; seg = it.seg.reindex(X.index)
    rows = {}
    for t in tg:
        ratio = X[t].groupby(seg).sum() / X[B].groupby(seg).sum()
        exp = X[B] * seg.map(ratio)
        def idx(mask): return X.loc[mask, t].sum() / exp[mask].sum(), int(mask.sum())
        r = {}
        o, ti = occ.reindex(X.index), tier.reindex(X.index)
        for tag in tags: r[tag] = idx(o.str.contains(tag, regex=False))
        anytag = o.str.contains('|'.join(tags))
        for tr in tiers:
            r['T:'+tr] = idx(ti.str.contains(tr, regex=False) & ~anytag)
        rows[str(t)] = r
    R = pd.DataFrame(rows)
    print('base', base); print(R.map(lambda v: f"{v[0]:.2f} ({v[1]})").to_string())
