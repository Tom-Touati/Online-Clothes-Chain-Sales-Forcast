# Oracle-site backtest of rules.py variants: origins Feb-25..Jan-26 with the scoreboard item lists,
# site plan dicts = actuals for past months.  usage: python research/r4_bt.py  (variants below)
import pandas as pd, numpy as np, sys, copy
sys.path.insert(0, '.')
import rules
it, di, st = rules.load()
mi = pd.read_parquet('data/monthly_item.parquet').set_index(['style_color','month']).units
sb = pd.read_csv('feedback/scoreboard.csv')
lists = sb.groupby('origin').style_color.unique()
P0, OCC0, TIER0 = copy.deepcopy(rules.P), copy.deepcopy(rules.OCCASION), copy.deepcopy(rules.TIER_IDX)
def run(name, over=None, mod=None, origins=None):
    rules.P = copy.deepcopy(P0); rules.P.update(over or {})
    rules.OCCASION, rules.TIER_IDX = copy.deepcopy(OCC0), copy.deepcopy(TIER0)
    if mod: mod(rules)
    rows, h1 = [], []
    for o in (origins or lists.index):
        B = pd.Period(o, 'M'); oe = B.end_time.normalize()
        items = list(lists[o])
        F = rules.forecast(oe, items, it, di, st)
        A = pd.DataFrame({c: mi.reindex(pd.MultiIndex.from_product([items,[c]])).fillna(0).values for c in F.columns}, index=items)
        avail = [c for c in F.columns if c <= '2026-02']
        e1 = (F[F.columns[0]]-A[F.columns[0]]).abs().sum()/A[F.columns[0]].sum()
        r = dict(origin=o, h1=e1, h1bias=F[F.columns[0]].sum()/A[F.columns[0]].sum()-1)
        if len(avail) == 4:
            p, a = F.sum(axis=1), A.sum(axis=1)
            r.update(wape=(p-a).abs().sum()/a.sum(), bias=p.sum()/a.sum()-1,
                     w20=((p-a).abs()<=0.2*a).mean(), w50=((p-a).abs()<=0.5*a).mean())
        rows.append(r)
    R = pd.DataFrame(rows).set_index('origin')
    m = R.mean(numeric_only=True)
    print(f"{name:28s} WAPE {m.wape:.4f} bias {m.bias:+.3f} w20 {m.w20:.3f} w50 {m.w50:.3f} h1 {m.h1:.4f} h1bias {m.h1bias:+.3f}", flush=True)
    return R
if __name__ == '__main__':
    R = run('v13 base'); print(R.round(3).to_string())
    run('SEAS_ANCHOR 0.5', {'SEAS_ANCHOR': 0.5})
    run('SEAS_ANCHOR 0.25', {'SEAS_ANCHOR': 0.25})
    run('SEAS_ANCHOR 1.0', {'SEAS_ANCHOR': 1.0})
    run('AMP floor 0.8', {'AMP_CLIP': (0.8, 1.0)})
    run('DRIFT_ADJ 0.98', {'DRIFT_ADJ': 0.98})
    run('W14 0.5', {'W14': 0.5})
