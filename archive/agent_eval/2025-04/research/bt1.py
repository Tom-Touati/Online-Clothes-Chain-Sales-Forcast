"""Honest backtests: origins Jan/Feb/Mar-25, only data <= origin; scored on target months <= Apr-25."""
import pandas as pd, numpy as np, sys, itertools
sys.path.insert(0,"."); import rules as R
it, di, st = R.load()
mi = pd.read_parquet("data/monthly_item.parquet")
A = mi.pivot_table(index="style_color", columns="month", values="units", aggfunc="sum").fillna(0)
sb = pd.read_csv("feedback/scoreboard.csv")
def items_for(o):
    k = o[:7]
    if k in ("2025-02","2025-03"): return sorted(sb[sb.origin==k].style_color.unique())
    d = di[(di.date<=o)&(di.date>pd.Timestamp(o)-pd.Timedelta(days=60))]
    return sorted(d.groupby("style_color").units.sum().pipe(lambda x:x[x>0]).index)
ORIG = ["2025-01-31","2025-02-28","2025-03-31"]
CACHE = {o: items_for(o) for o in ORIG}
def run(over, verbose=False):
    old = dict(R.P); R.P.update(over); res=[]
    try:
        for o in ORIG:
            items = CACHE[o]
            f = R.forecast(o, items, it, di[di.date<=o], st[st.date<=o])
            cols=[c for c in f.columns if c<="2025-04"]
            p = f[cols].sum(axis=1); a = A.reindex(index=items, columns=cols).fillna(0).sum(axis=1)
            mb = {c: f[c].sum()/A.reindex(index=items)[c].sum()-1 for c in cols}
            res.append(dict(o=o[:7], wape=(p-a).abs().sum()/a.sum(), bias=p.sum()/a.sum()-1,
                            in50=((p-a).abs()<=0.5*a).mean(), mb=" ".join(f"{v:+.2f}" for v in mb.values())))
    finally:
        R.P.clear(); R.P.update(old)
    r = pd.DataFrame(res)
    return r
variants = {
 "prev (v2)": dict(PRE_ASYM=False),
 "asym": dict(),
 "pre0": dict(ANCHOR_PRE=0.0, RET_ANCHOR_PRE=0.0),
 "defl0.5": dict(W_DEFL=0.5),
 "defl1": dict(W_DEFL=1.0),
 "oth0.9": dict(OTHER_MONTHLY=0.9),
 "defl0.5+oth0.9": dict(W_DEFL=0.5, OTHER_MONTHLY=0.9),
 "defl1+oth0.9": dict(W_DEFL=1.0, OTHER_MONTHLY=0.9),
 "defl0.5+oth0.9 pre0": dict(W_DEFL=0.5, OTHER_MONTHLY=0.9, ANCHOR_PRE=0.0, RET_ANCHOR_PRE=0.0),
}
for k,v in variants.items():
    r = run(v)
    print(f"{k:24s} mean WAPE {r.wape.mean():.3f} | " + " | ".join(f"{x.o}: w{x.wape:.3f} b{x.bias:+.3f} ±50 {x.in50:.2f} [{x.mb}]" for x in r.itertuples()))
