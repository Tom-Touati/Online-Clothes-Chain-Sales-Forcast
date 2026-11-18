"""Honest backtest of rule variants: origins Jan..May-25, data <= origin only, scored on per-item totals over
the complete target months (<= Jun-25).  Usage: python research/bt.py   (variants defined below)"""
import sys, copy
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import rules
it, di, st = rules.load()
sb = pd.read_csv("feedback/scoreboard.csv")
act = di.groupby(["style_color", "m"]).units.sum()
ORIG = ["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31"]
LAST = pd.Period("2025-06")

def items_for(o):
    key = o[:7]
    s = sb[sb.origin == key].style_color.unique()
    if len(s):
        return list(s)
    d = di[(di.date <= o) & (di.date > pd.Timestamp(o) - pd.Timedelta(days=60))]
    return list(d.groupby("style_color").units.sum().loc[lambda x: x > 0].index.intersection(it.index))

ITEMS = {o: items_for(o) for o in ORIG}

def run(P_over=None, glob_over=None, chan=False):
    P0, G0 = dict(rules.P), {k: copy.deepcopy(getattr(rules, k)) for k in (glob_over or {})}
    rules.P.update(P_over or {})
    for k, v in (glob_over or {}).items():
        setattr(rules, k, v)
    res = []
    try:
        for o in ORIG:
            items = ITEMS[o]
            f = rules.forecast(o, items, it, di, st)
            cols = [c for c in f.columns if pd.Period(c) <= LAST]
            pr = f[cols].sum(axis=1)
            a = pd.Series([sum(act.get((i, pd.Period(c)), 0) for c in cols) for i in items], index=items)
            e = (pr - a).abs()
            res.append(dict(origin=o[:7], k=len(cols), wape=e.sum() / a.sum(), bias=pr.sum() / a.sum() - 1,
                            w20=(e <= 0.2 * a).mean(), w50=(e <= 0.5 * a).mean(),
                            h1=(f[cols[0]] - [act.get((i, pd.Period(cols[0])), 0) for i in items]).abs().sum()
                            / sum(act.get((i, pd.Period(cols[0])), 0) for i in items)))
    finally:
        rules.P.clear(); rules.P.update(P0)
        for k, v in G0.items():
            setattr(rules, k, v)
    r = pd.DataFrame(res)
    return r

VARIANTS = {
    "v4": ({}, {}),
    "closed7": ({"STORE_CLOSED_DAYS": 5}, {}),
    "wdefl1": ({"W_DEFL": 1.0}, {}),
    "spow0.8": ({"SEAS_POW": 0.8}, {}),
    "spow0.6": ({"SEAS_POW": 0.6}, {}),
    "anchor0.5": ({"SEAS_ANCHOR": 0.5}, {}),
    "pow0.05": ({"POW": 0.05}, {}),
    "other0.9": ({"OTHER_MONTHLY": 0.9}, {}),
    "k16": ({"SHRINK_K": 16}, {}),
}
if __name__ == "__main__":
    sel = sys.argv[1:] or list(VARIANTS)
    rows = []
    for name in sel:
        r = run(*VARIANTS[name])
        print(f"{name:12s} mean {r.wape.mean():.4f}  w20 {r.w20.mean():.3f} w50 {r.w50.mean():.3f}  h1 {r.h1.mean():.4f} | "
              + " ".join(f"{o}:{w:.3f}/{b:+.2f}" for o, w, b in zip(r.origin, r.wape, r.bias)), flush=True)
