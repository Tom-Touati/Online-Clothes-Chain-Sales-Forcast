"""Backtest of rule variants: origins Jan-Jun 2025, data <= origin only (site plan dicts hold actual site
levels for past months -> the web / other LEVEL is oracle; this tests everything else). Scored on per-item
totals of complete target months <= 2025-07."""
import pandas as pd, numpy as np, sys, copy
sys.path.insert(0, "."); import rules as R
it, di, st = R.load()
mi = pd.read_parquet("data/monthly_item.parquet")
A = mi.pivot_table(index="style_color", columns="month", values="units", aggfunc="sum").fillna(0)
sb = pd.read_csv("feedback/scoreboard.csv")
LAST = pd.Period("2025-07")
ORIG = ["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31", "2025-06-30"]
def items_for(o):
    k = o[:7]
    s = sb[sb.origin == k].style_color.unique()
    if len(s): return list(s)
    d = di[(di.date <= o) & (di.date > pd.Timestamp(o) - pd.Timedelta(days=60))]
    return list(d.groupby("style_color").units.sum().loc[lambda x: x > 0].index)
ITEMS = {o: items_for(o) for o in ORIG}
def run(label, **over):
    P0, O0 = dict(R.P), copy.deepcopy(R.OCCASION)
    for k, v in over.items():
        if k in R.P: R.P[k] = v
        else: raise KeyError(k)
    res = []
    for o in ORIG:
        f = R.forecast(o, ITEMS[o], it, di[di.date <= o], st[st.date <= o])
        cols = [c for c in f.columns if pd.Period(c) <= LAST]
        p = f[cols].sum(axis=1); a = A.reindex(index=p.index, columns=cols).fillna(0).sum(axis=1)
        e = (p - a).abs().sum() / a.sum(); b = p.sum() / a.sum() - 1
        r = p / a.replace(0, np.nan)
        res.append((e, b, ((r > 0.8) & (r < 1.2)).mean(), ((r > 0.5) & (r < 1.5)).mean()))
    R.P.clear(); R.P.update(P0); R.OCCASION.clear(); R.OCCASION.update(O0)
    r = np.array(res)
    print(f"{label:28s} WAPE {r[:,0].mean():.4f} | " + " ".join(f"{x:.3f}/{y:+.0%}" for x, y, *_ in res)
          + f" | +-20% {r[:,2].mean():.3f} +-50% {r[:,3].mean():.3f}", flush=True)
if __name__ == "__main__":
    run("v6 base")
    run("other 0.9^h", OTHER_PATH=False)
    run("SEAS_POW 0.8", SEAS_POW=0.8); run("SEAS_POW 1.0", SEAS_POW=1.0)
    run("SEAS_ANCHOR 0", SEAS_ANCHOR=0.0); run("SEAS_ANCHOR 1.0", SEAS_ANCHOR=1.0)
    run("SHRINK_K 32", SHRINK_K=32)
    run("POW 0.03", POW=0.03)
    run("W14 0.3 / 0.5", W14=0.5)
    run("WLONG 0.1", WLONG=0.1); run("WLONG 0.3", WLONG=0.3)
    run("RET_POW 0.8", RET_POW=0.8); run("RET_POW 0.4", RET_POW=0.4)
    run("RAMP 1.4/1.25/1.1", RAMP0=1.4, RAMP1=1.25, RAMP2=1.1)
    run("CH_K 10", CH_K=10)
