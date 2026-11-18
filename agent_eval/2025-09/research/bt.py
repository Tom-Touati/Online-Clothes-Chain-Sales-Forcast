"""Backtest harness: origins Feb..Aug-25, scoreboard item lists, totals of complete months <= Sep-25.
Oracle site levels (SITE_WEB_PD / OTHER_PD hold actuals for past months)."""
import pandas as pd, numpy as np, sys, importlib, json
sys.path.insert(0, "."); import rules as R
it, di, st = R.load()
sb = pd.read_csv("feedback/scoreboard.csv")
mi = pd.read_parquet("data/monthly_item.parquet")
ORIG = ["2025-02","2025-03","2025-04","2025-05","2025-06","2025-07","2025-08"]
def run(over, label, extra=None):
    P0 = dict(R.P); R.P.update(over)
    rows = []; wrel = []
    for o in ORIG:
        items = sb[sb.origin == o].style_color.unique().tolist()
        end = pd.Period(o) + 4
        origin = pd.Period(o).end_time.normalize()
        F = R.forecast(origin, items, it, di, st)
        ms = [c for c in F.columns if c <= "2025-09"]
        act = mi[mi.month.isin(ms)].pivot_table(index="style_color", columns="month", values="units", aggfunc="sum").reindex(index=items, columns=ms).fillna(0)
        p, a = F[ms].sum(axis=1), act.sum(axis=1)
        wape = (p - a).abs().sum() / a.sum(); bias = p.sum() / a.sum() - 1
        r = a / p.replace(0, np.nan)
        within20 = ((p - a).abs() <= 0.2 * a).mean(); within50 = ((p - a).abs() <= 0.5 * a).mean()
        g = it.grp.reindex(items)
        wr = (a.groupby(g).sum() / p.groupby(g).sum() / (a.sum() / p.sum()))["WINTER"]
        rows.append((o, wape, bias, within20, within50, wr))
    R.P.clear(); R.P.update(P0)
    df = pd.DataFrame(rows, columns=["o", "wape", "bias", "w20", "w50", "wint"])
    print(f"{label:34s} WAPE {df.wape.mean():.4f} | " + " ".join(f"{w:.3f}/{b*100:+.0f}" for w, b in zip(df.wape, df.bias))
          + f" | 20% {df.w20.mean():.3f} 50% {df.w50.mean():.3f} | WINT " + " ".join(f"{x:.2f}" for x in df.wint), flush=True)
    return df
if __name__ == "__main__":
    V = {"v7 (fixed pow .8, anchor)": dict(ADAPT_AMP=False),
         "v8 adapt amp + anchor": {},
         "adapt amp, anchor off": dict(SEAS_ANCHOR=0.0),
         "fixed pow .6, anchor off": dict(ADAPT_AMP=False, SEAS_POW=0.6, SEAS_ANCHOR=0.0),
         "fixed pow .6, anchor": dict(ADAPT_AMP=False, SEAS_POW=0.6),
         "adapt, AMP_WIN 2": dict(AMP_WIN=2),
         "adapt, AMP_WIN 4": dict(AMP_WIN=4),
         "adapt + RET_POW 1.0": dict(RET_POW=1.0),
         "adapt + RET_POW 0.8": dict(RET_POW=0.8),
         "adapt + POW .03": dict(POW=0.03),
         "adapt + DRIFT 1.02": dict(DRIFT_ADJ=1.02)}
    sel = sys.argv[1:] or list(V)
    for k in V:
        if k in sel or sel == list(V): run(V[k], k)
