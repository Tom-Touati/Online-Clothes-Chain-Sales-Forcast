"""Backtest harness: oracle site (dicts hold actuals), scoreboard item lists, actuals from monthly_item."""
import sys, importlib, pandas as pd, numpy as np
sys.path.insert(0, ".")
import rules as R
it, di, st = R.load()
mi = pd.read_parquet("data/monthly_item.parquet")
sb = pd.read_csv("feedback/scoreboard.csv")
ACT = mi.pivot_table(index="style_color", columns="month", values="units", aggfunc="sum").fillna(0)
ORIG = ["2025-03","2025-04","2025-05","2025-06","2025-07","2025-08","2025-09","2025-10","2025-11","2025-12","2026-01","2026-02"]
BASEP = dict(R.P)
def run(name, over={}, origins=ORIG, verbose=False, hook=None):
    R.P.clear(); R.P.update(BASEP); R.P.update(over)
    rows = []
    for o in origins:
        B = pd.Period(o, "M")
        items = sb[sb.origin == o].style_color.unique().tolist()
        months = [str(B + h) for h in range(1, 5) if str(B + h) in ACT.columns]
        if len(months) < 4: continue
        f = R.forecast(B.end_time.normalize(), items, it, di, st)
        if hook: f = hook(f, o, items)
        a = ACT.reindex(index=items, columns=months).fillna(0)
        p = f[months]
        pt, at_ = p.sum(axis=1), a.sum(axis=1)
        ae = (pt - at_).abs()
        h1 = (p.iloc[:, 0] - a.iloc[:, 0]).abs().sum() / a.iloc[:, 0].sum()
        mb = (p.sum() / a.sum().values).round(3).tolist()
        rows.append(dict(o=o, wape=ae.sum() / at_.sum(), bias=pt.sum() / at_.sum() - 1, w20=(ae <= .2 * at_).mean(),
                         w50=(ae <= .5 * at_).mean(), h1=h1, mbias=mb))
    df = pd.DataFrame(rows)
    if verbose: print(df.round(4).to_string())
    m = df[["wape", "bias", "w20", "w50", "h1"]].mean()
    print(f"{name:40s} wape {m.wape:.4f} bias {m.bias:+.3f} w20 {m.w20:.3f} w50 {m.w50:.3f} h1 {m.h1:.4f} | summer(05-08) wape {df[df.o.isin(['2025-05','2025-06','2025-07','2025-08'])].wape.mean():.4f}")
    return df
if __name__ == "__main__":
    run("v13 base (oracle site)", verbose=True)
