"""Backtest harness: origins Feb-Jul-25 (scoreboard item lists), data <= origin, scored on per-item totals of
complete months <= Aug-25.  Site plans hold actual levels for past months (oracle site level)."""
import os, sys, json, pandas as pd, numpy as np
sys.path.insert(0, os.getcwd()); import rules as R
it, di, st = R.load()
sb = pd.read_csv("feedback/scoreboard.csv")
act = di.groupby(["style_color", "m"]).units.sum()
ORIG = ["2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07"]
def run(over, label, verbose=False):
    old = dict(R.P); R.P.update(over)
    rows, res = [], []
    for o in ORIG:
        op = pd.Period(o); origin = op.end_time.normalize()
        items = sb[sb.origin == o].style_color.unique().tolist()
        F = R.forecast(origin, items, it, di[di.date <= origin], st[st.date <= origin])
        ms = [c for c in F.columns if c <= "2025-08"]
        A = pd.DataFrame({m: act.xs(pd.Period(m), level="m").reindex(items).fillna(0) for m in ms})
        p, a = F[ms].sum(axis=1), A.sum(axis=1)
        e = (p - a).abs().sum() / a.sum(); b = p.sum() / a.sum() - 1
        r = p / a.replace(0, np.nan)
        w20 = ((p - a).abs() <= 0.2 * a).mean(); w50 = ((p - a).abs() <= 0.5 * a).mean()
        g = it.grp.reindex(items)
        grel = (a.groupby(g).sum() / p.groupby(g).sum()) / (a.sum() / p.sum())
        res.append(dict(o=o, wape=e, bias=b, w20=w20, w50=w50, W=grel.get("WINTER"), S=grel.get("SUMMER")))
    R.P.clear(); R.P.update(old)
    df = pd.DataFrame(res)
    print(f"{label:28s} mean {df.wape.mean():.4f} | " + " ".join(f"{x.wape:.3f}/{x.bias:+.0%}" for x in df.itertuples())
          + f" | w20 {df.w20.mean():.3f} w50 {df.w50.mean():.3f} | W " + " ".join(f"{x:.2f}" for x in df.W)
          + " S " + " ".join(f"{x:.2f}" for x in df.S))
    return df
if __name__ == "__main__":
    variants = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"v6": {}}
    for k, v in variants.items(): run(v, k)
