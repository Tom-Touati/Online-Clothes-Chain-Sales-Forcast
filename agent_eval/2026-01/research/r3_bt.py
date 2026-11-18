"""Full-rules backtest on past origins with the site plan = actuals (isolates the item logic).
Usage: python research/r3_bt.py [variant ...]   variants defined in VARIANTS below."""
import sys, copy
import pandas as pd, numpy as np
sys.path.insert(0, ".")
import rules
it, di, st = rules.load()
sb = pd.read_csv("feedback/scoreboard.csv")
mi = pd.read_parquet("data/monthly_item.parquet")
act = mi.pivot_table(index="style_color", columns="month", values="units", aggfunc="sum").fillna(0)
P0 = copy.deepcopy(rules.P)
ORIGINS = ["2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09",
           "2025-10", "2025-11", "2025-12"]
VARIANTS = {"base": {}}
for k in [0.1, 0.2, 0.3]:
    VARIANTS[f"drop{k}"] = dict(DROP_GROW=k)
for k in [0.03, 0.06, 0.1]:
    VARIANTS[f"roll{k}"] = dict(RET_ROLL=k)
VARIANTS["d2r6"] = dict(DROP_GROW=0.2, RET_ROLL=0.06)
VARIANTS["drift102"] = dict(DRIFT_ADJ=1.02)
VARIANTS["drift104"] = dict(DRIFT_ADJ=1.04)
VARIANTS["ampclip07"] = dict(AMP_CLIP=(0.7, 1.0))
VARIANTS["amp_fixed1"] = dict(ADAPT_AMP=False, SEAS_POW=1.0)
VARIANTS["anc0"] = dict(SEAS_ANCHOR=0.0)
VARIANTS["anc025"] = dict(SEAS_ANCHOR=0.25)
VARIANTS["chk10"] = dict(CH_K=10)
VARIANTS["chk40"] = dict(CH_K=40)
VARIANTS["k8"] = dict(SHRINK_K=8)
VARIANTS["w14_3"] = dict(W14=0.3)
VARIANTS["w14_5"] = dict(W14=0.5)
for f in [0.6, 0.8, 0.9]:
    VARIANTS[f"ampclip{f}"] = dict(AMP_CLIP=(f, 1.0))
VARIANTS["anc075"] = dict(SEAS_ANCHOR=0.75, AMP_CLIP=(0.7, 1.0))
VARIANTS["c7_w5"] = dict(AMP_CLIP=(0.7, 1.0), W14=0.5)
VARIANTS["c7_w6"] = dict(AMP_CLIP=(0.7, 1.0), W14=0.6)
VARIANTS["c7_win2"] = dict(AMP_CLIP=(0.7, 1.0), AMP_WIN=2)
VARIANTS["c7_win4"] = dict(AMP_CLIP=(0.7, 1.0), AMP_WIN=4)
VARIANTS["fade90"] = dict(AGE_FADE=0.90)
VARIANTS["fade_r6"] = dict(AGE_FADE_RANGE=(1.0, 6.0))
VARIANTS["wlong35"] = dict(WLONG=0.35)
VARIANTS["ramp0_25"] = dict(RAMP0_NEW=2.5)


def run(over, keep=False):
    rules.P.clear(); rules.P.update(P0); rules.P.update(over)
    rows = []
    for o in ORIGINS:
        B = pd.Period(o, "M")
        items = sb[sb.origin == o].style_color.unique().tolist()
        f = rules.forecast(B.end_time.normalize(), items, it, di, st)
        for t in f.columns:
            if t in act.columns:
                rows.append(pd.DataFrame({"origin": o, "style_color": items, "month": t,
                                          "h": (pd.Period(t, "M") - B).n, "pred": f[t].values,
                                          "actual": act[t].reindex(items).fillna(0).values}))
    r = pd.concat(rows)
    full = r.groupby("origin").month.nunique() == 4
    w = r[r.origin.isin(full[full].index)].groupby(["origin", "style_color"])[["pred", "actual"]].sum().reset_index()
    def sc(x):
        e = (x.pred - x.actual).abs()
        return pd.Series({"wape": e.sum() / x.actual.sum(), "bias": x.pred.sum() / x.actual.sum() - 1,
                          "w20": (e <= 0.2 * x.actual).mean(), "w50": (e <= 0.5 * x.actual).mean()})
    by = w.groupby("origin").apply(sc)
    h1 = r[r.h == 1].groupby("origin").apply(sc).wape
    out = by.mean().round(4).to_dict(); out["h1_wape"] = round(h1.mean(), 4)
    out["h1_last3"] = round(h1.iloc[-3:].mean(), 4)
    return (out, r) if keep else out


if __name__ == "__main__":
    names = sys.argv[1:] or list(VARIANTS)
    for n in names:
        print(n, run(VARIANTS[n]), flush=True)
