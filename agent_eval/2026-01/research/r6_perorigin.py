"""Per-origin 4-month WAPE (and Jan-26 h=1 for the Dec origin) for selected variants."""
import sys
import pandas as pd, numpy as np
sys.path.insert(0, "research"); sys.path.insert(0, ".")
import r3_bt as bt
V = {"base": {}, "c7": dict(AMP_CLIP=(0.7, 1.0)), "c7a75": dict(AMP_CLIP=(0.7, 1.0), SEAS_ANCHOR=0.75),
     "c7a75w5": dict(AMP_CLIP=(0.7, 1.0), SEAS_ANCHOR=0.75, W14=0.5),
     "c7w5": dict(AMP_CLIP=(0.7, 1.0), W14=0.5)}
res = {}
for n, over in V.items():
    out, r = bt.run(over, keep=True)
    w = r.groupby(["origin", "style_color"])[["pred", "actual"]].sum()
    s = w.groupby("origin").apply(lambda x: (x.pred - x.actual).abs().sum() / x.actual.sum())
    j = r[(r.origin == "2025-12")]
    s["Dec->Jan h1"] = (j.pred - j.actual).abs().sum() / j.actual.sum()
    jg = j.assign(g=j.style_color.map(bt.it.grp)).groupby("g")[["pred", "actual"]].sum()
    s["JanW a/p"] = jg.loc["WINTER", "actual"] / jg.loc["WINTER", "pred"]
    s["JanS a/p"] = jg.loc["SUMMER", "actual"] / jg.loc["SUMMER", "pred"]
    res[n] = s; print(n, out, flush=True)
print(pd.DataFrame(res).round(4).to_string())
