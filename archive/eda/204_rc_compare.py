"""D333-D335 (fork 2): compares every recommender run (data/rc_recsys_preds_<tag>.parquet) on the agreed scores, recent origins:
WAPE / R3, within +/-20% (all items, and in the smallest and largest volume fifth), bias, GMRAE (D313: geometric mean of
(|F - y| + 1) / (|R3 - y| + 1), every item equal); alone, added to the five-member mean as a sixth member, and in place of the member
it helps most; item-bootstrap interval of the six-member difference against the five-member mean (0.3939).
Usage: python eda/204_rc_compare.py tags=_e4,_if_poisson_w0,...   stdout -> eda/rc_compare.txt, data/rc_compare.json"""
import sys, json, glob, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 240)
KW = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
TAGS = KW["tags"].split(",") if "tags" in KW else sorted(f[len("data/rc_recsys_preds"):-8] for f in glob.glob("data/rc_recsys_preds_*.parquet") if "test" not in f)
B = pd.read_parquet("data/dec_base_ecnsnn.parquet"); G = pd.read_parquet("data/geo_ecnsnn_x_preds.parquet")[["origin", "style_color", "geo"]]; B = B.merge(G, on=["origin", "style_color"])
MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; y = B.y.values; rec = (B.origin >= "2025-06").values; ar = np.abs(B.R3.values - y); base = B[MEM].values.mean(1); rng = np.random.default_rng(0)
fifth = B.groupby("origin")["level"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)).values
def sc(f, m=rec):
    e = np.abs(f - y)
    return dict(wape=round(float(e[m].sum() / ar[m].sum()), 4), hit=round(float((e[m] <= 0.2 * y[m]).mean()), 3), hit_small=round(float((e[m & (fifth == 0)] <= 0.2 * y[m & (fifth == 0)]).mean()), 3),
                hit_large=round(float((e[m & (fifth == 4)] <= 0.2 * y[m & (fifth == 4)]).mean()), 3), bias=round(float(f[m].sum() / y[m].sum() - 1), 3), gmrae=round(float(np.exp(np.log((e[m] + 1) / (ar[m] + 1)).mean())), 3),
                all13=round(float(e.sum() / ar.sum()), 4))
def boot(fb, B_=1000):
    Z = pd.DataFrame({"s": B.style_color, "a": np.abs(base - y), "b": np.abs(fb - y), "r": ar})[rec].groupby("s").sum().values; n = len(Z); T = Z[rng.integers(0, n, (B_, n))].sum(1); d = T[:, 1] / T[:, 2] - T[:, 0] / T[:, 2]
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
rows = [dict(run="R3", model="", **sc(B.R3.values)), dict(run="R3 + rules", model="", **sc(B.rules.values)), dict(run="five-member mean", model="", **sc(base)), dict(run="submitted (geo)", model="", **sc(B.geo.values))]
for tg in TAGS:
    R = pd.read_parquet(f"data/rc_recsys_preds{tg}.parquet"); D = B[["origin", "style_color"]].merge(R, on=["origin", "style_color"], how="left")
    for c in [c for c in R.columns if c.startswith("f_")]:
        f = D[c].fillna(pd.Series(B.R3.values)).values; six = (B[MEM].values.sum(1) + f) / 6; sw = {m: (B[MEM].values.sum(1) - B[m].values + f) / 5 for m in MEM}; bm = min(sw, key=lambda m: sc(sw[m])["wape"])
        rows.append(dict(run=tg.lstrip("_"), model=c[2:], **sc(f), six=sc(six)["wape"], six_hit=sc(six)["hit"], six_ci=boot(six), swap=f"{bm} {sc(sw[bm])['wape']}"))
T = pd.DataFrame(rows); print(T.to_string(index=False)); json.dump(rows, open("data/rc_compare.json", "w"), indent=0)
