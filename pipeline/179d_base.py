"""D238 base table for the level decomposition and the deep error analysis (user, night of 17 Nov: "the true challenge is to know how to
decompose the levels ... deep error analysis: what are the items we ain't good at, why and when ... try PCA like hierarchical").
One row per backtest item-origin (13 origins, 11,727 rows): actual 4-month total y, R3, the final system F (p2c32a: arithmetic mean of
rules, D198 quantile, NN, conformal TabPFN v2 dyn, TabPFN v2 dyn 32) and its members, the hierarchy (site / macro / product type /
look cluster / style / colour), and descriptors: age, volume, Black Friday window, member spread, TikTok share, discount, out-of-stock
share, launch-in-look count, seasonality strength. Look clusters: k-means (K = 40) at each origin on the cart PPMI-SVD vector (12 months
up to the origin) and the content vector (CLIP image + text), each L2-normalised and half-weighted.
Writes data/dec_base.parquet."""
import sys, contextlib, io, numpy as np, pandas as pd
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import KMeans
sys.stdout.reconfigure(encoding="utf-8")
with contextlib.redirect_stdout(io.StringIO()): exec(open("pipeline/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); qv = lambda f: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(key).q50.values
T["qdyn"] = qv("data/q_backtest_fkdyn_dyn.parquet"); T["pf2c"] = qv("data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet"); T["pf2d32"] = qv("data/q_backtest_fktab_tabpfn_v2_dyn_e32.parquet")
MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; T["F"] = T[MEM].clip(lower=0).mean(1)
it = pd.read_parquet("data/items.parquet").set_index("style_color")
B = T[["origin", "style_color", "y", "R3", "F", "level", "age_m", "target"] + MEM].copy()
B["ptype"] = it.product_type.reindex(B.style_color).fillna("na").values; B["style"] = B.style_color.str.split("_").str[0]
MAC = {"Top": "top", "Tank Top": "top", "T-Shirt": "top", "Crop Top": "top", "Corset": "top", "Strapless": "top", "Bodysuit": "top", "Shirt": "top", "Knitted Top": "top",
       "Jeans": "bottom", "Shorts": "bottom", "Skirt": "bottom", "Skort": "bottom", "Pants": "bottom", "Short Jeans": "bottom", "Sweatpants": "warm",
       "Hoodie": "warm", "Sweater": "warm", "Jacket": "warm", "Cardigan": "warm", "Dress": "dress", "Swimwear Top": "swim", "Swimwear Bottom": "swim"}
B["macro"] = B.ptype.map(MAC).fillna("other")
B["bf_window"] = B.origin.isin(["2025-07", "2025-08", "2025-09", "2025-10"]).astype(int)
B["spread"] = np.log(B[MEM].clip(lower=1).max(1) / B[MEM].clip(lower=1).min(1))
F0 = pd.read_parquet("data/features.parquet"); F0 = F0[F0.h == 1][["origin", "style_color", "sh_tt", "disc_item", "own_sea", "trend91", "f_r28"]]
B = B.merge(F0, on=["origin", "style_color"], how="left")
try: B = B.merge(pd.read_parquet("data/q_backtest_oos_level.parquet")[["origin", "style_color"]].assign(_=1).drop(columns="_"), on=["origin", "style_color"], how="left")
except Exception: pass
# look clusters at each origin
items = np.sort(it.index.values); n = len(items); P = pd.read_parquet("data/basket_pairs_month.parquet"); CM = {}
for m, g in P.groupby("m"):
    c = np.zeros((n, n), np.float32); c[g.it.values, g.it_b.values] = g.n.values; CM[m] = c + c.T
CE = np.load("data/content_emb.npz", allow_pickle=True); cix = pd.Series(range(len(CE["items"])), index=CE["items"].astype(str)).reindex(items).values
CON = np.hstack([CE["img"][cix], CE["txt"][cix]]); u, s_, _ = randomized_svd(CON, 24, random_state=0); CONV = u * s_
rown = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9); Pm = lambda o, k: str(pd.Period(o, "M") + k)
rows = []
for o in sorted(B.origin.unique()):
    C = sum(CM[m] for m in CM if Pm(o, -11) <= m <= o); r = C.sum(1); tot = max(r.sum(), 1); ctx = r ** 0.75; ctx = ctx / max(ctx.sum(), 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"): pmi = np.nan_to_num(np.maximum(np.log((C / tot) / np.outer(r / tot, ctx)), 0), posinf=0, neginf=0)
    uu, ss, _ = randomized_svd(rown(pmi), 24, random_state=0); X = np.hstack([np.sqrt(0.5) * rown(uu * np.sqrt(ss)), np.sqrt(0.5) * rown(CONV)])
    lab = KMeans(40, n_init=4, random_state=0).fit_predict(X); rows.append(pd.DataFrame({"origin": o, "style_color": items, "look": lab}))
B = B.merge(pd.concat(rows), on=["origin", "style_color"], how="left"); B["look"] = B.origin + "_L" + B.look.astype(str)
B["e"] = np.log((B.y + 1) / (B.F + 1)); B["g"] = np.log((B.y + 1) / (B.R3 + 1)); B["ae"] = (B.F - B.y).abs()
B.to_parquet("data/dec_base.parquet", index=False); print(B.shape, "| look clusters per origin", B.groupby("origin").look.nunique().median())
REC = B.origin >= "2025-06"; print("recent WAPE / R3 of F:", round(B[REC].ae.sum() / (B[REC].R3 - B[REC].y).abs().sum(), 4))
