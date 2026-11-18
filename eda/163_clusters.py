"""D158: multi-view item clusters (user request: purchase similarity, sales correlation / covariance, fashion descriptors, weather).
At every origin o (2024-03 .. 2026-06), from data up to o only, each item alive in the last 12 months gets four view vectors:
  cart      PPMI of cart co-occurrence (12 months, context smoothing 0.75), SVD to 24 dims             (purchase similarity)
  comove    correlation of monthly log-sales changes over the last 18 months (items with >= 6 months of sales),
            positive part as an affinity, spectral embedding to 12 dims                             (sales correlation / covariance)
  descr     tagger attributes (one-hot) + the 14 LLM scores (standardised), SVD to 12 dims          (fashion descriptors)
  weather   temperature preference beta_T, warm-state skew of the item's mix, cold-state skew   (weather)
Each view is L2-normalised per item and weighted (cart 0.40, comove 0.25, descr 0.25, weather 0.10; a missing view's weight is
spread over the others), concatenated, and clustered with k-means (K = 30, 5 restarts).
Cluster-level features per item (labels change between origins, so only summaries are passed on):
  cl_sea4   the cluster's last-year ramp: its units in [o-11 .. o-8] / units in [o-14 .. o-12] x 3/4 (log)
  cl_mom3   the cluster's 3-month momentum; cl_yoy its year-on-year growth (last 3 months)
  cl_betaT  the cluster's mean temperature preference; cl_size its number of live items; cl_dev the item's log level minus the
            cluster's median log level
Writes data/cluster_feats.parquet and data/clusters_2026-06.parquet (labels and view vectors at the last origin, for profiles)."""
import sys, glob, numpy as np, pandas as pd, time
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import KMeans
sys.stdout.reconfigure(encoding="utf-8")
K, WTS = 30, {"cart": 0.40, "comove": 0.25, "descr": 0.25, "weather": 0.10}
items = np.sort(pd.read_parquet("data/items.parquet").style_color.values); n = len(items)
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; M = M.reindex(items).fillna(0)
P = pd.read_parquet("data/basket_pairs_month.parquet"); CM = {}
for m, g in P.groupby("m"):
    c = np.zeros((n, n), np.float32); c[g.it.values, g.it_b.values] = g.n.values; CM[m] = c + c.T
# descriptors: same attribute block as eda/153 plus LLM scores
src = open("eda/153_basket_emb.py", encoding="utf-8").read(); exec(src[src.index("it = pd.read_parquet"):src.index("P = pd.read_parquet")])
ATTn = ATT / np.maximum(np.linalg.norm(ATT, axis=1, keepdims=True), 1e-9)
Lr = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("data/llm/scores_part*.csv"))]).drop_duplicates("style_color").set_index("style_color").reindex(items)
Lz = ((Lr - Lr.mean()) / Lr.std()).fillna(0).values
u, s, _ = randomized_svd(np.hstack([ATTn, 0.35 * Lz / np.sqrt(Lz.shape[1])]), 12, random_state=0); DESCR = u * s
WF = pd.read_parquet("data/weather_event_feats.parquet")[["origin", "style_color", "beta_T"]]
RF = pd.read_parquet("data/region_feats.parquet")[["origin", "style_color", "clim_warm_rel", "clim_cold_rel"]]
Pm = lambda o, k: str(pd.Period(o, "M") + k)
U = lambda o, ks: M.reindex(columns=[Pm(o, k) for k in ks]).fillna(0).values
rown = lambda X: X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
rows, t0 = [], time.time()
for o in [str(p) for p in pd.period_range("2024-03", "2026-06", freq="M")]:
    alive = U(o, range(-11, 1)).sum(1) > 0
    win = [m for m in CM if Pm(o, -11) <= m <= o]; C = sum(CM[m] for m in win); r = C.sum(1); tot = r.sum()
    ctx = r ** 0.75; ctx = ctx / max(ctx.sum(), 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"): pmi = np.nan_to_num(np.maximum(np.log((C / max(tot, 1)) / np.outer(r / max(tot, 1), ctx)), 0), posinf=0, neginf=0)
    u, s, _ = randomized_svd(rown(pmi), 24, random_state=0); CART = u * np.sqrt(s); has_cart = r > 0
    H = np.log1p(U(o, range(-17, 1))); live_m = (U(o, range(-17, 1)) > 0).sum(1); dH = np.diff(H, axis=1)
    okc = live_m >= 6; Z = dH[okc] - dH[okc].mean(1, keepdims=True); Z = Z / np.maximum(Z.std(1, keepdims=True), 1e-9)
    R = np.clip(Z @ Z.T / Z.shape[1], 0, None); dg = R.sum(1); Ln = R / np.sqrt(np.outer(dg, dg) + 1e-9)
    ev, evec = np.linalg.eigh(Ln); CO = np.zeros((n, 12)); CO[okc] = evec[:, -13:-1] if okc.sum() > 13 else 0
    wf = WF[WF.origin == o].set_index("style_color").reindex(items); rf = RF[RF.origin == o].set_index("style_color").reindex(items)
    WE = np.column_stack([np.nan_to_num(wf.beta_T.values) * 10, np.nan_to_num(rf.clim_warm_rel.values) * 5, np.nan_to_num(rf.clim_cold_rel.values) * 5])
    views = {"cart": (rown(CART), has_cart), "comove": (rown(CO), okc), "descr": (rown(DESCR), np.ones(n, bool)), "weather": (np.tanh(WE), np.isfinite(wf.beta_T.values))}
    wsum = sum(WTS[v] * ok for v, (_, ok) in views.items())
    X = np.hstack([np.sqrt(WTS[v] * ok / np.maximum(wsum, 1e-9))[:, None] * V for v, (V, ok) in views.items()])
    idx = np.where(alive)[0]; km = KMeans(K, n_init=5, random_state=0).fit(X[idx]); lab = np.full(n, -1); lab[idx] = km.labels_
    l3 = U(o, [-2, -1, 0]).sum(1); p3 = U(o, [-5, -4, -3]).sum(1); y3 = U(o, [-14, -13, -12]).sum(1); yt = U(o, [-11, -10, -9, -8]).sum(1)
    df = pd.DataFrame({"lab": lab, "l3": l3, "p3": p3, "y3": y3, "yt": yt, "beta": wf.beta_T.values, "lvl": np.log1p(l3 / 3)})
    g = df[df.lab >= 0].groupby("lab").agg(l3=("l3", "sum"), p3=("p3", "sum"), y3=("y3", "sum"), yt=("yt", "sum"), beta=("beta", "mean"), size=("l3", "size"), lvl=("lvl", "median"))
    gi = g.reindex(lab)
    f = pd.DataFrame({"origin": o, "style_color": items, "cl_sea4": np.where(gi.y3.values > 0, np.log((gi.yt.values + 1) / (gi.y3.values * 4 / 3 + 1)), np.nan),
                      "cl_mom3": np.log((gi.l3.values + 1) / (gi.p3.values + 1)), "cl_yoy": np.where(gi.y3.values > 0, np.log((gi.l3.values + 1) / (gi.y3.values + 1)), np.nan),
                      "cl_betaT": gi.beta.values, "cl_size": gi["size"].values, "cl_dev": df.lvl.values - gi.lvl.values})
    f.loc[lab < 0, [c for c in f.columns if c.startswith("cl_")]] = np.nan
    rows.append(f)
    if o == "2026-06":
        pd.DataFrame({"style_color": items, "cluster": lab, **{f"x{j}": X[:, j] for j in range(X.shape[1])}}).to_parquet("data/clusters_2026-06.parquet", index=False)
    print(o, f"alive {alive.sum()} | comove items {okc.sum()} | {time.time() - t0:.0f}s", flush=True)
pd.concat(rows, ignore_index=True).to_parquet("data/cluster_feats.parquet", index=False); print("saved")
