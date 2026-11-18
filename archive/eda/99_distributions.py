"""Distribution forecasts for the 4-month total (D117), centred on the three-way combination c (rules x quantile x NN).
All variants are time-honest: what is used at origin o is learned from origins o' with o' + 4 <= o; origins with fewer than
1,500 such rows use D0.
  D0 production : the quantile model's own log spread around c, widened by k = 1.25 (k was fitted on the recent origins: in-sample)
  D1 k honest   : the same, k chosen on the history so that P25-P75 covers 50% (grid 0.8-2.0)
  D2 conformal  : quantiles of the history's log residual r = log((y+1)/(c+1)) within bins of member disagreement (quintiles) x
                  the quantile model's log IQR (terciles); bins under 150 rows fall back to the disagreement bin alone; median kept at c
  D3 lgb dist   : LightGBM quantile regression of r at the 5 levels from disagreement, member log-differences, the quantile model's
                  log IQR, level, age, volume, retail share, trend, product group, occasion tag, origin month; median kept at c
                  (each quantile is shifted by minus the predicted median, so only the shape is learned)
  D3m           : D3 with its own median (the centre moves too)
Scores per variant: pinball (sum over P10/P25/P50/P75/P90) / R3's (R3 with the D0 spread), coverage of P25-P75 and P10-P90,
interval score of P10-P90 (width + 2/0.2 x miss distance) relative to D0, and by disagreement quintile.
Writes data/dist_items.parquet (log-quantile offsets per variant) for the buying simulation."""
import numpy as np, pandas as pd, sys, lightgbm as lgb, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
T = pd.read_parquet("data/fixes_items.parquet")
q = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))
QC = ["q10", "q25", "q50", "q75", "q90"]; TAU = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
LQ = np.log1p(q[QC].values); SPR = LQ - LQ[:, [2]]                     # quantile-model log offsets around its median
c = T.r_q_nn.values; lc = np.log1p(c); y = T.y.values; ly = np.log1p(y); r = ly - lc
L = np.log(T[["rules", "qmodel", "nn", "lgb"]].clip(lower=0.5).values); dis = L.std(axis=1); qiqr = SPR[:, 3] - SPR[:, 1]
ORIG = sorted(T.origin.unique()); ORD = {o: pd.Period(o, "M").ordinal for o in ORIG}; OO = T.origin.map(ORD).values
OFF = {k: 1.25 * SPR.copy() for k in ["D0", "D1", "D2", "D3", "D3m"]}
X = pd.DataFrame({"dis": dis, "dr": L[:, 0] - L[:, 1], "dn": L[:, 2] - L[:, 1], "dl": L[:, 3] - L[:, 1], "qiqr": qiqr, "lvl": np.log1p(T.level.values),
                  "age": T.age_m.values, "vol": T.vol.values, "ret": T.sh_ret.values, "trend": T.trend91.values,
                  "macro": T.macro.astype("category").values, "occ": (T.occ == "tagged").astype(int).values, "om": T.origin.str[-2:].astype(int).values})
KLOG = []
for o in ORIG:
    te = np.flatnonzero(T.origin.values == o); h = np.flatnonzero(OO + 4 <= ORD[o])
    if len(h) < 1500: continue
    # D1: k so that the history's P25-P75 (around c) covers 50%
    ks = np.arange(0.8, 2.01, 0.05); cov = [np.mean((r[h] >= k * SPR[h, 1]) & (r[h] <= k * SPR[h, 3])) for k in ks]
    k1 = ks[int(np.argmin(np.abs(np.array(cov) - 0.5)))]; OFF["D1"][te] = k1 * SPR[te]; KLOG.append((o, len(h), round(float(k1), 2)))
    # D2: binned empirical residual quantiles (median forced to 0)
    db = np.quantile(dis[h], [0.2, 0.4, 0.6, 0.8]); qb = np.quantile(qiqr[h], [1 / 3, 2 / 3])
    bd_h, bq_h = np.digitize(dis[h], db), np.digitize(qiqr[h], qb); bd_t, bq_t = np.digitize(dis[te], db), np.digitize(qiqr[te], qb)
    for i, t in enumerate(te):
        sel = (bd_h == bd_t[i]) & (bq_h == bq_t[i])
        if sel.sum() < 150: sel = bd_h == bd_t[i]
        qq = np.quantile(r[h][sel], TAU); OFF["D2"][t] = qq - qq[2]
    # D3: LightGBM quantile regression of the residual
    P = np.zeros((len(te), 5))
    for j, tau in enumerate(TAU):
        m = lgb.LGBMRegressor(objective="quantile", alpha=tau, n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=100,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0).fit(X.iloc[h], r[h])
        P[:, j] = m.predict(X.iloc[te])
    P = np.sort(P, axis=1); OFF["D3"][te] = P - P[:, [2]]; OFF["D3m"][te] = P
print("time-honest k for the quantile-model spread (origin, history rows, k):", KLOG)
def quants(v, centre=lc): return np.expm1(centre[:, None] + OFF[v])
def score(idx):
    out = {}; r3c = np.log1p(T.R3.values[idx]); Q3 = np.expm1(r3c[:, None] + 1.25 * SPR[idx]); yy = y[idx]
    def pinball(Q): u = yy[:, None] - Q; return np.maximum(TAU * u, (TAU - 1) * u).sum() / (5 * yy.sum())
    def iscore(Q): lo, hi = Q[:, 0], Q[:, 4]; return ((hi - lo) + 10 * np.maximum(lo - yy, 0) + 10 * np.maximum(yy - hi, 0)).sum() / yy.sum()
    p3 = pinball(Q3); Q0 = quants("D0")[idx]; i0 = iscore(Q0)
    for v in OFF:
        Q = quants(v)[idx]
        out[v] = {"pinball / R3": pinball(Q) / p3, "P25-75 cover": np.mean((yy >= Q[:, 1]) & (yy <= Q[:, 3])), "P10-90 cover": np.mean((yy >= Q[:, 0]) & (yy <= Q[:, 4])),
                  "interval score / D0": iscore(Q) / i0, "median WAPE": np.abs(Q[:, 2] - yy).sum() / yy.sum(),
                  "P10-90 width (median, log)": float(np.median(np.log1p(Q[:, 4]) - np.log1p(Q[:, 0])))}
    return pd.DataFrame(out).T
REC = T.origin.values >= "2025-06"; HAS = np.isin(T.origin.values, [k[0] for k in KLOG])
for nm, idx in [("origins with a fitted variant (history >= 1,500 rows)", np.flatnonzero(HAS)), ("nine recent origins", np.flatnonzero(REC)), ("recent, 100 targets", np.flatnonzero(REC & T.target.values))]:
    print(f"\n######## {nm}: {len(idx)} item-origins"); print(score(idx).round(4).to_string())
print("\n######## P10-90 coverage by disagreement quintile (origins with a fitted variant)")
idx = np.flatnonzero(HAS); dq = pd.qcut(dis[idx], 5, labels=["d1 agree", "d2", "d3", "d4", "d5 disagree"])
rows = {}
for v in OFF:
    Q = quants(v)[idx]; inside = (y[idx] >= Q[:, 0]) & (y[idx] <= Q[:, 4]); rows[v] = pd.Series(inside).groupby(dq).mean()
print(pd.DataFrame(rows).round(3).T.to_string())
np.savez("data/dist_items.npz", **{v: OFF[v] for v in OFF}, origin=T.origin.values, style_color=T.style_color.values)
