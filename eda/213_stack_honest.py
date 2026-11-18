"""Fork 2 D387 (user: "use tabpfn as initial model, then stack the ensemble afterwards"): a time-honest stack. For each origin o, non-negative
weights summing to 1 that minimise the summed absolute error on the member forecasts of past origins whose window has closed (o' + 4 <= o);
equal weights while fewer than MINO such origins exist. Scored with the submitted correction multiplier F/F0, WAPE / R3."""
import sys, numpy as np, pandas as pd
from scipy.optimize import minimize
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); K = D[["origin", "style_color"]]
D["huber"] = K.merge(pd.read_parquet("data/dec_base_ecnsnnhub.parquet")[["origin", "style_color", "qdyn"]], how="left").qdyn.values
top = D[["rules", "qdyn", "nn", "pf2c", "pf2d32"]].max(1).values
def rc(fn, c):
    v = K.merge(pd.read_parquet(fn)[["origin", "style_color", c]], how="left")[c].astype(float).values; return np.minimum(np.where(np.isnan(v), D.R3, v), 3 * top)
D["biz"] = rc("data/rc_recsys_preds_x_biz.parquet", "f_biz_gbm"); D["rcd"] = rc("data/rc_recsys_preds.parquet", "rc_deepfm")
D["pfinit"] = K.merge(pd.read_parquet("data/q_backtest_ls_huber_pfinit.parquet")[["origin", "style_color", "q50"]], how="left").q50.values
y = D.y.values; ar = np.abs(D.R3 - y).values; rec = (D.origin >= "2025-06").values; al = np.ones(len(D), bool); er = ~rec; ho = (D.origin >= "2025-10").values; corr = (D.F / D.F0).values
w = lambda f, m: round(float(np.abs(f - y)[m].sum() / ar[m].sum()), 4)
OR = sorted(D.origin.unique()); oo = {o: pd.Period(o).ordinal for o in OR}
def stack(cols, mino=3):
    X = D[cols].values; out = X.mean(1).copy(); W = {}
    for o in OR:
        past = D.origin.map(lambda q: oo[q] + 4 <= oo[o]).values; te = (D.origin == o).values
        if len(set(D.origin[past])) < mino: W[o] = None; continue
        Xp, yp = X[past], y[past]; f = lambda a: np.abs(Xp @ (np.abs(a) / np.abs(a).sum()) - yp).sum()
        a = minimize(f, np.ones(len(cols)) / len(cols), method="Powell").x; a = np.abs(a) / np.abs(a).sum(); W[o] = a; out[te] = X[te] @ a
    return out, W
for name, cols in [("five (Huber)", ["rules", "huber", "nn", "pf2c", "pf2d32"]), ("five + biz", ["rules", "huber", "nn", "pf2c", "pf2d32", "biz"]),
                   ("tree-on-TabPFN in place of Huber", ["rules", "pfinit", "nn", "pf2c", "pf2d32"]), ("six + tree-on-TabPFN + biz", ["rules", "huber", "pfinit", "nn", "pf2c", "pf2d32", "biz"])]:
    s, W = stack(cols); e = D[cols].values.mean(1)
    last = [x for x in W.values() if x is not None][-1]
    print(f"{name:34s} mean: {w(e*corr,rec)} {w(e*corr,al)} {w(e*corr,ho)} {w(e,er)} | stack: {w(s*corr,rec)} {w(s*corr,al)} {w(s*corr,ho)} {w(s,er)} | before corr rec: mean {w(e,rec)} stack {w(s,rec)} | last", dict(zip(cols, np.round(last, 2))))
