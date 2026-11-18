"""Residual-distribution members (D148, user request): remove R3 (4 x mean of the last 3 calendar months) from each item's
4-month total, predict the residual's distribution parameters, sample residuals from that distribution and add R3 back.
  add : r = y - R3 (units),                      r ~ Normal(mu, sigma)  -> y = max(0, R3 + r)
  log : r = log1p(y) - log1p(R3),                r ~ Normal(mu, sigma)  -> y = expm1(log1p(R3) + r)
  logt: as log, r ~ Student-t(nu = 4, mu, s)     (heavier tails; s scaled so the t has the predicted spread)
mu = LightGBM L2 on r; sigma = LightGBM L2 on log|r - mu_oof| (mu_oof from 3 style-grouped folds inside the training set), corrected
by E[log|e|] of the unit-scale distribution. Lean + occasion-timing features of eda/20b plus log1p(R3). Time-honest: at origin o train
on origins o' with o' + 4 <= o (same as the quantile model). 1,000 draws per item; point = sample median, also the sample mean.
Writes data/q_backtest_rd_{add,log,logt}.parquet in the q_backtest layout (+ column 'mean')."""
import sys, importlib.util, time, numpy as np, pandas as pd, lightgbm as lgb, warnings
from scipy import stats
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
spec = importlib.util.spec_from_file_location("qm", "eda/20b_quantiles_lean.py"); qm = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(qm); sys.argv = _a
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
T1 = qm.T1.copy()
P = lambda o, k: str(pd.Period(o, "M") + k)
r3 = {}
for o in T1.origin.unique():
    cols = [P(o, -k) for k in range(3)]
    r3[o] = 4 * M.reindex(columns=cols).fillna(0).mean(axis=1)
T1["R3"] = [r3[o].get(s, 0.0) for o, s in zip(T1.origin, T1.style_color)]
T1["lr3"] = np.log1p(T1.R3)
X = qm.FULL + ["lr3"]
PAR = dict(qm.P) | {"n_jobs": 8}
NU = 4
C_NORM = -0.6352                                   # E[log|Z|], Z ~ N(0, 1)
C_T = float(np.mean(np.log(np.abs(stats.t(NU).rvs(2_000_000, random_state=1)))))   # E[log|t_nu|]
QOUT = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
rng = np.random.default_rng(0)
VAR = {"add": "add", "log": "log", "logt": "log"}


def resid(df, kind): return (df.ytot - df.R3).values if kind == "add" else (np.log1p(df.ytot) - df.lr3).values


def fit_mu_sigma(tr, te, kind, cfac):
    r = resid(tr, kind)
    mu_te = lgb.LGBMRegressor(objective="l2", **PAR).fit(tr[X], r).predict(te[X])
    sty = tr.style_color.str.split("_").str[0].values; fold = pd.Series(sty).map(lambda s: hash(s) % 3).values
    oof = np.zeros(len(tr))
    for k in range(3):
        m = fold == k
        oof[m] = lgb.LGBMRegressor(objective="l2", **PAR).fit(tr.loc[~m, X], r[~m]).predict(tr.loc[m, X])
    la = np.log(np.abs(r - oof) + 1e-3)
    ls = lgb.LGBMRegressor(objective="l2", **PAR).fit(tr[X], la).predict(te[X]) - cfac
    return mu_te, np.exp(ls)


rows = {v: [] for v in VAR}; t0 = time.time()
for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
    oo = pd.Period(o).ordinal; tr = T1[(T1.o_ord + 4 <= oo) & T1.ytot.notna()].reset_index(drop=True); te = T1[T1.origin == o]
    cache = {}
    for v, kind in VAR.items():
        cfac = C_T if v == "logt" else C_NORM
        mu, sd = fit_mu_sigma(tr, te, kind, cfac)
        if v == "logt": Z = stats.t(NU).rvs((len(te), 1000), random_state=rng)
        else: Z = rng.standard_normal((len(te), 1000))
        S = mu[:, None] + sd[:, None] * Z
        Y = np.maximum(0, te.R3.values[:, None] + S) if kind == "add" else np.maximum(0, np.expm1(te.lr3.values[:, None] + S))
        R = te[["origin", "style_color", "level", "age_m", "ytot", "R3"]].copy()
        for q in QOUT: R[f"q{int(q*100):02d}"] = np.quantile(Y, q, axis=1)
        R["mean"] = Y.mean(axis=1); R["mu"] = mu; R["sigma"] = sd
        rows[v].append(R)
    print(o, "train", len(tr), "items", len(te), round(time.time() - t0), "s", flush=True)
for v in VAR:
    D = pd.concat(rows[v]); D.to_parquet(f"data/q_backtest_rd_{v}.parquet")
    rec = D[D.origin >= "2025-06"]
    print(v, "recent WAPE (median, mean, R3):", [round((rec[c] - rec.ytot).abs().sum() / rec.ytot.sum(), 4) for c in ["q50", "mean", "R3"]],
          "| P10-P90 coverage", round(((rec.ytot >= rec.q10) & (rec.ytot <= rec.q90)).mean(), 3))
