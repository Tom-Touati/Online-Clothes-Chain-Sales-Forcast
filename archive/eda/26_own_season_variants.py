"""Variants of the own-season correction, all fitted on earlier origins only (origin + 4 <= o) and scored on later origins.
V1 shift = b*gap (no intercept) | V2 shift = a + b*gap (full regression) | V3 shift = b*(gap - mean gap of past) (re-ranks, keeps the level).
Shift is applied to every quantile of items that have their own last-year history; others unchanged."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 230)
QS = np.array([0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]); QC = ["q10", "q25", "q40", "q50", "q60", "q75", "q90"]
Q = pd.read_parquet("data/q_backtest_own.parquet")
def q_at(Lq, tau):
    return np.array([np.interp(tau, QS, r) for r in Lq])
def met(f, a):
    ape = np.abs(f - a) / np.clip(a, 1, None); return dict(n=len(a), wape=np.abs(f - a).sum() / a.sum(), bias=f.sum() / a.sum() - 1, w20=(ape <= .2).mean(), w50=(ape <= .5).mean())
H = Q[Q.z_own.notna()]
out = []
for o in sorted(Q.origin.unique()):
    oo = pd.Period(o).ordinal; past = H[H.oo + 4 <= oo]
    if past.origin.nunique() < 2: continue
    b, a = np.polyfit(past.gap, past.res, 1); gbar = past.gap.mean()
    g = Q[Q.origin == o]; has = g.z_own.notna().values; gap = g.gap.fillna(0).values
    shifts = {"current": np.zeros(len(g)), "V1 b*gap": np.where(has, b * gap, 0), "V2 a+b*gap": np.where(has, a + b * gap, 0), "V3 b*(gap-mean)": np.where(has, b * (gap - gbar), 0)}
    for nm, sh in shifts.items():
        L = np.log((g[QC].values + 1) * np.exp(sh)[:, None])
        for tau in (0.5, 0.61):
            f = np.expm1(q_at(L, tau))
            out.append(dict(origin=o, version=nm, tau=tau, seg="all", **met(f, g.ytot.values)))
            out.append(dict(origin=o, version=nm, tau=tau, seg="own history", **met(f[has], g.ytot.values[has])))
R = pd.DataFrame(out)
for seg in ["all", "own history"]:
    x = R[R.seg == seg]; t = x.groupby(["tau", "version"])[["wape", "bias", "w20", "w50"]].mean()
    base = x[x.version == "current"].set_index(["tau", "origin"]).wape
    t["origins_better"] = [int((x[(x.version == v) & (x.tau == ta)].set_index(["tau", "origin"]).wape < base.reindex(x[(x.version == v) & (x.tau == ta)].set_index(["tau", "origin"]).index)).sum()) for ta, v in t.index]
    print(f"== {seg} ({R.origin.nunique()} later origins)"); print(t.round(3).to_string())
REP = [str(p) for p in pd.period_range("2025-06", "2026-02", freq="M")]
print("\nfull-regression coefficients on the representative window:", np.round(np.polyfit(H[H.origin.isin(REP)].gap, H[H.origin.isin(REP)].res, 1), 3).tolist(), "(slope, intercept); mean gap", round(H[H.origin.isin(REP)].gap.mean(), 3))
