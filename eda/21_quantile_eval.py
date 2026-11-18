"""Evaluates the quantile backtest (data/q_backtest.parquet).
A calibration | B median vs current point model | C answer quantile chosen from bias on earlier origins |
D IQR width calibration on earlier origins | E confidence levels from the IQR vs the +-20% / +-50% bands.
Every choice for origin o uses only origins whose 4-month actuals were known at o (origin + 4 <= o)."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
QS = np.array([0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]); QC = ["q10", "q25", "q40", "q50", "q60", "q75", "q90"]
Q = pd.read_parquet("data/q_backtest.parquet"); T = set(pd.read_csv("data/targets.csv").style_color)
Q["oo"] = Q.origin.map(lambda s: pd.Period(s, freq="M").ordinal); Q["a"] = Q.ytot
L = np.log1p(Q[QC].values); LA = np.log1p(Q.a.values)

def q_at(Lq, tau, k=1.0):
    """quantile tau (array or scalar) from log-space quantiles; spread around the median scaled by k; linear extrapolation beyond 0.10/0.90"""
    med = Lq[:, 3:4]; Ls = med + k * (Lq - med)
    tau = np.broadcast_to(np.asarray(tau, float), (len(Lq),))
    out = np.empty(len(Lq))
    for i in range(len(Lq)):
        out[i] = np.interp(tau[i], QS, Ls[i]) if QS[0] <= tau[i] <= QS[-1] else (Ls[i, 0] + (tau[i] - QS[0]) * (Ls[i, 1] - Ls[i, 0]) / (QS[1] - QS[0]) if tau[i] < QS[0] else Ls[i, -1] + (tau[i] - QS[-1]) * (Ls[i, -1] - Ls[i, -2]) / (QS[-1] - QS[-2]))
    return out
def metrics(f, a):
    ape = np.abs(f - a) / np.clip(a, 1, None)
    return dict(wape=np.abs(f - a).sum() / a.sum(), bias=f.sum() / a.sum() - 1, w20=(ape <= .2).mean(), w50=(ape <= .5).mean())

# ---- A calibration (pooled and by origin)
below = (LA[:, None] <= L).mean(axis=0)
print("== A calibration: share of actual totals at or below each predicted quantile (pooled, 13 origins)")
print(pd.Series(below, index=QC).round(3).to_dict())
cov = lambda lo, hi: ((LA >= L[:, lo]) & (LA <= L[:, hi]))
Q["in_iqr"] = cov(1, 5); Q["in_80"] = cov(0, 6)
print("coverage [P25,P75] nominal 50%:", round(Q.in_iqr.mean(), 3), "| [P10,P90] nominal 80%:", round(Q.in_80.mean(), 3))
print("by origin:"); print(Q.groupby("origin").agg(iqr=("in_iqr", "mean"), p80=("in_80", "mean"), below_p50=("a", lambda s: (np.log1p(s) <= np.log1p(Q.loc[s.index, "q50"])).mean())).round(2).T.to_string())

# ---- B point accuracy: quantile median vs current point model (same items)
res = []
for o, g in Q.groupby("origin"):
    res.append(dict(origin=o, model="P50 of quantile model", **metrics(g.q50.values, g.a.values)))
    res.append(dict(origin=o, model="current point model", **metrics(g.point.values, g.a.values)))
B = pd.DataFrame(res); print("\n== B point accuracy on 4-month totals, mean over 13 origins")
print(B.groupby("model")[["wape", "bias", "w20", "w50"]].agg(["mean", "std"]).round(3).to_string())
w = B.pivot(index="origin", columns="model", values="wape"); print("origins where P50 beats the point model:", int((w["P50 of quantile model"] < w["current point model"]).sum()), "of 13")

# ---- C answer quantile chosen from bias on earlier origins
grid = np.round(np.arange(0.30, 0.701, 0.01), 2)
cache = {t: np.expm1(q_at(L, t)) for t in grid}
out = []
for o in sorted(Q.origin.unique()):
    oo = pd.Period(o).ordinal; past = (Q.oo + 4 <= oo).values; now = (Q.origin == o).values
    if Q[past].origin.nunique() < 2: continue
    a_p = Q.a.values[past]
    bias_t = np.array([cache[t][past].sum() / a_p.sum() - 1 for t in grid]); t_bias = grid[np.argmin(np.abs(bias_t))]
    hit_t = np.array([(np.abs(cache[t][past] - a_p) / np.clip(a_p, 1, None) <= .2).mean() for t in grid]); t_hit = grid[np.argmax(hit_t)]
    a_n = Q.a.values[now]
    for lab, t in [("P50", 0.50), ("quantile with zero past bias", t_bias), ("quantile with best past +-20% hit rate", t_hit)]:
        out.append(dict(origin=o, rule=lab, tau=t, **metrics(cache[t][now], a_n)))
C = pd.DataFrame(out)
print(f"\n== C answer quantile chosen on earlier origins, scored on {C.origin.nunique()} later origins")
print(C.groupby("rule")[["tau", "wape", "bias", "w20", "w50"]].mean().round(3).to_string())
print("tau chosen by origin (zero bias):", C[C.rule.str.startswith("quantile with zero")].set_index("origin").tau.to_dict())
print("tau chosen by origin (best hit):", C[C.rule.str.startswith("quantile with best")].set_index("origin").tau.to_dict())
print("bias by origin:"); print(C.pivot(index="origin", columns="rule", values="bias").round(3).to_string())
allb = {t: cache[t].sum() / Q.a.sum() - 1 for t in grid}; allh = {t: (np.abs(cache[t] - Q.a.values) / np.clip(Q.a.values, 1, None) <= .2).mean() for t in grid}
print("pooled over all 13 origins (in-sample view): tau -> bias / within20:", {t: (round(allb[t], 3), round(allh[t], 3)) for t in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]})

# ---- D + E: IQR scale calibrated on earlier origins, then confidence from the IQR
def conf_from_iqr(f, p25, p75):
    one = (p25 >= f / 1.2) & (p75 <= f / 0.8)
    two = (p25 >= f / 1.5) & (p75 <= f / 0.5)
    return np.where(one, 1, np.where(two, 2, 3))
ks = np.round(np.arange(0.5, 2.51, 0.05), 2)
rows = []
for o in sorted(Q.origin.unique()):
    oo = pd.Period(o).ordinal; past = (Q.oo + 4 <= oo).values; now = (Q.origin == o).values
    if Q[past].origin.nunique() < 2: continue
    cov_k = np.array([((LA[past] >= q_at(L[past], .25, k)) & (LA[past] <= q_at(L[past], .75, k))).mean() for k in ks]); k = ks[np.argmin(np.abs(cov_k - 0.5))]
    f = Q.q50.values[now]; a = Q.a.values[now]
    for lab, kk in [("raw IQR", 1.0), ("calibrated IQR", k)]:
        p25 = np.expm1(q_at(L[now], .25, kk)); p75 = np.expm1(q_at(L[now], .75, kk))
        c = conf_from_iqr(f, p25, p75); ape = np.abs(f - a) / np.clip(a, 1, None)
        rows.append(pd.DataFrame(dict(origin=o, version=lab, k=kk, conf=c, w20=ape <= .2, w50=ape <= .5, in_iqr=(a >= p25) & (a <= p75), ratio=p75 / np.clip(p25, 1, None), sc=Q.style_color.values[now], level=Q.level.values[now])))
E = pd.concat(rows)
print(f"\n== D IQR scale k chosen on earlier origins (coverage of [P25,P75] -> 50%), scored on later origins")
print(E.groupby("version").agg(k_mean=("k", "mean"), iqr_coverage=("in_iqr", "mean")).round(3).to_string())
print("k by origin:", E[E.version == "calibrated IQR"].groupby("origin").k.first().to_dict())
print("\n== E confidence from the IQR: level 1 if [P25,P75] sits inside the +-20% band around the answer, 2 if inside the +-50% band, else 3")
for v, g in E.groupby("version"):
    print(f"-- {v}"); print(g.groupby("conf").agg(n=("w20", "size"), share=("w20", lambda s: len(s) / len(g)), within20=("w20", "mean"), within50=("w50", "mean")).round(3).to_string())
    tg = g[g.sc.isin(T)]; print("   targets only:", tg.groupby("conf").agg(n=("w20", "size"), within20=("w20", "mean"), within50=("w50", "mean")).round(3).to_dict("index"))
g = E[E.version == "calibrated IQR"]
print("calibrated IQR: within20 / within50 by decile of P75/P25:"); print(g.groupby(pd.qcut(g.ratio, 10, labels=False)).agg(ratio=("ratio", "median"), within20=("w20", "mean"), within50=("w50", "mean")).round(2).T.to_string())
E.to_parquet("data/q_conf_backtest.parquet")
