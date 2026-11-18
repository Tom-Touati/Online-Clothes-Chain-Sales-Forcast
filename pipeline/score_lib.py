"""Standard scoring table (user rule, D109-D111): every metric as a ratio to R3 (last 3 months x 4), with R3 + rules (the rule
agents) as a row, win rate vs R3 and vs R3 + rules, MAE + |bias| per origin, pinball (each centre carries the quantile model's
log spread widened by k = 1.25), and bias for context. table(F, cols, D) where D maps quantile names to log-spread arrays
aligned with F's positional index. boot(F, a, b) = 95% item-bootstrap interval of WAPE(a) - WAPE(b) within origins."""
import numpy as np, pandas as pd
TAUS = {"q10": 0.1, "q25": 0.25, "q50": 0.5, "q75": 0.75, "q90": 0.9}; K = 1.25


def pin(F, c, D):
    i = F.index.values; y = F.y.values; tot = 0
    for t, tau in TAUS.items():
        Q = np.expm1(np.log1p(np.maximum(F[c].values, 0)) + K * D[t][i]); u = y - Q; tot += np.maximum(tau * u, (tau - 1) * u).sum()
    return tot / (len(TAUS) * y.sum())


def mb(F, c):
    e = F[c] - F.y
    return (e.abs().groupby(F.origin).sum() + e.groupby(F.origin).sum().abs()).div(F.y.groupby(F.origin).sum()).mean()


def table(F, cols, D, names=None, sim=None):
    e3 = (F.R3 - F.y).abs(); eru = (F.rules - F.y).abs(); m3 = mb(F, "R3"); p3 = pin(F, "R3", D); out = {}
    for c in ["rules"] + [c for c in cols if c != "rules"]:
        e = (F[c] - F.y).abs()
        r = {"WAPE / R3": e.sum() / e3.sum(), "relMAE vs R3": np.exp(np.mean(np.log((e + 1) / (e3 + 1)))),
             "win vs R3": ((e < e3) + 0.5 * (e == e3)).mean(), "win vs R3+rules": ((e < eru) + 0.5 * (e == eru)).mean(),
             "(MAE+|bias|) / R3": mb(F, c) / m3, "pinball / R3": pin(F, c, D) / p3, "bias": F[c].sum() / F.y.sum() - 1}
        if sim is not None: r["sim loss / R3"] = sim.get(c, np.nan)
        out["R3 + rules (rule agents)" if c == "rules" else (names or {}).get(c, c)] = r
    print(f"   R3 itself: WAPE {e3.sum() / F.y.sum():.3f} | MAE+|bias| {m3:.3f} | pinball {p3:.3f} | bias {F.R3.sum() / F.y.sum() - 1:+.3f} | {len(F)} item-origins")
    return pd.DataFrame(out).T


def boot(F, a, b, B=400, seed=0):
    rng = np.random.default_rng(seed); g = [x for _, x in F.groupby("origin")]; d = []
    for _ in range(B):
        s = pd.concat([x.iloc[rng.integers(0, len(x), len(x))] for x in g])
        d.append(((s[a] - s.y).abs().sum() - (s[b] - s.y).abs().sum()) / s.y.sum())
    return np.percentile(d, [2.5, 97.5])
