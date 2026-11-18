"""D206 F6 (fork, M5 uncertainty track): conformalised quantiles. For a quantile backtest file (q10 .. q90), at each test origin o the
calibration set is every earlier backtest row whose 4-month outcome is known by o (origin o' with o' + 4 <= o). In log space,
score s = log1p(y) - log1p(q_tau); the adjustment is the empirical tau-quantile of s (conformalised quantile regression, per tau),
  global   one adjustment per tau
  seg      per segment (age < 6 / 6-12 / 12+ months x level tercile at the origin), shrunk to the global one with weight n / (n + 200)
Variants adjust only the outer quantiles ("tails") or all of them including the median ("all", i.e. a per-segment bias multiplier,
the M5 'magic multiplier' done time-honestly). Scored on the nine recent origins (the first has one calibration origin):
  pinball (mean over the 7 quantiles, units, / sum y), P10-P90 coverage (nominal 80%), P25-P75 coverage (50%), P50 WAPE.
Usage: python eda/199c_conformal.py name=path [...]; writes data/q_backtest_fkconf_<name>_<variant>.parquet."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
FILES = dict(a.split("=", 1) for a in sys.argv[1:]); QC = ["q10", "q25", "q40", "q50", "q60", "q75", "q90"]; TAU = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
def score(D, m):
    D = D[m]; y = D.ytot.values; pin = np.mean([np.sum(np.maximum(t * (y - D[c].values), (t - 1) * (y - D[c].values))) for c, t in zip(QC, TAU)]) / y.sum()
    return {"pinball/y": round(pin, 4), "cov80": round(((y >= D.q10) & (y <= D.q90)).mean(), 3), "cov50": round(((y >= D.q25) & (y <= D.q75)).mean(), 3), "WAPE": round(np.abs(D.q50 - y).sum() / y.sum(), 4)}
rows = []
for name, path in FILES.items():
    Q = pd.read_parquet(path).reset_index(drop=True); Q = Q[Q.ytot.notna()].reset_index(drop=True); Q["o_ord"] = Q.origin.map(lambda o: pd.Period(o, "M").ordinal)
    Q["seg"] = pd.cut(Q.age_m, [-1, 6, 12, 1e9], labels=False).astype(int) * 3 + Q.groupby("origin")["level"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False)).astype(int)
    L = np.log1p(Q[QC].values); ly = np.log1p(Q.ytot.values)[:, None]; S = ly - L
    out = {v: Q[QC].values.copy() for v in ["global_tails", "seg_tails", "global_all", "seg_all"]}
    for o in sorted(Q.origin.unique()):
        te = (Q.origin == o).values; cal = (Q.o_ord + 4 <= pd.Period(o, "M").ordinal).values
        if cal.sum() < 200: continue
        g = np.array([np.quantile(S[cal, j], t) for j, t in enumerate(TAU)])
        dseg = np.zeros((te.sum(), len(TAU)))
        for k, sg in enumerate(Q.seg.values[te]):
            c = cal & (Q.seg.values == sg); n = c.sum(); w = n / (n + 200)
            dseg[k] = w * (np.array([np.quantile(S[c, j], t) for j, t in enumerate(TAU)]) if n >= 30 else g) + (1 - w) * g
        for v, d in [("global", np.broadcast_to(g, dseg.shape)), ("seg", dseg)]:
            for scope in ["tails", "all"]:
                dd = d.copy()
                if scope == "tails": dd[:, 3] = 0
                out[f"{v}_{scope}"][te] = np.sort(np.expm1(L[te] + dd), axis=1)
    REC = (Q.origin >= "2025-06").values
    rows.append(dict(member=name, variant="raw", **score(Q, REC)))
    for v, A in out.items():
        D = Q.copy(); D[QC] = np.clip(A, 0, None); rows.append(dict(member=name, variant=v, **score(D, REC)))
        D.drop(columns=["o_ord", "seg"]).to_parquet(f"data/q_backtest_fkconf_{name}_{v}.parquet")
print(pd.DataFrame(rows).to_string(index=False))
