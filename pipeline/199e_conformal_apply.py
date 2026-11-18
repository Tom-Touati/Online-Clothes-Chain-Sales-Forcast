"""D209 (fork): apply the D206 F6 conformal adjustment ("seg_tails", pipeline/199c) to a deliverable-origin quantile file. Calibration
set = every backtest row of the same member whose outcome is known by the deliverable origin (all 13 origins for June 2026). Same
rule as the backtest: log-space residual quantile per tau and per segment (age < 6 / 6-12 / 12+ months x level tercile within the
origin), shrunk to the global one with weight n / (n + 200); the median is not shifted directly; quantiles are sorted afterwards.
Usage: python pipeline/199e_conformal_apply.py backtest=<q_backtest file> target=<deliverable file> out=<file>"""
import sys, numpy as np, pandas as pd
KW = dict(a.split("=", 1) for a in sys.argv[1:]); QC = ["q10", "q25", "q40", "q50", "q60", "q75", "q90"]; TAU = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
def seg(Q): return pd.cut(Q.age_m, [-1, 6, 12, 1e9], labels=False).astype(int) * 3 + Q.groupby("origin")["level"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False)).astype(int)
B = pd.read_parquet(KW["backtest"]); B = B[B.ytot.notna()].reset_index(drop=True); D = pd.read_parquet(KW["target"]).reset_index(drop=True)
o = D.origin.iloc[0]; B = B[B.origin.map(lambda x: pd.Period(x, "M").ordinal + 4 <= pd.Period(o, "M").ordinal)].reset_index(drop=True)
B["seg"] = seg(B); D["seg"] = seg(D)
S = np.log1p(B.ytot.values)[:, None] - np.log1p(B[QC].values)
g = np.array([np.quantile(S[:, j], t) for j, t in enumerate(TAU)]); adj = {}
for sg in range(9):
    c = (B.seg == sg).values; n = c.sum(); w = n / (n + 200)
    adj[sg] = w * (np.array([np.quantile(S[c, j], t) for j, t in enumerate(TAU)]) if n >= 30 else g) + (1 - w) * g
d = np.stack([adj[s] for s in D.seg.values]); d[:, 3] = 0
D[QC] = np.clip(np.sort(np.expm1(np.log1p(D[QC].values) + d), axis=1), 0, None)
for q, t in [("q65", 1 / 3), ("q70", 2 / 3)]: D[q] = np.expm1((1 - t) * np.log1p(D.q60) + t * np.log1p(D.q75))
D.drop(columns="seg").to_parquet(KW["out"], index=False)
print("calibration rows", len(B), "origins", B.origin.nunique(), "| global shifts (log):", np.round(g, 3).tolist(), "| saved", KW["out"])
