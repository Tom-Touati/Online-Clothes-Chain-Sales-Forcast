"""D149 diagnostic: does a style's forecast error persist? Three-way (rules x quantile x NN) style-level log residual
log((sum actual + 1) / (sum forecast + 1)) per style x origin, correlated with the same style's residual at earlier origins
(lag 4 = the latest fully known window at the origin; lags 1-3 overlap). Also the item-level version and the within-style
(item minus style) part, and the share of styles whose sign repeats. Styles with >= 50 actual units in both windows."""
import sys, contextlib, io, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
T["style"] = T.style_color.str.split("_").str[0]; T["f"] = T.r_q_nn
S = T.groupby(["style", "origin"]).agg(y=("y", "sum"), f=("f", "sum"), n=("style_color", "size")).reset_index()
S["r"] = np.log((S.y + 1) / (S.f + 1)); S["o"] = pd.PeriodIndex(S.origin, freq="M").astype(int)
T["ri"] = np.log((T.y + 1) / (T.f + 1)); T["o"] = pd.PeriodIndex(T.origin, freq="M").astype(int)
T = T.merge(S[["style", "origin", "r"]].rename(columns={"r": "rs"}), on=["style", "origin"]); T["rw"] = T.ri - T.rs
def lagcorr(D, key, col, lag, mask_col="y"):
    A = D[[*key, "o", col, mask_col]]; B = A.copy(); B["o"] = B.o + lag
    M = A.merge(B, on=[*key, "o"], suffixes=("", "_lag")); M = M[(M[mask_col] >= 50) & (M[mask_col + "_lag"] >= 50)]
    w = np.sqrt(M[mask_col])
    c = np.corrcoef(M[col], M[col + "_lag"])[0, 1]; cw = np.cov(M[col], M[col + "_lag"], aweights=w)[0, 1] / np.sqrt(np.cov(M[col], aweights=w) * np.cov(M[col + "_lag"], aweights=w))
    sg = (np.sign(M[col]) == np.sign(M[col + "_lag"])).mean()
    b = np.polyfit(M[col + "_lag"], M[col], 1)[0]
    return len(M), c, cw, sg, b
print("lag | pairs | corr | corr (sqrt-units weighted) | same sign | slope")
for nm, D, key, col in [("style residual", S, ["style"], "r"), ("item residual", T, ["style_color"], "ri"), ("within-style part", T, ["style_color"], "rw")]:
    print("--", nm)
    for lag in [1, 2, 3, 4, 6, 8, 12]:
        n, c, cw, sg, b = lagcorr(D, key, col, lag)
        if n > 30: print(f"  {lag:2d} | {n:5d} | {c:+.3f} | {cw:+.3f} | {sg:.3f} | {b:+.3f}")
print("\nstyle residual spread (sd, styles >= 50 units):", round(S[S.y >= 50].r.std(), 3), "| item:", round(T[T.y >= 50].ri.std(), 3))
