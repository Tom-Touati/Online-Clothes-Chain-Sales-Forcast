"""Is the NN blend's WAPE gain distinguishable from noise? Paired bootstrap over items (an item keeps all its origins together),
1,000 resamples, for: monthly blend vs LightGBM, re-centred production model vs quantile model (13 and 9 representative origins).
Statistic = mean over origins of WAPE(candidate) - WAPE(base)."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); rng = np.random.default_rng(0)
NM = pd.read_parquet("data/nn_backtest_preds_B13x6.parquet"); B = pd.read_parquet("data/best_backtest_preds.parquet")
X = B.merge(NM[["origin", "style_color", "h", "q50"]], on=["origin", "style_color", "h"]).groupby(["origin", "style_color"])[["pred", "q50", "y"]].sum().reset_index()
Q = pd.read_parquet("data/q_backtest.parquet")[["origin", "style_color", "q50", "ytot"]].rename(columns={"q50": "qm"}).merge(X, on=["origin", "style_color"])
Q["blend_m"] = 0.5 * Q.pred + 0.5 * Q.q50; Q["blend_p"] = np.expm1(0.5 * np.log1p(Q.qm) + 0.5 * np.log1p(Q.q50))
items = Q.style_color.unique(); ii = {s: i for i, s in enumerate(items)}; Q["ii"] = Q.style_color.map(ii)
for name, base, cand, act, origins in [("monthly blend vs LightGBM, 13 origins", "pred", "blend_m", "y", None),
                                       ("production re-centred vs quantile model, 13 origins", "qm", "blend_p", "ytot", None),
                                       ("production re-centred vs quantile model, 9 representative", "qm", "blend_p", "ytot", "2025-06")]:
    D = Q if origins is None else Q[Q.origin >= origins]
    O = sorted(D.origin.unique()); n = len(items)
    # per origin x item arrays of |error| and actual, for fast resampling
    E = {c: np.zeros((len(O), n)) for c in ["b", "c", "a"]}
    for k, o in enumerate(O):
        d = D[D.origin == o]; E["b"][k, d.ii] = np.abs(d[base] - d[act]); E["c"][k, d.ii] = np.abs(d[cand] - d[act]); E["a"][k, d.ii] = d[act]
    def stat(w): return np.mean((E["c"] @ w - E["b"] @ w) / (E["a"] @ w))
    obs = stat(np.ones(n)); bs = np.array([stat(np.bincount(rng.integers(0, n, n), minlength=n).astype(float)) for _ in range(1000)])
    print(f"{name}: diff {obs:+.4f} | 95% CI [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}] | P(diff >= 0) = {(bs >= 0).mean():.3f}")
