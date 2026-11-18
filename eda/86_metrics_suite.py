"""Metric suite on 4-month item totals (D110). Same 13 origins and items as eda/77/79/85.
1) MASE (Hyndman): |error| / the item's in-sample error of the recency forecast R3 (last 3 months x 4) over every past 4-month
   window that ended by the origin; mean over items. Items with no past window (young) are excluded and counted.
2) Relative MAE (Davydenko & Fildes AvgRelMAE): geometric mean over items of (|e_sys| + 1) / (|e_ref| + 1), against two references:
   R3 (recency) and the rule agents (recency rate x rule multipliers). < 1 = better than the reference on a typical item.
3) MAE + |bias| (Vandeput's score): per origin (sum|e| + |sum e|) / sum actual, then averaged over origins; bias hurts as much as noise.
4) Pinball loss on P10/P25/P50/P75/P90: each system's median carries the quantile model's log spread around it, widened by
   k = 1.25 (the production k*); scaled by total actuals; skill = 1 - loss / loss(R3 with the same spread). Coverage of P25-P75, P10-P90."""
import numpy as np, pandas as pd, contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("eda/79_combo_eval.py", encoding="utf-8").read().split('C = ["rules"')[0])
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
T["R3"] = T.runrate
SY = {"R3": "Recency (last 3 months x4)", "rules": "Rule agents", "qmodel": "Quantile median", "lgb": "LightGBM", "nn": "Embedding NN",
      "lgb_nn": "LightGBM+NN", "q_nn": "Quantile x NN", "r_q": "Rules x quantile", "r_q_nn": "Rules x quantile x NN"}
# ---- in-sample recency error per item and past window
mons = [c for c in M.columns if c[:2] == "20"]; P = lambda o, k: str(pd.Period(o, "M") + k)
E = {}
for o in mons:
    need = [P(o, -k) for k in range(3)] + [P(o, h) for h in range(1, 5)]
    if not all(c in M.columns for c in need): continue
    h = M[[P(o, -k) for k in range(3)]]; a = M[[P(o, h) for h in range(1, 5)]].sum(axis=1)
    e = (4 * h.mean(axis=1) - a).abs(); E[o] = e[(h.sum(axis=1) > 0) | (a > 0)]
E = pd.DataFrame(E)                     # items x window origin
def scale(o, s):
    cols = [c for c in E.columns if pd.Period(c, "M") + 4 <= pd.Period(o, "M")]
    v = E.loc[s, cols].dropna() if s in E.index else pd.Series(dtype=float)
    return v.mean() if len(v) else np.nan
T["mase_scale"] = [scale(o, s) for o, s in zip(T.origin, T.style_color)]
# ---- pinball
q = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"])
TAUS = {"q10": 0.1, "q25": 0.25, "q50": 0.5, "q75": 0.75, "q90": 0.9}; K = 1.25
Qb = q.reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))
D = {c: np.log1p(Qb[c].values) - np.log1p(Qb.q50.values) for c in TAUS}
def pin(F, c):
    i = F.index; y = F.y.values; tot = 0; cov = {}
    Qs = {t: np.expm1(np.log1p(np.maximum(F[c].values, 0)) + K * D[t][i]) for t in TAUS}
    for t, tau in TAUS.items():
        u = y - Qs[t]; tot += np.maximum(tau * u, (tau - 1) * u).sum()
    return tot / (len(TAUS) * y.sum()), ((y >= Qs["q25"]) & (y <= Qs["q75"])).mean(), ((y >= Qs["q10"]) & (y <= Qs["q90"])).mean()
T = T.reset_index(drop=True)
def suite(F):
    out = {}; m = F.mase_scale.notna() & (F.mase_scale > 0)
    for c in SY:
        e = (F[c] - F.y).abs(); sp, c50, c80 = pin(F, c)
        per = F.assign(e=F[c] - F.y).groupby("origin").apply(lambda x: (x.e.abs().sum() + abs(x.e.sum())) / x.y.sum())
        out[SY[c]] = {"MASE": (e[m] / F.mase_scale[m]).mean(), "MdASE": (e[m] / F.mase_scale[m]).median(),
                      "relMAE vs R3": np.exp(np.mean(np.log((e + 1) / ((F.R3 - F.y).abs() + 1)))),
                      "relMAE vs rules": np.exp(np.mean(np.log((e + 1) / ((F.rules - F.y).abs() + 1)))),
                      "MAE%": e.sum() / F.y.sum(), "bias%": (F[c] - F.y).sum() / F.y.sum(), "MAE+|bias| (per origin)": per.mean(),
                      "pinball": sp, "P25-75 cov": c50, "P10-90 cov": c80}
    R = pd.DataFrame(out).T
    R.insert(R.columns.get_loc("pinball") + 1, "pinball skill vs R3", 1 - R.pinball / R.loc[SY["R3"], "pinball"])
    return R, int(m.sum())
REC_ = [o for o in ORIGINS if o >= "2025-06"]
for nm, F in [("NINE RECENT ORIGINS", T[T.origin.isin(REC_)]), ("ALL 13 ORIGINS", T), ("RECENT, THE 100 TARGETS", T[T.origin.isin(REC_) & T.target])]:
    R, nm_ = suite(F)
    print(f"\n######## {nm}: {len(F)} item-origins; MASE on {nm_} with a past recency window")
    print(R.round(3).to_string())
# bootstrap for the headline comparisons, recent origins
rng = np.random.default_rng(0); F = T[T.origin.isin(REC_)].reset_index(drop=True); F["mase_ok"] = F.mase_scale.notna() & (F.mase_scale > 0)
g = [x for _, x in F.groupby("origin")]
def stat(s, c):
    e = (s[c] - s.y).abs(); ok = s.mase_ok
    return np.array([(e[ok] / s.mase_scale[ok]).mean(), np.exp(np.mean(np.log((e + 1) / ((s.rules - s.y).abs() + 1)))), pin(s, c)[0]])
print("\n== recent origins, 95% bootstrap of the difference (MASE | relMAE vs rules | pinball): system minus comparison")
B = [pd.concat([x.iloc[rng.integers(0, len(x), len(x))] for x in g]) for _ in range(500)]
for a, b in [("r_q_nn", "q_nn"), ("r_q_nn", "qmodel"), ("r_q_nn", "rules"), ("qmodel", "R3")]:
    dd = np.array([stat(s, a) - stat(s, b) for s in B])
    print(f"  {a} - {b}: " + " | ".join(f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(np.percentile(dd, 2.5, axis=0), np.percentile(dd, 97.5, axis=0))))
