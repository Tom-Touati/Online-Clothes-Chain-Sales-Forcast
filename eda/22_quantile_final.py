"""Quantile-based answer and confidence, and the final Jul-Oct 2026 forecast.
1 Representative origins: the backtest folds whose training set is at least as large as at the June 2025 origin (2025-06 .. 2026-02).
  Early folds had 2.5k-4.2k training rows and over-forecast; the final fit has far more (see output).
2 Answer quantile tau*: the quantile with zero aggregate bias on the representative origins.
3 IQR scale k*: widens [P25,P75] around the median so it covers 50% of actuals on the representative origins.
4 Confidence: 1 if the calibrated [P25,P75] lies inside the +-20% band around the answer, 2 if inside the +-50% band, else 3.
5 Final: refit on all origins with a known 4-month total, score the June 2026 origin, split totals into months with the point model's monthly profile,
  then the same judgement adjustments as before (campaign uplift, World Cup rule, out-of-stock item)."""
import pandas as pd, numpy as np, warnings, importlib.util
warnings.filterwarnings("ignore"); pd.set_option("display.width", 230)
spec = importlib.util.spec_from_file_location("q20", "eda/20_quantiles.py"); q20 = importlib.util.module_from_spec(spec); spec.loader.exec_module(q20)
QS = np.array(q20.QS); QC = ["q10", "q25", "q40", "q50", "q60", "q75", "q90"]
def q_at(Lq, tau, k=1.0):
    med = Lq[:, 3:4]; Ls = med + k * (Lq - med); out = np.empty(len(Lq))
    for i in range(len(Lq)):
        t = tau if np.isscalar(tau) else tau[i]
        out[i] = np.interp(t, QS, Ls[i]) if QS[0] <= t <= QS[-1] else (Ls[i, 0] + (t - QS[0]) * (Ls[i, 1] - Ls[i, 0]) / (QS[1] - QS[0]) if t < QS[0] else Ls[i, -1] + (t - QS[-1]) * (Ls[i, -1] - Ls[i, -2]) / (QS[-1] - QS[-2]))
    return out
def met(f, a):
    ape = np.abs(f - a) / np.clip(a, 1, None); return dict(n=len(a), wape=np.abs(f - a).sum() / a.sum(), bias=f.sum() / a.sum() - 1, w20=(ape <= .2).mean(), w50=(ape <= .5).mean())
def conf_band(f, p25, p75): return np.where((p25 >= f / 1.2) & (p75 <= f / 0.8), 1, np.where((p25 >= f / 1.5) & (p75 <= f / 0.5), 2, 3))

Q = pd.read_parquet("data/q_backtest.parquet"); T = pd.read_csv("data/targets.csv").style_color.tolist()
REP = [str(p) for p in pd.period_range("2025-06", "2026-02", freq="M")]
R = Q[Q.origin.isin(REP)].copy(); L = np.log1p(R[QC].values); A = R.ytot.values
print("== 1 representative origins:", REP[0], "to", REP[-1], f"({len(REP)} origins, {len(R)} item forecasts)")
rows = []
for o, g in R.groupby("origin"):
    rows += [dict(origin=o, model="P50", **met(g.q50.values, g.ytot.values)), dict(origin=o, model="current point model", **met(g.point.values, g.ytot.values)),
             dict(origin=o, model="average of the two", **met((g.q50.values + g.point.values) / 2, g.ytot.values))]
B = pd.DataFrame(rows); print(B.groupby("model")[["wape", "bias", "w20", "w50"]].agg(["mean", "std"]).round(3).to_string())
w = B.pivot(index="origin", columns="model", values="wape"); print("P50 beats the point model in", int((w["P50"] < w["current point model"]).sum()), "of", len(w), "origins; by origin:", w.round(3).to_dict("index"))

grid = np.round(np.arange(0.40, 0.701, 0.01), 2); bias = {}
for t in grid: f = np.expm1(q_at(L, t)); bias[t] = f.sum() / A.sum() - 1
tau = min(grid, key=lambda t: abs(bias[t]))
print("\n== 2 aggregate bias by answer quantile on representative origins:", {t: round(bias[t], 3) for t in grid[::2]})
print("tau* =", tau, "| per-origin bias at tau*:", {o: round(met(np.expm1(q_at(np.log1p(g[QC].values), tau)), g.ytot.values)["bias"], 3) for o, g in R.groupby("origin")})
FT = np.expm1(q_at(L, tau)); print("metrics at P50:", {k: round(v, 3) for k, v in met(R.q50.values, A).items()}, "| at tau*:", {k: round(v, 3) for k, v in met(FT, A).items()})
J = R[R.origin == "2025-06"]; print("June 2025 origin, bias at P50:", round(met(J.q50.values, J.ytot.values)["bias"], 3), "| at tau*:", round(met(np.expm1(q_at(np.log1p(J[QC].values), tau)), J.ytot.values)["bias"], 3))

ks = np.round(np.arange(0.8, 2.51, 0.05), 2)
cov = {k: ((np.log1p(A) >= q_at(L, .25, k)) & (np.log1p(A) <= q_at(L, .75, k))).mean() for k in ks}
k = min(ks, key=lambda x: abs(cov[x] - 0.5))
cov80 = ((np.log1p(A) >= q_at(L, .10, k)) & (np.log1p(A) <= q_at(L, .90, k))).mean()
print("\n== 3 IQR coverage raw:", round(cov[1.0], 3), "| k* =", k, "-> coverage", round(cov[k], 3), "| [P10,P90] coverage at k*:", round(cov80, 3), "(nominal 0.80)")
P25 = np.expm1(q_at(L, .25, k)); P75 = np.expm1(q_at(L, .75, k)); C = conf_band(FT, P25, P75)
ape = np.abs(FT - A) / np.clip(A, 1, None)
S = pd.DataFrame(dict(conf=C, w20=ape <= .2, w50=ape <= .5, tgt=R.style_color.isin(T).values))
print("== 4 confidence from the calibrated IQR around the tau* answer (representative origins; tau* and k* were fitted on these same origins)")
print(S.groupby("conf").agg(n=("w20", "size"), share=("w20", lambda s: len(s) / len(S)), within20=("w20", "mean"), within50=("w50", "mean"), off_more_than_50=("w50", lambda s: 1 - s.mean())).round(3).to_string())
print("targets only:", S[S.tgt].groupby("conf").agg(n=("w20", "size"), within20=("w20", "mean"), within50=("w50", "mean")).round(3).to_dict("index"))

# ---- 5 final forecast
T1 = q20.T1; O = pd.Period("2026-06").ordinal
tr = T1[(T1.o_ord + 4 <= O) & T1.ytot.notna()]; te = T1[T1.origin == "2026-06"]
print("\n== 5 final fit: training rows", len(tr), "from", tr.origin.nunique(), "origins (largest backtest fold had 11,241)")
zq = q20.fit_q(tr, te); lv = 4 * te.level.values + 1
Lf = np.log1p(np.clip(lv[:, None] * np.exp(zq) - 1, 0, None))
QF = pd.DataFrame(np.expm1(Lf), index=te.style_color.values, columns=QC)
QF["answer"] = np.expm1(q_at(Lf, tau)); QF["p25c"] = np.expm1(q_at(Lf, .25, k)); QF["p75c"] = np.expm1(q_at(Lf, .75, k))
QF["p10c"] = np.expm1(q_at(Lf, .10, k)); QF["p90c"] = np.expm1(q_at(Lf, .90, k)); QF["level"] = te.level.values
QF["conf_iqr"] = conf_band(QF.answer.values, QF.p25c.values, QF.p75c.values)
print("all eligible items: aggregate answer / (4 x level) =", round(QF.answer.sum() / (4 * QF.level.sum()), 3), "| items >= 12 months:", round(QF[te.age_m.values >= 12].answer.sum() / (4 * QF[te.age_m.values >= 12].level.sum()), 3))

W = pd.read_csv("data/final_forecast.csv", index_col=0)                      # previous run: monthly profile, web share, adjustments
out = pd.DataFrame(index=T)
prof = W[["jul", "aug", "sep", "oct"]].div(W[["jul", "aug", "sep", "oct"]].sum(axis=1), axis=0)
elig = [s for s in T if s in QF.index]
for c in ["q10", "q25", "q50", "q75", "q90", "p10c", "p25c", "p75c", "p90c", "answer", "conf_iqr", "level"]: out.loc[elig, c] = QF.loc[elig, c]
out["raw_total"] = out.answer                                                   # model answer before judgement
sc = "S16079_DARK-GRAY"; out.loc[sc, "raw_total"] = W.loc[sc, "raw_total"]       # not eligible: in-stock rule from D034
M4 = prof.mul(out.raw_total, axis=0); M4.columns = ["jul", "aug", "sep", "oct"]
M4["aug"] *= 1 + 0.07 * W.web_share; M4["sep"] *= 1 + 0.04 * W.web_share         # D031 campaign uplift
dd = pd.read_parquet("data/daily_item_chan.parquet"); dd = dd[~dd.cancelled]; dd["ny_date"] = pd.to_datetime(dd.ny_date)
WC = ["S24840_LIGHT-GRAY-MELANGE", "S24847_NAVY", "S25039_YELLOW"]
r14 = dd[dd.style_color.isin(WC) & (dd.ny_date >= "2026-06-17")].groupby("style_color").qty.sum() / 14
for s in WC:                                                                    # D033 World Cup rule
    M4.loc[s, "jul"] = min(M4.loc[s, "jul"], r14[s] * (19 + 0.75 * 12)); j = M4.loc[s, "jul"]
    for m, cap in [("aug", 0.75), ("sep", 0.50), ("oct", 0.45)]: M4.loc[s, m] = min(M4.loc[s, m], cap * j)
out[["f_jul", "f_aug", "f_sep", "f_oct"]] = M4.round(0).values
out["final_total"] = out[["f_jul", "f_aug", "f_sep", "f_oct"]].sum(axis=1); out["adj_pct"] = out.final_total / out.raw_total - 1
out["conf"] = out.conf_iqr
out.loc[WC + [sc], "conf"] = 3                                                  # judgement dominates; model quantiles do not describe these numbers
out["prev_final"] = W.final_total; out["prev_raw"] = W.raw_total; out["prev_conf"] = W.conf; out["b0_total"] = W.b0_total
for c in ["ptype", "age_m", "macro", "web_share"]: out[c] = W[c]
out["iqr_ratio"] = out.p75c / out.p25c
out.to_csv("data/final_forecast_q.csv")
print("\ntau* =", tau, "| k* =", k)
print("totals: raw answer", int(out.raw_total.sum()), "| final", int(out.final_total.sum()), f"({out.final_total.sum() / out.raw_total.sum() - 1:+.1%})", "| previous final", int(W.final_total.sum()),
      "| calibrated P25 / P75 sum (not a portfolio interval):", int(out.p25c.sum()), "/", int(out.p75c.sum()))
print("confidence:", out.conf.value_counts().sort_index().to_dict(), "| previous:", W.conf.value_counts().sort_index().to_dict())
print("cross-tab previous (rows) vs new (cols):"); print(pd.crosstab(out.prev_conf, out.conf))
cols = ["ptype", "age_m", "level", "q25", "q50", "q75", "p25c", "answer", "p75c", "iqr_ratio", "raw_total", "final_total", "prev_final", "conf", "prev_conf"]
print(out.sort_values("final_total", ascending=False)[cols].round(2).to_string())
