"""Style-level information (D145): the models see only the item's own history, never its siblings'. Style features at each origin (data <= origin):
  sty_lvl    log1p of the style's mean monthly units over the last 3 months (all colours)
  sty_n      colours of the style with sales in the last 3 months
  sty_share  the item's share of the style's last-3-month units
  sty_mom    log((style units last month + 1) / (style mean monthly units in the 3 months before + 1))
  sty_new2   a sibling colour had its first sale in the last 2 months (knowable launch signal; D144 found a halo)
Added to both feature sets of the lean + occasion-timing quantile model (eda/20b, D135); variant also on the production model.
Writes data/style_feats.parquet and data/q_backtest_sty_{lean,prod}.parquet; scored by eda/91_score_context.py sty."""
import sys, numpy as np, pandas as pd, warnings, importlib.util, time
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
sty = pd.Series(M.index.str.split("_").str[0], index=M.index); S = M.groupby(sty).sum()
first = M.gt(0).idxmax(axis=1).where(M.gt(0).any(axis=1)); P = lambda o, k: str(pd.Period(o, "M") + k)
rows = []
for o in [str(p) for p in pd.period_range("2024-04", "2026-06", freq="M")]:
    l3 = [P(o, -k) for k in range(3)]; p3 = [P(o, -k) for k in range(1, 4)]
    if not all(c in M.columns for c in p3): continue
    st3 = S[l3].mean(axis=1); it3 = M[l3].sum(axis=1)
    live = (M[l3].sum(axis=1) > 0).groupby(sty).sum()
    newsib = first[(first > P(o, -2)) & (first <= o)].index; newsty = set(sty[newsib])
    X = pd.DataFrame({"origin": o, "style_color": M.index})
    X["sty_lvl"] = np.log1p(st3.reindex(sty).values); X["sty_n"] = live.reindex(sty).values
    X["sty_share"] = (it3 / S[l3].sum(axis=1).reindex(sty).values).replace([np.inf], np.nan).values
    X["sty_mom"] = np.log((S[o].reindex(sty).values + 1) / (S[p3].mean(axis=1).reindex(sty).values + 1))
    X["sty_new2"] = [int(st in newsty and s not in set(newsib)) for s, st in zip(M.index, sty)]
    rows.append(X)
SF = pd.concat(rows, ignore_index=True); SF.to_parquet("data/style_feats.parquet", index=False)
print("style feature rows", len(SF), "| share with a new sibling in the last 2 months:", round(SF.sty_new2.mean(), 3))
NEW = ["sty_lvl", "sty_n", "sty_share", "sty_mom", "sty_new2"]
for var, mod in [("lean", "eda/20b_quantiles_lean.py"), ("prod", "eda/20_quantiles.py")]:
    spec = importlib.util.spec_from_file_location("qm", mod); qm = importlib.util.module_from_spec(spec)
    _a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(qm); sys.argv = _a
    qm.P["n_jobs"] = 4; T1 = qm.T1.merge(SF, on=["origin", "style_color"], how="left"); qm.BASE = qm.BASE + NEW; qm.FULL = qm.FULL + NEW
    out, t0 = [], time.time()
    for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
        oo = pd.Period(o).ordinal; tr = T1[(T1.o_ord + 4 <= oo) & T1.ytot.notna()]; te = T1[T1.origin == o]
        zq = qm.fit_q(tr, te); R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
        for j, q in enumerate(qm.QS): R[f"q{int(q*100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
        out.append(R)
    pd.concat(out).to_parquet(f"data/q_backtest_sty_{var}.parquet"); print(var, round(time.time() - t0), "s", flush=True)
