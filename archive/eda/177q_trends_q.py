"""D183 (overnight M5 goal, direction 6): Google Trends features (eda/177g, data/gtrends_feats.parquet) in the D159 quantile model.
  gt_a   gt_mom3, gt_yoy, gt_lvl (search momentum, year-on-year growth and level of the item's looks, up to the origin)
  gt_b   + gt_ramp (last year's search ramp from the origin months into the target months)
  gt_c   + interactions: gt_mom3 x the item's own 3-month momentum (lagr1), gt_yoy x level
Writes data/q_backtest_gt_{a,b,c}.parquet; scored by eda/91_score_context.py gt data/q_backtest_wx_cpllm_we.parquet."""
import sys, numpy as np, pandas as pd, time
src = open("eda/164_q_weather.py", encoding="utf-8").read(); _a = sys.argv; sys.argv = sys.argv[:1]
exec(src[src.index("import sys"):src.index("for var in RUN:")]); sys.argv = _a
GT = pd.read_parquet("data/gtrends_feats.parquet"); T0 = T0.merge(GT, on=key, how="left")
T0["gt_x_mom"] = T0.gt_mom3 * T0.lagr1; T0["gt_x_lvl"] = T0.gt_yoy * T0.lvl
V = {"a": ["gt_mom3", "gt_yoy", "gt_lvl"], "b": ["gt_mom3", "gt_yoy", "gt_lvl", "gt_ramp"], "c": ["gt_mom3", "gt_yoy", "gt_lvl", "gt_ramp", "gt_x_mom", "gt_x_lvl"]}
for var in (sys.argv[1:] or list(V)):
    NEW = CP + LL + WE + V[var]; qm.BASE = B0 + NEW; qm.FULL = F0 + NEW; out, t0 = [], time.time()
    for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
        oo = pd.Period(o).ordinal; tr = T0[(T0.o_ord + 4 <= oo) & T0.ytot.notna()]; te = T0[T0.origin == o]
        zq = qm.fit_q(tr, te); R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
        for j, q in enumerate(qm.QS): R[f"q{int(q*100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
        out.append(R)
    pd.concat(out).to_parquet(f"data/q_backtest_gt_{var}.parquet"); print(var, round(time.time() - t0), "s", flush=True)
