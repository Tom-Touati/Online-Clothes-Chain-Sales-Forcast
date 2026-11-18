"""Production quantile model (eda/20, 13 origins) with the D122 retail rollout features added to both feature sets.
Variants: rr (item rollout: stores selling, coverage, change, rate per store, days since first store sale, new-store share) and
rrsite (rr + site stores open and stores opened in the last 90 days). Writes data/q_backtest_rr_<var>.parquet; scored by eda/91_score_context.py rr."""
import sys, numpy as np, pandas as pd, warnings, importlib.util, time
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
VARS = sys.argv[1:] or ["rr", "rrsite"]
spec = importlib.util.spec_from_file_location("q20", "eda/20_quantiles.py"); q20 = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(q20); sys.argv = _a
q20.P["n_jobs"] = 4
RF = pd.read_parquet("data/rollout_feats.parquet"); T1 = q20.T1.merge(RF, on=["origin", "style_color"], how="left")
ITEM = ["rr_n28", "rr_cov", "rr_dn", "rr_rate", "rr_age", "rr_newshare"]; SITE = ["s_open", "s_new90"]
B0, F0 = list(q20.BASE), list(q20.FULL)
for var in VARS:
    add = ITEM + (SITE if var == "rrsite" else []); q20.BASE, q20.FULL = B0 + add, F0 + add; rows, t0 = [], time.time()
    for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
        oo = pd.Period(o).ordinal; tr = T1[(T1.o_ord + 4 <= oo) & T1.ytot.notna()]; te = T1[T1.origin == o]
        zq = q20.fit_q(tr, te); R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
        for j, q in enumerate(q20.QS): R[f"q{int(q*100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
        rows.append(R)
    pd.concat(rows).to_parquet(f"data/q_backtest_rr_{var}.parquet"); print(var, round(time.time() - t0), "s", flush=True)
