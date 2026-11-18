"""Production quantile model (eda/20_quantiles.py, 4-month totals, 13 rolling origins) with the D112 context features added to both
feature sets. Variants: occ (occasion timing), disc (discount momentum), both. Writes data/q_backtest_ctx_<var>.parquet.
Usage: python eda/90_q_context.py occ disc both"""
import sys, numpy as np, pandas as pd, warnings, importlib.util, time
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
VARS = sys.argv[1:] or ["occ", "disc", "both"]
spec = importlib.util.spec_from_file_location("q20", "eda/20_quantiles.py"); q20 = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(q20); sys.argv = _argv
q20.P["n_jobs"] = 6
CF = pd.read_parquet("data/context_feats.parquet")
T1 = q20.T1.merge(CF, on=["origin", "style_color"], how="left")
OCC = ["occ_next", "occ_last", "occ_n4"]; DISC = ["dm_7_28", "dm_28_p91", "dm_vs_min", "s_dm_28_p91"]
B0, F0 = list(q20.BASE), list(q20.FULL)
for var in VARS:
    add = {"base": [], "occ": OCC, "disc": DISC, "both": OCC + DISC}[var]          # base = rerun check of the stored backtest
    q20.BASE, q20.FULL = B0 + add, F0 + add
    rows, t0 = [], time.time()
    for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
        oo = pd.Period(o).ordinal; tr = T1[(T1.o_ord + 4 <= oo) & T1.ytot.notna()]; te = T1[T1.origin == o]
        zq = q20.fit_q(tr, te); R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
        for j, q in enumerate(q20.QS): R[f"q{int(q*100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
        rows.append(R)
    pd.concat(rows).to_parquet(f"data/q_backtest_ctx_{var}.parquet"); print(var, "done", round(time.time() - t0), "s", flush=True)
