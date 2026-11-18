"""The time dimension of the training data (D139): is old history helping or hurting, given the drift found in D115/D138?
Production quantile model (eda/20, both feature sets, 13 origins), training rows restricted to the last N origins with complete outcomes
(win6, win9, win12) or all rows with recency weights 0.5 ** (age in months / half-life) (hl6, hl12). Writes data/q_backtest_tw_<var>.parquet,
scored by eda/91_score_context.py tw."""
import sys, numpy as np, pandas as pd, warnings, importlib.util, time, lightgbm as lgb
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
VARS = sys.argv[1:] or ["win6", "win9", "win12", "hl6", "hl12"]
spec = importlib.util.spec_from_file_location("q20", "eda/20_quantiles.py"); q20 = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(q20); sys.argv = _a
q20.P["n_jobs"] = 8
def fit_q_w(tr, te, w):
    out = np.zeros((len(te), len(q20.QS)))
    for cols in (q20.BASE, q20.FULL):
        for j, q in enumerate(q20.QS):
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, **q20.P).fit(tr[cols], tr.z, sample_weight=w); out[:, j] += m.predict(te[cols]) / 2
    return np.sort(out, axis=1)
for var in VARS:
    rows, t0 = [], time.time()
    for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
        oo = pd.Period(o).ordinal; tr = q20.T1[(q20.T1.o_ord + 4 <= oo) & q20.T1.ytot.notna()]; te = q20.T1[q20.T1.origin == o]
        last = oo - 4                                                                   # the latest origin with a complete window
        if var.startswith("win"):
            n = int(var[3:]); tr = tr[tr.o_ord > last - n]; w = np.ones(len(tr))
        else:
            w = 0.5 ** ((last - tr.o_ord.values) / int(var[2:]))
        zq = fit_q_w(tr, te, w); R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
        for j, q in enumerate(q20.QS): R[f"q{int(q*100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
        rows.append(R)
    pd.concat(rows).to_parquet(f"data/q_backtest_tw_{var}.parquet"); print(var, round(time.time() - t0), "s", flush=True)
