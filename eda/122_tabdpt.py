"""Tabular foundation model (D130): TabDPT (open weights, in-context learning, CPU) on the production quantile model's task, 13 origins.
Target z = log((4-month total + 1) / (4 x level + 1)); features = the eda/20 FULL set (categoricals as integer codes); context = training rows
whose window ended by the origin (subsampled to CONTEXT rows per prediction by TabDPT). TabDPT returns a full predictive distribution
(a histogram over bins); P10/P25/P40/P50/P60/P75/P90 are read from its CDF. Writes data/q_backtest_tfm_dpt.parquet in the q_backtest
layout, so eda/91_score_context.py tfm scores it (alone and inside the three-way) in the standard table.
TabPFN was not used: its weights need a licence accepted through a browser sign-in (the user's decision)."""
import sys, time, numpy as np, pandas as pd, torch, warnings, importlib.util
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4)
from tabdpt import TabDPTRegressor
ARGS = dict(a.split("=") for a in sys.argv[1:]); CONTEXT = int(ARGS.get("context", 2048)); NE = int(ARGS.get("ens", 4))
spec = importlib.util.spec_from_file_location("q20", "eda/20_quantiles.py"); q20 = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(q20); sys.argv = _a
T1 = q20.T1.copy(); COLS = list(q20.FULL)
for c in COLS:
    if str(T1[c].dtype) == "category": T1[c] = T1[c].cat.codes.astype(float)
QS = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
def quantiles(full, qs):
    p = torch.softmax(full["logits"].float(), dim=1).numpy(); b = full["borders"].float().numpy(); cdf = np.cumsum(p, axis=1); out = np.zeros((p.shape[0], len(qs)))
    for j, q in enumerate(qs):
        k = (cdf < q).sum(axis=1); k = np.clip(k, 0, p.shape[1] - 1); prev = np.where(k > 0, cdf[np.arange(len(k)), k - 1], 0.0)
        frac = np.clip((q - prev) / np.maximum(p[np.arange(len(k)), k], 1e-12), 0, 1); out[:, j] = b[k] + frac * (b[k + 1] - b[k])
    return out
rows, t0 = [], time.time()
for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
    oo = pd.Period(o).ordinal; tr = T1[(T1.o_ord + 4 <= oo) & T1.ytot.notna()]; te = T1[T1.origin == o]
    m = TabDPTRegressor(device="cpu", compile=False, verbose=False); m.fit(tr[COLS].values.astype("float32"), tr.z.values.astype("float32"))
    full = m.predict(te[COLS].values.astype("float32"), n_ensembles=NE, context_size=min(CONTEXT, len(tr)), seed=0, output_type="full")
    zq = np.sort(quantiles(full, QS), axis=1)
    R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
    for j, q in enumerate(QS): R[f"q{int(q*100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
    rows.append(R); w = (R.q50 - R.ytot).abs().sum() / R.ytot.sum()
    print(o, "context pool", len(tr), "test", len(te), f"P50 WAPE {w:.3f}", f"{time.time() - t0:.0f}s", flush=True)
pd.concat(rows).to_parquet(ARGS.get("out", "data/q_backtest_tfm_dpt.parquet"))
