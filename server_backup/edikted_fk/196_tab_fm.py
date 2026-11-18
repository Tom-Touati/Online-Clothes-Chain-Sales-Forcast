"""D201 (fork): tabular foundation models on the remote GPU (venv_tab), on the D159 quantile model's task and features
(data/fk_explain_table.parquet, FULL set, categoricals as integer codes). Target z = log((4-month total + 1) / (4 x level + 1)); context =
training rows whose window ended by the origin, the same time-honest rule as the trees.
  tabdpt  TabDPT (open weights), full context on the GPU (up to all training rows), 8 ensembles; quantiles read from its predictive CDF
  mitra   Mitra (AutoGluon, Apache-2.0) regressor, zero fine-tuning, mean prediction used as the median
TabPFN is not used: its weights need a licence accepted through a browser sign-in (the user's decision).
Writes out/q_backtest_tab_<model>.parquet (q_backtest layout; Mitra gives P50 only, copied to every quantile column)."""
import sys, time, json, numpy as np, pandas as pd, torch, warnings
warnings.filterwarnings("ignore")
T = pd.read_parquet("data/fk_explain_table.parquet"); META = json.load(open("data/fk_explain_meta.json")); COLS = META["FULL"]
for c in META["CAT"]: T[c] = T[c].astype("category").cat.codes.astype(float)
QS = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
ORIG = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]


def quantiles(full, qs):
    p = torch.softmax(full["logits"].float(), dim=1).cpu().numpy(); b = full["borders"].float().cpu().numpy(); cdf = np.cumsum(p, axis=1); out = np.zeros((p.shape[0], len(qs)))
    for j, q in enumerate(qs):
        k = np.clip((cdf < q).sum(axis=1), 0, p.shape[1] - 1); prev = np.where(k > 0, cdf[np.arange(len(k)), k - 1], 0.0)
        frac = np.clip((q - prev) / np.maximum(p[np.arange(len(k)), k], 1e-12), 0, 1); out[:, j] = b[k] + frac * (b[k + 1] - b[k])
    return out


def save(name, rows):
    pd.concat(rows).to_parquet(f"out/q_backtest_tab_{name}.parquet"); print("saved", name, flush=True)


for model in sys.argv[1:] or ["tabdpt", "mitra"]:
    rows, t0 = [], time.time()
    for o in ORIG:
        oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()]; te = T[T.origin == o]
        Xtr, ytr, Xte = tr[COLS].values.astype("float32"), tr.z.values.astype("float32"), te[COLS].values.astype("float32")
        if model == "tabdpt":
            from tabdpt import TabDPTRegressor
            import tabdpt.utils as _tu                                   # the T4 has no flash attention: allow the other kernels
            from torch.nn.attention import SDPBackend, sdpa_kernel as _k
            _tu.sdpa_kernel = lambda *a, **kw: _k([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
            m = TabDPTRegressor(device="cuda", compile=False, verbose=False); m.fit(Xtr, ytr)
            full = m.predict(Xte, n_ensembles=8, context_size=min(len(tr), 12000), seed=0, output_type="full"); zq = np.sort(quantiles(full, QS), axis=1)
        else:
            from autogluon.tabular import TabularPredictor
            d = pd.DataFrame(Xtr, columns=COLS); d["z"] = ytr
            pr = TabularPredictor(label="z", problem_type="regression", eval_metric="mean_absolute_error", path=f"out/ag_{o}", verbosity=0).fit(d, hyperparameters={"MITRA": {"fine_tune": False}}, num_gpus=1)
            z50 = pr.predict(pd.DataFrame(Xte, columns=COLS)).values; zq = np.repeat(z50[:, None], len(QS), 1)
        R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
        for j, q in enumerate(QS): R[f"q{int(q * 100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
        rows.append(R); w = (R.q50 - R.ytot).abs().sum() / R.ytot.sum()
        print(model, o, "context", len(tr), "test", len(te), f"P50 WAPE {w:.3f}", f"{time.time() - t0:.0f}s", flush=True)
    save(model, rows)
