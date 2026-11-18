"""D201b (fork): tabular foundation models through AutoGluon 1.6 on the remote GPU (venv_tab), free for commercial use:
TABICL (TabICLv2), MITRA, NORI, TABDPT-TURBO. TabPFN variants are not used (licence acceptance is the user's decision).
Same task as eda/196: target z, D159 FULL features (categoricals as integer codes), time-honest context = training rows whose window
ended by the origin, capped at the 8,000 most recent rows; a single model per fit (no bagging, no stacking, no fine-tuning where the
model allows); point prediction used as the median. Writes out/q_backtest_tab_<model>.parquet (q_backtest layout, P50 in every column)."""
import sys, json, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from autogluon.tabular import TabularPredictor
T = pd.read_parquet("data/fk_explain_table.parquet"); META = json.load(open("data/fk_explain_meta.json")); COLS = META["FULL"]
for c in META["CAT"]: T[c] = T[c].astype("category").cat.codes.astype(float)
QS = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
CAP = int(next((a.split('=')[1] for a in sys.argv[1:] if a.startswith('cap=')), 8000)); sys.argv = [a for a in sys.argv if not a.startswith('cap=')]   # TabICL needs <= 3000 rows on the T4
HP = {"TABICL": {}, "MITRA": {"fine_tune": False, "ag.max_memory_usage_ratio": 1.5}, "NORI": {}, "TABDPT-TURBO": {}}
for model in sys.argv[1:] or list(HP):
    rows, t0 = [], time.time()
    try:
        for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
            oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()].sort_values("o_ord").tail(CAP); te = T[T.origin == o]
            d = tr[COLS].astype("float32").copy(); d["z"] = tr.z.values
            pr = TabularPredictor(label="z", problem_type="regression", eval_metric="mean_absolute_error", path=f"out/ag_{model}_{o}", verbosity=0).fit(
                d, hyperparameters={model: HP[model]}, num_gpus=1, raise_on_no_models_fitted=True, time_limit=900)
            z50 = pr.predict(te[COLS].astype("float32")).values
            R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
            for q in QS: R[f"q{int(q * 100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(z50) - 1, 0, None)
            rows.append(R); w = (R.q50 - R.ytot).abs().sum() / R.ytot.sum()
            print(model, o, "context", len(tr), f"P50 WAPE {w:.3f}", f"{time.time() - t0:.0f}s", flush=True)
        pd.concat(rows).to_parquet(f"out/q_backtest_tab_{model.lower().replace('-', '_')}_c{CAP}.parquet"); print("saved", model, flush=True)
    except Exception as e: print(model, "FAILED:", repr(e)[:400], flush=True)
