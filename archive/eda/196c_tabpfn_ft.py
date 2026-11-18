"""D206 F5 (fork, M5 initiatives / augmentation): TabPFN fine-tuned on our data, time-honest. At each of the 13 backtest origins the
pretrained TabPFN (v2 by default: Prior Labs License, production-usable) is fine-tuned (tabpfn 10 FinetunedTabPFNRegressor; CRPS + MSE
loss on the bar distribution, AdamW lr 1e-5, warm-up schedule, early stopping on a random 10% of the training rows, at most `tl`
seconds) on the rows whose outcome is known by the origin, with the D198 features (D159 FULL + dynamic co-occurrence neighbours);
quantiles from the fine-tuned model's predictive distribution (8 estimators at inference).
Usage: python 196c_tabpfn_ft.py [v2|v3.5-fast] [tl=600] [epochs=30] [lr=1e-5]
Writes out/q_backtest_tabpfn_ft_<version>.parquet (q_backtest layout) and a per-origin log line."""
import sys, time, json, numpy as np, pandas as pd, warnings, torch
warnings.filterwarnings("ignore")
from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor
from tabpfn.constants import ModelVersion
KW = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); VER = ([a for a in sys.argv[1:] if "=" not in a] or ["v2"])[0]
TL, EP, LR = int(KW.get("tl", 600)), int(KW.get("epochs", 30)), float(KW.get("lr", 1e-5))
T = pd.read_parquet("data/fk_explain_table.parquet"); META = json.load(open("data/fk_explain_meta.json")); COLS = META["FULL"]
for c in META["CAT"]: T[c] = T[c].astype("category").cat.codes.astype(float)
DY = ["dy_nb_sea4", "dy_nb_lvl", "dy_nb_mom3", "dy_nb_yoy"]
T = T.merge(pd.read_parquet("data/fk_dyn_feats.parquet")[["origin", "style_color"] + DY], on=["origin", "style_color"], how="left"); COLS = COLS + DY
QS = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]; rows, t0 = [], time.time()
for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
    oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()]; te = T[T.origin == o]
    m = FinetunedTabPFNRegressor(device="cuda", epochs=EP, time_limit=TL, learning_rate=LR, random_state=0, model_version=ModelVersion(VER),
                                 n_finetune_ctx_plus_query_samples=min(len(tr), 12000), save_checkpoint_interval=None,
                                 extra_regressor_kwargs={"ignore_pretraining_limits": True})
    m.fit(tr[COLS].values.astype("float32"), tr.z.values.astype("float32"))
    zq = np.sort(np.column_stack(m.predict(te[COLS].values.astype("float32"), output_type="quantiles", quantiles=QS)), axis=1)
    R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
    for j, q in enumerate(QS): R[f"q{int(q * 100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
    rows.append(R); w = (R.q50 - R.ytot).abs().sum() / R.ytot.sum()
    print("tabpfn-ft", VER, o, "context", len(tr), f"P50 WAPE {w:.3f}", f"{time.time() - t0:.0f}s", flush=True)
    del m; torch.cuda.empty_cache()
pd.concat(rows).to_parquet(f"out/q_backtest_tabpfn_ft_{VER.replace('.', '_')}.parquet"); print("saved", VER)
