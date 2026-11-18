"""D345 (fork 2; user: "/goal hyperparameter optimization on all models through the night"). Optuna TPE search for the location recommender
(eda/202, the D358 setup: honest setting, members' item inputs, log-error loss, clip, fallback below 12 closed origins), following the
overnight protocol in SYNC.md:
  tune on origins 2025-02 .. 2025-09 only; criterion = WAPE / R3 of the candidate mean (five members + rc at a quarter weight) on those
  origins, with the recommender alone reported; held-out origins 2025-10 .. 2026-02 are NOT used here (eda/208 scores the best trials).
Each trial runs eda/202 as a subprocess on the tuning origins with 2 seeds. Trials go to data/hpo_rc_trials.csv; the Optuna study to
data/hpo_rc.db (resumable). Usage on the server: python eda/207_rc_hpo.py trials=60 threads=8"""
import sys, os, json, time, subprocess, numpy as np, pandas as pd, optuna
KW = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); NT = int(KW.get("trials", 60)); TH = KW.get("threads", "8"); PY = sys.executable
TUNE = [str(p) for p in pd.period_range("2025-02", "2025-09", freq="M")]
B = pd.read_parquet("data/dec_base_ecnsnn.parquet"); B = B[B.origin.isin(TUNE)].reset_index(drop=True); MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]
y = B.y.values; ar = np.abs(B.R3.values - y); w = lambda f: float(np.abs(f - y).sum() / ar.sum()); top = B[MEM].max(axis=1).values
BASE_SYS = w(B[MEM].values.mean(1)); print("five-member mean on the tuning origins:", round(BASE_SYS, 4), flush=True)
LOG = "data/hpo_rc_trials.csv"
def run(params, tag):
    args = [PY, "eda/202_rc_model.py", "settings=honest", "itemfeat=1", "loss=logl1", "etaclip=" + str(params.pop("etaclip")), "mino=12", "seeds=2", f"threads={TH}",
            "origins=" + ",".join(TUNE), f"tag=_hpo_{tag}"] + [f"{k}={v}" for k, v in params.items()]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError(r.stderr[-600:])
    R = pd.read_parquet(f"data/rc_recsys_preds_hpo_{tag}.parquet"); os.remove(f"data/rc_recsys_preds_hpo_{tag}.parquet")
    for f_ in (f"data/rc_pair_preds_2026_hpo_{tag}.parquet", f"data/rc_factors_2026_hpo_{tag}.parquet"):
        if os.path.exists(f_): os.remove(f_)
    D = B[["origin", "style_color"]].merge(R, on=["origin", "style_color"], how="left"); out = {}
    for k in ("deepfm", "gbm"):
        f = np.minimum(D[f"f_honest_{k}"].fillna(pd.Series(B.R3.values)).values, 3 * top); out[f"alone_{k}"] = w(f); out[f"sys_{k}"] = w((B[MEM].values.sum(1) + 0.25 * f) / 5.25)
    return out
def objective(trial):
    p = dict(lr=trial.suggest_float("lr", 5e-4, 1e-2, log=True), wd=trial.suggest_float("wd", 1e-6, 1e-2, log=True), dim=trial.suggest_categorical("dim", [8, 16, 32]),
             hid=trial.suggest_categorical("hid", [64, 128, 256]), drop=trial.suggest_float("drop", 0.0, 0.3), idrop=trial.suggest_float("idrop", 0.0, 0.6),
             epochs=trial.suggest_int("epochs", 3, 12), bs=trial.suggest_categorical("bs", [4096, 8192, 16384]), etaclip=trial.suggest_float("etaclip", 1.5, 4.0),
             gleaves=trial.suggest_categorical("gleaves", [15, 31, 63, 127]), glr=trial.suggest_float("glr", 0.02, 0.1, log=True), grounds=trial.suggest_int("grounds", 200, 800, step=100),
             gmind=trial.suggest_categorical("gmind", [30, 100, 300]), gff=trial.suggest_float("gff", 0.5, 1.0))
    t0 = time.time(); rec = dict(p); out = run(dict(p), f"t{trial.number}"); rec.update(out); rec.update(trial=trial.number, secs=round(time.time() - t0))
    pd.DataFrame([rec]).to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)
    for k, v in out.items(): trial.set_user_attr(k, v)
    print(f"trial {trial.number}: sys_deepfm {out['sys_deepfm']:.4f} alone {out['alone_deepfm']:.4f} | gbm {out['alone_gbm']:.4f} | {rec['secs']} s", flush=True)
    return out["sys_deepfm"]
st = optuna.create_study(direction="minimize", study_name="rc", storage="sqlite:///data/hpo_rc.db", load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=10))
if len(st.trials) == 0:   # the D358 setting first, as the reference trial
    st.enqueue_trial(dict(lr=3e-3, wd=1e-4, dim=16, hid=128, drop=0.1, idrop=0.3, epochs=6, bs=8192, etaclip=3.0, gleaves=63, glr=0.05, grounds=400, gmind=100, gff=0.8))
st.optimize(objective, n_trials=max(NT - len(st.trials), 0))
best = st.best_trial; print("best", best.number, round(best.value, 4), best.params, flush=True)
json.dump(dict(best=best.params, value=best.value, base_sys=BASE_SYS, n=len(st.trials)), open("data/hpo_rc_best.json", "w"), indent=1)
