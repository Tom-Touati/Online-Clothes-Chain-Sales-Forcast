"""D325 / D326 (fork; overnight tuning, user in Fork 2's session: "/goal hyperparameter optimization on all models through the night"; split and
protocol in SYNC.md, "Overnight tuning"). Runs ON THE GPU 44.222.103.126 in ~/edikted_fk/hpo, with the venv that has Optuna; each trial is a
subprocess of the member's own script:
  nn      eda/197b_nn_ec.py (eda/57 + the D253 inputs), seeds=1, origins 2025-02..2025-09; search: dim, epochs, lr, weight decay, dropout,
          embedding penalty, item x month interaction
  tabpfn  eda/196b_tabpfn.py v2 feat=dyn (copy with temp= and out=), origins 2025-02..2025-09; search: n_estimators, softmax temperature,
          context cap (most recent rows), average before softmax
Criterion (tune origins only): mean of the member's WAPE / R3 alone and the five-member mean's WAPE / R3 with the trial in place of the member
(nn -> nn; tabpfn -> pf2d32), before corrections; the other members from data/fk_err_master.parquet. Trial 0 = the current settings.
Usage: python hpo_gpu_driver.py nn|tabpfn <n_trials> <max hours>
Writes out/hpo_<model>_trials.csv and out/hpo_<model>_best.json."""
import sys, os, json, time, subprocess, datetime, numpy as np, pandas as pd, optuna
MODEL, NT, STOP_AT = sys.argv[1], int(sys.argv[2]), time.time() + float(sys.argv[3]) * 3600   # stop starting new trials after this many hours
TUNE = [str(p) for p in pd.period_range("2025-02", "2025-09", freq="M")]
D = pd.read_parquet("data/fk_err_master.parquet"); D = D[D.origin.isin(TUNE)].reset_index(drop=True); K = pd.MultiIndex.from_arrays([D.origin, D.style_color])
y = D.y.values; ar = np.abs(D.R3.values - y); MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; SLOT = {"nn": "nn", "tabpfn": "pf2d32"}[MODEL]
wape = lambda f: float(np.abs(f - y).sum() / ar.sum())
def crit(pred):
    M = {m: D[m].values for m in MEM}; M[SLOT] = pred; mean5 = np.mean([M[m] for m in MEM], 0); a, s = wape(pred), wape(mean5); return (a + s) / 2, a, s
def q50(path):
    q = pd.read_parquet(path); q = q.groupby(["origin", "style_color"]).q50.sum() if "h" in q.columns and q.h.nunique() > 1 else q.set_index(["origin", "style_color"]).q50; return q.reindex(K).values
def run_nn(p, tag):
    a = ["../venv/bin/python", "eda/197b_nn_ec.py", "seq=1", "seeds=1", "seed0=0", "origins=" + ",".join(TUNE), f"tag={tag}"] + [f"{k}={v}" for k, v in p.items()]
    r = subprocess.run(a, capture_output=True, text=True); out = f"data/nn_backtest_preds{tag}.parquet"
    if r.returncode or not os.path.exists(out): raise RuntimeError(r.stderr[-800:])
    v = q50(out); os.remove(out); return v
def run_pf(p, tag):
    out = f"out/hpo_{tag}.parquet"; a = ["../venv_tab/bin/python", "eda/196b_tabpfn.py", "v2", "feat=dyn", "orig=" + ",".join(TUNE), f"out={out}"] + [f"{k}={v}" for k, v in p.items()]
    r = subprocess.run(a, capture_output=True, text=True)
    if r.returncode or not os.path.exists(out): raise RuntimeError((r.stdout + r.stderr)[-800:])
    v = q50(out); os.remove(out); return v
rows = []; f_csv = f"out/hpo_{MODEL}_trials.csv"
def objective(t):
    if time.time() > STOP_AT: t.study.stop(); raise optuna.TrialPruned()
    if MODEL == "nn":
        p = dict(dim=t.suggest_categorical("dim", [8, 16, 24, 32]), epochs=t.suggest_int("epochs", 15, 45, step=5), lr=round(t.suggest_float("lr", 1e-3, 1e-2, log=True), 5), wd=float(f'{t.suggest_float("wd", 1e-6, 1e-3, log=True):.2e}'),
                 drop=round(t.suggest_float("drop", 0.0, 0.3), 3), pen=round(t.suggest_float("pen", 0.1, 10.0, log=True), 3), inter=t.suggest_categorical("inter", [0, 1]))
    else:
        p = dict(est=t.suggest_categorical("est", [4, 8, 16, 32]), temp=round(t.suggest_float("temp", 0.5, 1.3), 3), cap=t.suggest_categorical("cap", [0, 3000, 6000, 10000]), abs_=t.suggest_categorical("abs_", [0, 1]))
    t0 = time.time(); tag = f"_hpo{t.number}"
    try: v = run_nn(p, tag) if MODEL == "nn" else run_pf({("abs" if k == "abs_" else k): x for k, x in p.items()}, f"pf{t.number}")
    except Exception as e:
        rows.append(dict(trial=t.number, **p, crit=np.nan, alone=np.nan, mean5=np.nan, secs=round(time.time() - t0), error=str(e)[:300])); pd.DataFrame(rows).to_csv(f_csv, index=False); raise optuna.TrialPruned()
    c, a, s = crit(v); rows.append(dict(trial=t.number, **p, crit=round(c, 5), alone=round(a, 5), mean5=round(s, 5), secs=round(time.time() - t0), error="")); pd.DataFrame(rows).to_csv(f_csv, index=False)
    print(t.number, p, round(c, 4), round(a, 4), round(s, 4), f"{time.time() - t0:.0f}s", flush=True); return c
st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=8))
st.enqueue_trial(dict(dim=16, epochs=30, lr=3e-3, wd=1e-4, drop=0.1, pen=1.0, inter=1) if MODEL == "nn" else dict(est=32, temp=0.9, cap=0, abs_=0))   # the current settings
st.optimize(objective, n_trials=NT)
R = pd.DataFrame(rows); ok = R[R.crit.notna()]; b = ok.loc[ok.crit.idxmin()].to_dict(); base = R.iloc[0].to_dict()
json.dump(dict(best={k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in b.items() if k not in ("error",)}, base={k: v for k, v in base.items() if k != "error"}, n=len(R), n_ok=len(ok)), open(f"out/hpo_{MODEL}_best.json", "w"), indent=1, default=float)
print("best", b, "| current", base, flush=True)
