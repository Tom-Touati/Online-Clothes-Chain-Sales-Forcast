"""D362 (user, overnight goal via Fork 2: "hyperparameter optimization on all models through the night"; protocol in SYNC.md "Overnight
tuning"): Optuna TPE on the Huber tree member (eda/187_losses.py 'huber': ec_ns inputs, LightGBM objective huber on
z = log((4-month total + 1) / (4 x level + 1)), BASE and FULL fits averaged).
Search space: num_leaves, learning_rate, n_estimators, min_child_samples, colsample_bytree (feature fraction), subsample (bagging,
freq 1), reg_lambda, reg_alpha, Huber delta (alpha), and the BASE / FULL blend weight w_base.
Tuning origins 2025-02 .. 2025-09 only (time-honest training, o' + 4 <= o). Criterion = WAPE / R3 of the member alone + WAPE / R3 of the
five-member mean (rules, member, new NN S13ecx6, pf2c, pf2d32 from data/dec_base_ecnsnn.parquet) on those origins; lower is better.
The default parameters (D353 'huber') are trial 0. Hold-out origins 2025-10 .. 2026-02 are never seen here.
Usage: python eda/191_hpo_tree.py trials=80 [threads=8]; writes data/hpo_tree_trials.csv and data/hpo_tree_best.json."""
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb, warnings, optuna
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
KW = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); NT = int(KW.get("trials", 80)); TH = int(KW.get("threads", 8))
sys.argv = ["eda/180q_models.py"]; ns = {"__name__": "hpo"}
exec(compile(open("eda/180q_models.py", encoding="utf-8").read(), "eda/180q_models.py", "exec"), ns)
T, sets, P0 = ns["T"], ns["sets"], dict(ns["P"]); B_, F_ = sets("ec_ns")
TUNE = [str(p) for p in pd.period_range("2025-02", "2025-09", freq="M")]
DB = pd.read_parquet("data/dec_base_ecnsnn.parquet").set_index(["origin", "style_color"])
def run(params, wb, delta, origins):
    P = dict(P0); P.update(params); P["n_jobs"] = TH; out = []
    for o in origins:
        oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()]; te = T[T.origin == o]
        zb = lgb.LGBMRegressor(objective="huber", alpha=delta, **P).fit(tr[B_], tr.z).predict(te[B_]); zf = lgb.LGBMRegressor(objective="huber", alpha=delta, **P).fit(tr[F_], tr.z).predict(te[F_])
        z = wb * zb + (1 - wb) * zf; out.append(pd.DataFrame({"origin": o, "style_color": te.style_color.values, "q50": np.clip((4 * te.level.values + 1) * np.exp(z) - 1, 0, None)}))
    return pd.concat(out)
def score(pred):
    D = DB.reindex(pd.MultiIndex.from_frame(pred[["origin", "style_color"]])); m = np.isfinite(D.y.values)
    x = pred.q50.values[m]; y, r3 = D.y.values[m], D.R3.values[m]
    S = np.nanmean(np.column_stack([np.clip(v, 0, None) for v in (D.rules.values[m], x, D.nn.values[m], D.pf2c.values[m], D.pf2d32.values[m])]), axis=1)
    e3 = np.abs(r3 - y).sum(); return np.abs(x - y).sum() / e3, np.abs(S - y).sum() / e3
rows = []
def objective(tr):
    params = dict(num_leaves=tr.suggest_int("num_leaves", 7, 63, log=True), learning_rate=tr.suggest_float("learning_rate", 0.01, 0.12, log=True),
                  n_estimators=tr.suggest_int("n_estimators", 100, 900, step=50), min_child_samples=tr.suggest_int("min_child_samples", 10, 200, log=True),
                  colsample_bytree=tr.suggest_float("colsample_bytree", 0.4, 1.0), subsample=tr.suggest_float("subsample", 0.5, 1.0),
                  reg_lambda=tr.suggest_float("reg_lambda", 0.01, 30, log=True), reg_alpha=tr.suggest_float("reg_alpha", 0.001, 5, log=True))
    wb, delta = tr.suggest_float("w_base", 0.0, 1.0), tr.suggest_float("huber_delta", 0.15, 1.5, log=True)
    t0 = time.time(); m, s = score(run(params, wb, delta, TUNE)); crit = m + s
    rows.append({"trial": tr.number, **params, "w_base": wb, "huber_delta": delta, "member_tune": round(m, 4), "system_tune": round(s, 4), "criterion": round(crit, 4), "seconds": round(time.time() - t0)})
    pd.DataFrame(rows).to_csv("data/hpo_tree_trials.csv", index=False); print(rows[-1], flush=True); return crit
st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
st.enqueue_trial({"num_leaves": P0["num_leaves"], "learning_rate": P0["learning_rate"], "n_estimators": P0["n_estimators"], "min_child_samples": P0["min_child_samples"],
                  "colsample_bytree": P0["colsample_bytree"], "subsample": P0["subsample"], "reg_lambda": P0["reg_lambda"], "reg_alpha": 0.001, "w_base": 0.5, "huber_delta": 0.5})
st.optimize(objective, n_trials=NT)
b = st.best_trial; json.dump({"params": {k: v for k, v in b.params.items() if k not in ("w_base", "huber_delta")}, "w_base": b.params["w_base"], "huber_delta": b.params["huber_delta"],
                              "criterion": b.value, "default_criterion": rows[0]["criterion"], "trial": b.number}, open("data/hpo_tree_best.json", "w"), indent=1)
print("best", b.number, round(b.value, 4), "| default", rows[0]["criterion"], b.params)
