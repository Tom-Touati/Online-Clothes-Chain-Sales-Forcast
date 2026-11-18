"""Fork 2 (user goal: "hyperparameter optimization on all models through the night"): Optuna TPE over the rule agents' knobs (the P dict of
agent_eval/2026-06/rules.py, v14, the version behind the deliverable's rules member), with the overnight protocol of SYNC.md:
  tune on origins 2025-03 .. 2025-09 (7), hold out 2025-10 .. 2026-02 (5); criterion = mean item WAPE of the 4-month totals over the tuning
  origins, on each origin's scoreboard items, from the agents' own backtest harness (agent_eval/2026-06/research/bt.py).
Caveat (as in the agents' notes): the harness is an "oracle site" backtest. The site-plan dicts hold actuals for past months, so it tunes
the item logic given the site level, not the site plan. rules.py is not edited; parameters are passed through bt.run(over=...).
Trial 0 = the v14 values. Writes data/hpo_rules_trials.csv and data/hpo_rules_best.json; the best and v14 are then scored on the held-out
origins. Run from the project root: python eda/213_rules_hpo.py trials=60"""
import os, sys, json, time, io, contextlib, numpy as np, pandas as pd, optuna
KW = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); NT = int(KW.get("trials", 60)); ROOT = os.getcwd()
os.chdir("agent_eval/2026-06"); sys.path.insert(0, "research"); sys.path.insert(0, "."); _a = sys.argv; sys.argv = ["x"]
import bt
sys.argv = _a; optuna.logging.set_verbosity(optuna.logging.WARNING)
TUNE = ["2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09"]; HOLD = ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02"]
V14 = dict(bt.BASEP)
def score(over, origins):
    with contextlib.redirect_stdout(io.StringIO()): df = bt.run("hpo", over=over, origins=origins)
    return float(df.wape.mean()), float(df.bias.mean()), float(df.w20.mean()), df
def objective(t):
    o = dict(W14=t.suggest_float("W14", 0.2, 0.7), WLONG=t.suggest_float("WLONG", 0.0, 0.4), RAMP0_NEW=t.suggest_float("RAMP0_NEW", 1.4, 2.4), RAMP1=t.suggest_float("RAMP1", 1.1, 1.7),
             RAMP2=t.suggest_float("RAMP2", 1.0, 1.3), RAMP_H2=t.suggest_float("RAMP_H2", 1.0, 1.2), SHRINK_K=t.suggest_float("SHRINK_K", 4, 40, log=True), CH_K=t.suggest_float("CH_K", 5, 60, log=True),
             RET_ANCHOR_PRE=t.suggest_float("RET_ANCHOR_PRE", 0.0, 0.5), RET_POW=t.suggest_float("RET_POW", 0.5, 1.0), SEAS_ANCHOR=t.suggest_float("SEAS_ANCHOR", 0.4, 1.0),
             SEAS_POW=t.suggest_float("SEAS_POW", 0.6, 1.0), AGE_FADE=t.suggest_float("AGE_FADE", 0.85, 1.0), HAL_SOFT=t.suggest_float("HAL_SOFT", 0.2, 1.0), COH_POW=t.suggest_float("COH_POW", 0.0, 1.0))
    amp = t.suggest_float("AMP_FLOOR", 0.5, 0.9); o["AMP_CLIP"] = (amp, 1.0)
    if o["W14"] + o["WLONG"] > 0.95: raise optuna.TrialPruned()
    t0 = time.time(); w, b, h, _ = score(o, TUNE)
    rec = dict(trial=t.number, wape=w, bias=b, w20=h, secs=round(time.time() - t0), **{k: (v if not isinstance(v, tuple) else v[0]) for k, v in o.items()})
    pd.DataFrame([rec]).to_csv(f"{ROOT}/data/hpo_rules_trials.csv", mode="a", header=not os.path.exists(f"{ROOT}/data/hpo_rules_trials.csv"), index=False)
    print(f"trial {t.number}: tune WAPE {w:.4f} bias {b:+.3f} | {rec['secs']} s", flush=True); return w
st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=12))
st.enqueue_trial({**{k: V14[k] for k in ["W14", "WLONG", "RAMP0_NEW", "RAMP1", "RAMP2", "RAMP_H2", "SHRINK_K", "CH_K", "RET_ANCHOR_PRE", "RET_POW", "SEAS_ANCHOR", "SEAS_POW", "AGE_FADE", "HAL_SOFT", "COH_POW"]}, "AMP_FLOOR": V14["AMP_CLIP"][0]})
st.optimize(objective, n_trials=NT)
b = st.best_trial; bp = dict(b.params); amp = bp.pop("AMP_FLOOR"); bp["AMP_CLIP"] = (amp, 1.0)
res = {}
for nm, over in [("v14", {}), ("tuned", bp)]:
    for tag, oo in [("tune", TUNE), ("hold", HOLD)]:
        w, bi, h, df = score(over, oo); res[f"{nm}_{tag}"] = dict(wape=round(w, 4), bias=round(bi, 3), w20=round(h, 3), per_origin=[round(x, 4) for x in df.wape])
print(json.dumps(res, indent=0))
json.dump(dict(best=b.params, value=b.value, n=len(st.trials), scores=res), open(f"{ROOT}/data/hpo_rules_best.json", "w"), indent=1)
