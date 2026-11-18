"""D206 F2 / F4 (fork, M5 initiatives): hyperparameter tuning and feature pruning of the D198 LightGBM, chosen on the four EARLY test
origins only (2025-02 .. 2025-05), so the nine recent origins stay untouched for the evaluation in eda/193b ("tuned", "sel").
Criterion: units WAPE of the median (alpha 0.5) model on the FULL feature set, pooled over the four early origins, time-honest training.
  tune    Optuna TPE, 60 trials: num_leaves, min_child_samples, learning_rate x n_estimators, colsample_bytree, subsample, reg_lambda,
          min_split_gain, max_bin. Writes data/fk_lgb_tuned.json.
  select  backward group elimination: the D198 features in 12 groups (the explain-table groups, events split into upcoming /
          past / interactions, plus the two neighbour blocks); each round drops the group whose removal lowers the early WAPE most,
          while it lowers it by more than 0.002; then single features within the remaining groups, same rule (one pass).
          Writes data/fk_feat_select.json (BASE, FULL lists for eda/193b "sel").
Usage: python eda/193c_tune_select.py tune|select [jobs=4]"""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
KW = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); MODE = [a for a in sys.argv[1:] if "=" not in a][0]; JOBS = int(KW.get("jobs", 4))
T = pd.read_parquet("data/fk_explain_table.parquet"); META = json.load(open("data/fk_explain_meta.json"))
DY = ["dy_nb_sea4", "dy_nb_lvl", "dy_nb_mom3", "dy_nb_yoy"]
T = T.merge(pd.read_parquet("data/fk_dyn_feats.parquet")[["origin", "style_color"] + DY], on=["origin", "style_color"], how="left")
for c in META["CAT"]: T[c] = T[c].astype("category")
BASE, FULL = META["BASE"] + DY, META["FULL"] + DY; P0 = dict(META["PARAMS"]); P0["n_jobs"] = JOBS
EARLY = ["2025-02", "2025-03", "2025-04", "2025-05"]
SPL = []
for o in EARLY:
    oo = pd.Period(o).ordinal; SPL.append((T[(T.o_ord + 4 <= oo) & T.ytot.notna()], T[T.origin == o]))


def early_wape(P, cols):
    num = den = 0.0
    for tr, te in SPL:
        z = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **P).fit(tr[cols], tr.z).predict(te[cols])
        f = np.clip((4 * te.level.values + 1) * np.exp(z) - 1, 0, None); num += np.abs(f - te.ytot.values).sum(); den += te.ytot.values.sum()
    return num / den


t0 = time.time(); ref = early_wape(P0, FULL); print("D198 params, FULL set, early WAPE", round(ref, 4), flush=True)
if MODE == "tune":
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def obj(tr):
        lr = tr.suggest_float("learning_rate", 0.015, 0.12, log=True)
        P = dict(P0, learning_rate=lr, n_estimators=int(min(1500, max(100, 15 / lr))), num_leaves=tr.suggest_int("num_leaves", 4, 63, log=True),
                 min_child_samples=tr.suggest_int("min_child_samples", 10, 300, log=True), colsample_bytree=tr.suggest_float("colsample_bytree", 0.3, 1.0),
                 subsample=tr.suggest_float("subsample", 0.5, 1.0), reg_lambda=tr.suggest_float("reg_lambda", 0.0, 30.0), min_split_gain=tr.suggest_float("min_split_gain", 0.0, 0.5),
                 max_bin=tr.suggest_categorical("max_bin", [63, 127, 255]))
        return early_wape(P, FULL)
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    st.enqueue_trial({"learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8, "reg_lambda": 5.0, "min_split_gain": 0.0, "max_bin": 255})
    st.optimize(obj, n_trials=int(KW.get("trials", 60)), show_progress_bar=False)
    bp = dict(st.best_params); bp["n_estimators"] = int(min(1500, max(100, 15 / bp["learning_rate"])))
    print("best early WAPE", round(st.best_value, 4), "vs D198", round(ref, 4), "|", bp, f"| {time.time() - t0:.0f}s")
    json.dump({"params": bp, "early_wape": st.best_value, "ref_early_wape": ref, "trials": len(st.trials)}, open("data/fk_lgb_tuned.json", "w"), indent=1)
    print(pd.DataFrame([dict(t.params, wape=t.value) for t in st.trials]).sort_values("wape").head(10).round(4).to_string())
else:
    G = META["GROUP"]; grp = {}
    for f in FULL:
        g = G.get(f, "dynamic co-purchase neighbours" if f in DY else "other")
        if g == "special events / occasions": g = "events: upcoming" if f.endswith("_t") or f in ("occ_next", "occ_n4") else ("events: past" if f.endswith("_o") or f == "occ_last" else "events: interactions")
        grp.setdefault(g, []).append(f)
    print({g: len(v) for g, v in grp.items()}, flush=True)
    keep, cur = dict(grp), ref; log = []
    while True:
        res = {g: early_wape(P0, [f for gg, fs in keep.items() if gg != g for f in fs]) for g in keep}
        g, w = min(res.items(), key=lambda kv: kv[1]); print("drop-group round:", {k: round(v, 4) for k, v in res.items()}, flush=True)
        if w < cur - 0.002: keep.pop(g); log.append(("group", g, round(cur, 4), round(w, 4))); cur = w
        else: break
    feats = [f for fs in keep.values() for f in fs]
    for f in list(feats):
        w = early_wape(P0, [x for x in feats if x != f])
        if w < cur - 0.002: feats.remove(f); log.append(("feature", f, round(cur, 4), round(w, 4))); cur = w
    FULLs = [f for f in FULL if f in feats]; BASEs = [f for f in BASE if f in feats]
    print("dropped:", log); print("early WAPE", round(ref, 4), "->", round(cur, 4), "| FULL", len(FULL), "->", len(FULLs), "| BASE", len(BASE), "->", len(BASEs), f"| {time.time() - t0:.0f}s")
    json.dump({"BASE": BASEs, "FULL": FULLs, "dropped": log, "early_wape": cur, "ref_early_wape": ref}, open("data/fk_feat_select.json", "w"), indent=1)
