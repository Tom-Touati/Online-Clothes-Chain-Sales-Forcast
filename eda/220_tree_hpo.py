"""Fork 2 (user goal, 18 Nov ~03:40: "try to improve the new-item version and the other version, optimize each of them and try multiple versions
for that ... optimize any other bits on the meaningful months with the new metrics"; run on the CPU server, ~/edikted_main).
Two Optuna studies on the tree member of the new nx5 (routing: items under 6 months old at the origin get the new-item tree, the rest the
old-item tree):
  role new  objective = WAPE / R3 of the nx5 system (the other members as stored, the correction multiplier of the routed nx5 held fixed)
            over items under 6 months, tune origins 2025-02 .. 2025-09
  role old  the same over items 6 months and older
Space: loss (Huber with delta / log L1), num_leaves, min_child_samples, learning_rate x n_estimators, colsample, subsample, reg_lambda, the
weight of new items' training rows (new role 1-12, old role 0.2-3) and the age below which a training row counts as new (3 / 6 / 9 / 12).
Trial 0 = the current tree of that role (new: huber_new5; old: huber). Finals (FINALS=1): the top trials on all 13 origins, scored on the
held-out origins (2025-10 .. 2026-02), the focus windows (2025-06 = Jul-Oct 2025, 2026-02 = Mar-Jun 2026), recent 9, all 13, and the
per-item metrics (median over items with 6+ origins of the item's correlation of log forecast with log actual, and of its R2).
Usage: ROLE=new|old TRIALS=n NJOBS=4 python eda/220_tree_hpo.py   ->  data/hpo_tree_<role>_trials.csv
       ROLE=.. FINALS=1 TOP=3 python eda/220_tree_hpo.py          ->  data/hpo_tree_<role>_finals.parquet (q50 per origin, all 13 + 2026-06), _finals.json"""
import os, sys, json, time, numpy as np, pandas as pd, optuna, warnings
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
ROLE = os.environ.get("ROLE", "new"); NT = int(os.environ.get("TRIALS", 100)); FIN = os.environ.get("FINALS") == "1"
src = open("eda/187_losses_f2.py", encoding="utf-8").read().split("for v in VARS:")[0]
sys.argv = ["x"]; exec(compile(src, "187", "exec"))
P0 = dict(P)
Bz = pd.read_parquet("data/dec_base_nx5r26.parquet"); Bz = Bz[Bz.y.notna()].reset_index(drop=True)
Gz = pd.read_parquet("data/geo_nx5r_x_preds.parquet")[["origin", "style_color", "F", "geo"]]
Bz = Bz.merge(Gz.rename(columns={"F": "F_g"}), on=["origin", "style_color"], how="left"); Bz["corr"] = (Bz.geo / Bz.F_g.clip(lower=1e-9)).fillna(1.0)
Bz["new"] = Bz.age_m.fillna(99) < 6; OTH = ["rules", "nn", "pf2c", "pf2d32"]
def system(tree_q, rows):
    b = Bz.loc[rows]; return ((b[OTH].clip(lower=0).sum(1) + np.maximum(tree_q, 0) + 0.25 * b.rc.clip(lower=0)) / 5.25 * b["corr"]).values
TUNE = [str(p) for p in pd.period_range("2025-02", "2025-09", freq="M")]; ALL13 = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
KEY = {(o, s): i for i, (o, s) in enumerate(zip(Bz.origin, Bz.style_color))}
def fit_predict(prm, origins):
    """train the tree with settings prm at each origin (time-honest: training origins o' + 4 <= o), return q50 per (origin, style_color)"""
    out = []
    for o in origins:
        oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()].copy(); te = T[T.origin == o].copy()
        if prm.get("lyd"): tr = tr[~tr.o_ord.between(oo - 13, oo - 11)]
        if prm.get("only"): tr = tr[tr.age_m.fillna(99) < prm["only"]]   # train on young items' rows only (GRID=2)
        wt = np.where(tr.age_m.fillna(99).values < prm["wcut"], prm["w_new"], 1.0)
        P.clear(); P.update(P0); P.update(dict(num_leaves=prm["num_leaves"], min_child_samples=prm["min_child"], learning_rate=prm["lr"], n_estimators=prm["n_est"],
                                              colsample_bytree=prm["colsample"], subsample=prm["subsample"], reg_lambda=prm["reg_lambda"], n_jobs=int(os.environ.get("NJOBS", 4))))
        if prm["loss"] == "huber": z = zfit(tr, te, "huber", w=wt, extra={"alpha": prm["delta"]})
        else: z = zfit(tr, te, "quantile", w=wt, extra={"alpha": 0.5})
        R = te[["origin", "style_color", "level"]].copy(); R["q50"] = np.clip((4 * te.level.values + 1) * np.exp(z) - 1, 0, None); out.append(R)
    return pd.concat(out)
def score(Q, origins, which):
    Q = Q[Q.origin.isin(origins)]; idx = np.array([KEY.get((o, s), -1) for o, s in zip(Q.origin, Q.style_color)]); ok = idx >= 0
    rows = idx[ok]; f = system(Q.q50.values[ok], rows); b = Bz.loc[rows]
    m = (b.new.values if which == "new" else ~b.new.values)
    return float(np.abs(f - b.y.values)[m].sum() / np.abs(b.R3.values - b.y.values)[m].sum())
CUR = dict(new=dict(loss="huber", delta=0.5, num_leaves=15, min_child=40, lr=0.05, n_est=300, colsample=0.8, subsample=0.8, reg_lambda=5.0, w_new=5.0, wcut=6),
           old=dict(loss="huber", delta=0.5, num_leaves=15, min_child=40, lr=0.05, n_est=300, colsample=0.8, subsample=0.8, reg_lambda=5.0, w_new=1.0, wcut=6))[ROLE]
OUT = f"data/hpo_tree_{ROLE}_trials.csv"
def objective(t):
    lr = t.suggest_float("lr", 0.02, 0.1, log=True)
    prm = dict(loss=t.suggest_categorical("loss", ["huber", "l1log"]), delta=t.suggest_float("delta", 0.2, 1.2), num_leaves=t.suggest_int("num_leaves", 7, 63, log=True),
               min_child=t.suggest_int("min_child", 10, 200, log=True), lr=lr, n_est=int(round(15 / lr)) if True else 300,
               colsample=t.suggest_float("colsample", 0.4, 1.0), subsample=t.suggest_float("subsample", 0.5, 1.0), reg_lambda=t.suggest_float("reg_lambda", 0.1, 50, log=True),
               w_new=t.suggest_float("w_new", 1.0, 12.0, log=True) if ROLE == "new" else t.suggest_float("w_new", 0.2, 3.0, log=True), wcut=t.suggest_categorical("wcut", [3, 6, 9, 12]))
    prm["n_est"] = int(round(t.suggest_float("n_mult", 0.5, 2.0) * 15 / lr))
    t0 = time.time(); s = score(fit_predict(prm, TUNE), TUNE, ROLE)
    pd.DataFrame([dict(trial=t.number, crit=s, secs=round(time.time() - t0), **prm)]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    print(f"{ROLE} trial {t.number}: {s:.4f} ({round(time.time() - t0)} s)", flush=True); return s
if not FIN:
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed={"new": 1, "old": 2}[ROLE], n_startup_trials=15))
    st.enqueue_trial(dict(loss="huber", delta=0.5, num_leaves=15, min_child=40, lr=0.05, n_mult=1.0, colsample=0.8, subsample=0.8, reg_lambda=5.0, w_new=CUR["w_new"], wcut=6))
    st.optimize(objective, n_trials=NT)
else:
    TT = pd.read_csv(OUT).sort_values("crit") if os.path.exists(OUT) else pd.DataFrame(); TOP = int(os.environ.get("TOP", 3))
    if os.environ.get("GRID") == "1":   # a fixed grid of simple versions at the default settings (after the search overfitted the tune origins)
        D0 = dict(CUR); grid = ([dict(D0, w_new=w, wcut=c) for w in (3.0, 5.0, 8.0, 12.0) for c in (3, 6, 9)] if ROLE == "new" else
                                [dict(D0, loss=l, w_new=w, wcut=c) for l in ("huber", "l1log") for w in (0.25, 0.5, 1.0) for c in ((3, 6) if w < 1 else (6,))])
        TT = pd.DataFrame(); cands = [(f"{g['loss']} w{g['w_new']:g} cut{g['wcut']}", g) for g in grid]
    if os.environ.get("GRID") == "2":   # trees trained only on young items' rows, for the new items
        D0 = dict(CUR, w_new=1.0); TT = pd.DataFrame(); cands = [(f"only<{c}m", dict(D0, only=c, min_child=mc)) for c in (6, 9, 12, 18) for mc in (20, 40)]
    else: cands = [("current", CUR)] + [(f"trial {int(r.trial)}", dict(loss=r.loss, delta=r.delta, num_leaves=int(r.num_leaves), min_child=int(r.min_child), lr=r.lr, n_est=int(r.n_est),
                                  colsample=r.colsample, subsample=r.subsample, reg_lambda=r.reg_lambda, w_new=r.w_new, wcut=int(r.wcut))) for r in TT.head(TOP).itertuples()]
    res, keep = [], []
    for nm, prm in cands:
        Q = fit_predict(prm, ALL13); Q2 = fit_predict(prm, ["2026-06"]); QL = fit_predict(dict(prm, lyd=True), ALL13)
        r = dict(cand=nm, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in prm.items()})
        for lab, oo in [("tune", TUNE), ("held out", ALL13[8:]), ("recent 9", ALL13[4:]), ("all 13", ALL13), ("Jul-Oct 25", ["2025-06"]), ("Mar-Jun 26", ["2026-02"])]:
            r[lab] = round(score(Q, oo, ROLE), 4)
        r["recent 9, last year removed"] = round(score(QL, ALL13[4:], ROLE), 4)
        # per-item metrics on the role's items (origins 2025-02 .. 2026-02, items with 6+ origins)
        idx = np.array([KEY.get((o, s), -1) for o, s in zip(Q.origin, Q.style_color)]); ok = idx >= 0; b = Bz.loc[idx[ok]].copy(); b["f"] = system(Q.q50.values[ok], idx[ok])
        b = b[b.new] if ROLE == "new" else b[~b.new]
        cs, r2 = [], []
        for s_, g in b.groupby("style_color"):
            if len(g) < 6 or g.y.std() == 0: continue
            cs.append(np.corrcoef(np.log1p(g.f), np.log1p(g.y))[0, 1]); r2.append(1 - ((g.f - g.y) ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum())
        r["item corr (median)"] = round(float(np.nanmedian(cs)), 3) if cs else None; r["item R2 (median)"] = round(float(np.nanmedian(r2)), 3) if r2 else None; r["items with 6+ origins"] = len(cs)
        res.append(r); print(r, flush=True)
        keep.append(pd.concat([Q, Q2]).assign(cand=nm))
    _sx = {"1": "_grid", "2": "_grid2"}.get(os.environ.get("GRID", ""), ""); pd.concat(keep).to_parquet(f"data/hpo_tree_{ROLE}_finals{_sx}.parquet", index=False); json.dump(res, open(f"data/hpo_tree_{ROLE}_finals{_sx}.json", "w"), indent=1, default=str)
