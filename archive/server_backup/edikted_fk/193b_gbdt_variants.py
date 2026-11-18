"""D206 (fork; user: "explore all the M5 initiatives"): M5-style gradient-boosting variants of the D198 quantile model (D159 features +
the dynamic co-occurrence neighbours). Same task (z = log((4-month total + 1) / (4 x level + 1))), same 13 time-honest origins, same
7 quantiles, the two feature sets (BASE, FULL) averaged and the quantiles sorted, as eda/193. Variants (argv):
  base          LightGBM as D198 (reproduces data/q_backtest_fkdyn_dyn.parquet)
  seeds5        F3 bagging: 5 LightGBM seeds (subsample / colsample draws differ), averaged in z
  win12 / win18 F3 training window: only rows whose origin is within the last 12 / 18 months before the test origin
  winmix        F3 multi-window: average of all-history, 18-month and 12-month models (M5 winners' multi-window averaging)
  dart          LightGBM DART boosting (drop-out trees), 300 rounds
  xgb           XGBoost hist, reg:quantileerror (one multi-quantile model per feature set), depth 4, 400 rounds
  cat           CatBoost MultiQuantile, depth 6, 600 rounds, native categoricals
  tuned         LightGBM with parameters from data/fk_lgb_tuned.json (F2: Optuna on the four early origins, eda/193c)
  wvol03 / wvol05 / wvol1   M5 metric weighting: rows weighted by (4 x level + 1)^0.3 / 0.5 / 1 (volume, as WRMSSE / units WAPE weigh them)
  tw12 / tw15 / tw18   Tweedie LightGBM on units with the level offset (variance power 1.2 / 1.5 / 1.8; Main's planned diversity member, merged into F1)
  sel           LightGBM on the pruned feature lists in data/fk_feat_select.json (F4: eda/193d)
  ... combinations with '+', e.g. tuned+seeds5, sel+seeds5
Writes out/q_backtest_gb_<variant>.parquet (q_backtest layout). Usage: python eda/193b_gbdt_variants.py <variant> [...] [jobs=4]"""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
KW = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); VARS = [a for a in sys.argv[1:] if "=" not in a] or ["base"]; JOBS = int(KW.get("jobs", 4))
T = pd.read_parquet("data/fk_explain_table.parquet"); META = json.load(open("data/fk_explain_meta.json"))
DY = ["dy_nb_sea4", "dy_nb_lvl", "dy_nb_mom3", "dy_nb_yoy"]
T = T.merge(pd.read_parquet("data/fk_dyn_feats.parquet")[["origin", "style_color"] + DY], on=["origin", "style_color"], how="left")
BASE0, FULL0 = META["BASE"] + DY, META["FULL"] + DY; CAT = META["CAT"]; P0 = dict(META["PARAMS"]); P0["n_jobs"] = JOBS
for c in CAT: T[c] = T[c].astype("category")
QS = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]; ORIG = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
if "orig" in KW: ORIG = KW["orig"].split(",")   # D209: e.g. orig=2026-06 for the deliverable (training rows: outcome known by the origin)


WPOW = [0.0]   # M5 metric weighting: sample weight = (4 x level + 1) ^ WPOW (units WAPE weighs items by volume; the log loss does not)


def fit_lgb(tr, te, cols, P):
    w = (4 * tr.level.values + 1) ** WPOW[0] if WPOW[0] else None
    return np.column_stack([lgb.LGBMRegressor(objective="quantile", alpha=q, **P).fit(tr[cols], tr.z, sample_weight=w).predict(te[cols]) for q in QS])


TWP = [1.5]   # Tweedie variance power (M5 winners' objective on units)


def fit_tw(tr, te, cols, P):
    """Tweedie LightGBM on 4-month units with the recency level as offset (log(4 x level + 1)); its mean is used for every quantile."""
    off = lambda d: np.log(4 * d.level.values + 1)
    m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=TWP[0], **P).fit(tr[cols], tr.ytot.values, init_score=off(tr))
    mu = np.exp(m.predict(te[cols], raw_score=True) + off(te)); z = np.log((mu + 1) / (4 * te.level.values + 1))
    return np.repeat(z[:, None], len(QS), 1)


def fit_xgb(tr, te, cols, P):
    import xgboost as xgb
    X = lambda d: d[cols].copy()
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(QS), tree_method="hist", max_depth=4, learning_rate=0.05, n_estimators=400,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=5, min_child_weight=10, enable_categorical=True, n_jobs=JOBS, random_state=0)
    return m.fit(X(tr), tr.z).predict(X(te))


def fit_cat(tr, te, cols, P):
    from catboost import CatBoostRegressor, Pool
    cc = [c for c in cols if c in CAT]
    prep = lambda d: d[cols].assign(**{c: d[c].astype(str) for c in cc})
    m = CatBoostRegressor(loss_function="MultiQuantile:alpha=" + ",".join(map(str, QS)), depth=6, iterations=600, learning_rate=0.06, l2_leaf_reg=5,
                          random_seed=0, verbose=0, thread_count=JOBS)
    m.fit(Pool(prep(tr), tr.z.values, cat_features=cc)); return m.predict(Pool(prep(te), cat_features=cc))


def run(var):
    parts = var.split("+"); WPOW[0] = 0.0; P = dict(P0); BASE, FULL = list(BASE0), list(FULL0); fitter = fit_lgb; seeds = [0]; wins = [None]
    for p in parts:
        if p == "seeds5": seeds = [0, 1, 2, 3, 4]
        elif p in ("win12", "win18"): wins = [int(p[3:])]
        elif p == "winmix": wins = [None, 18, 12]
        elif p == "dart": P.update(boosting_type="dart", drop_rate=0.1, skip_drop=0.5)
        elif p == "xgb": fitter = fit_xgb
        elif p == "cat": fitter = fit_cat
        elif p == "tuned": P.update(json.load(open("data/fk_lgb_tuned.json"))["params"])
        elif p == "sel": S = json.load(open("data/fk_feat_select.json")); BASE, FULL = S["BASE"], S["FULL"]
        elif p == "sels": S = json.load(open("data/fk_feat_select_stack.json")); BASE, FULL = S["BASE"], S["FULL"]   # F4b: stack-aware pruning (eda/193e)
        elif p in ("tw12", "tw15", "tw18"): fitter = fit_tw; TWP[0] = int(p[2:]) / 10
        elif p.startswith("wvol"): WPOW[0] = {"wvol03": 0.3, "wvol05": 0.5, "wvol1": 1.0}[p]
        elif p != "base": raise ValueError(p)
    out, t0 = [], time.time()
    for o in ORIG:
        oo = pd.Period(o).ordinal; te = T[T.origin == o]; z = np.zeros((len(te), len(QS))); k = 0
        for w in wins:
            tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna() & ((T.o_ord > oo - 4 - w) if w else True)]
            for s in seeds:
                for cols in (BASE, FULL):
                    Ps = dict(P, random_state=s); z += fitter(tr, te, cols, Ps); k += 1
        z = np.sort(z / k, axis=1); R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
        for j, q in enumerate(QS): R[f"q{int(q * 100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(z[:, j]) - 1, 0, None)
        out.append(R)
    tag = var.replace("+", "_") + ("_" + KW["orig"].replace(",", "_") if "orig" in KW else ""); pd.concat(out).to_parquet(f"out/q_backtest_gb_{tag}.parquet")
    R = pd.concat(out); R = R[R.ytot.notna()] if R.ytot.notna().any() else R.assign(ytot=1.0); rec = R[R.origin >= "2025-06"] if (R.origin >= "2025-06").any() else R
    print(var, f"{time.time() - t0:.0f}s", "| P50 WAPE recent", round((rec.q50 - rec.ytot).abs().sum() / rec.ytot.sum(), 4), "all", round((R.q50 - R.ytot).abs().sum() / R.ytot.sum(), 4), flush=True)


for v in VARS: run(v)
