"""Backtest of simple baselines: origin = end of Jun-2025, target = Jul-Oct 2025 (the exact analogue of the task)."""
import pandas as pd, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = pd.PeriodIndex(M.columns, freq="M")
it = pd.read_parquet("data/items.parquet").set_index("style_color")
it["pub"] = pd.to_datetime(it["published_date"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
it["first_sale"] = pd.to_datetime(it["first_sale"])
it["start"] = it[["pub", "first_sale"]].min(axis=1).clip(lower=pd.Timestamp("2024-01-01"))  # effective start inside the window
T = set(pd.read_csv("data/targets.csv").style_color)
occs = ["Halloween", "4th of July", "Homecoming", "Valentine", "Graduation", "New Year"]
H = 4

def metrics(name, pred, act, idx):
    p, a = pred.loc[idx].sum(axis=1), act.loc[idx].sum(axis=1)
    ape = (p - a).abs() / a.clip(lower=1)
    pm, am = pred.loc[idx].values, act.loc[idx].values
    return dict(model=name, n=len(idx), wape_total=(p - a).abs().sum() / a.sum(), wape_month=np.abs(pm - am).sum() / am.sum(),
                bias=(p.sum() - a.sum()) / a.sum(), med_ape=ape.median(), within20=(ape <= .2).mean(), within50=(ape <= .5).mean())

def features(origin, h):
    """Features known at end of month `origin` for target month origin+h. One row per item live for >=1 full month."""
    cols = [origin - k for k in range(6)]
    X = pd.DataFrame(index=M.index)
    for k, c in enumerate(cols): X[f"lag{k+1}"] = M[c] if c in M.columns else np.nan
    age_d = (origin.to_timestamp(how="end") - it.loc[M.index, "start"]).dt.days
    X["age_m"] = age_d / 30.4
    for k in range(6): X.loc[X.age_m < k, f"lag{k+1}"] = np.nan          # months before launch are missing, not zero
    X["m3"] = X[["lag1", "lag2", "lag3"]].mean(axis=1); X["m6"] = X[[f"lag{k}" for k in range(1, 7)]].mean(axis=1)
    X["trend"] = np.log1p(X.lag1) - np.log1p(X.m3); X["lvl"] = np.log1p(X.m3)
    ty = it.loc[M.index, "product_type"]
    X["ptype"] = ty.astype("category")
    tsum = M.groupby(ty.values).sum()
    # type-level prior-year seasonal ratio: type units in (target month - 12) / type mean units in the 3 months ending (origin - 12)
    tm, o12 = origin + h - 12, origin - 12
    if tm in M.columns and (o12 - 2) in M.columns:
        r = tsum[tm] / tsum[[o12 - 2, o12 - 1, o12]].mean(axis=1).replace(0, np.nan)
        g = M[tm].sum() / M[[o12 - 2, o12 - 1, o12]].mean(axis=1).sum()
        w = tsum[[o12 - 2, o12 - 1, o12]].mean(axis=1); k = w / (w + 2000)       # shrink small types to the global ratio
        X["type_sea"] = np.log((k * r.fillna(g) + (1 - k) * g).reindex(ty.values).values)
        own = (M[tm] + 5) / (M[[o12 - 2, o12 - 1, o12]].mean(axis=1) + 5)
        X["own_sea"] = np.where(it.loc[M.index, "start"] < (o12 - 3).to_timestamp(), np.log(own), np.nan)
    else: X["type_sea"] = np.nan; X["own_sea"] = np.nan
    X["tmonth"] = (origin + h).month; X["h"] = h; X["color_num"] = it.loc[M.index, "color_num"].values
    oc = it.loc[M.index, "occasion"].fillna("")
    for o in occs: X["occ_" + o.split()[0].lower()] = oc.str.contains(o).astype(int).values
    X["y"] = M[origin + h] if (origin + h) in M.columns else np.nan
    X["origin"] = str(origin)
    return X[X.age_m >= 1.0]

def run_fold(origin):
    tgt = [origin + h for h in range(1, H + 1)]; act = M[tgt]
    live = it.index[(it.start <= (origin.to_timestamp(how="end") - pd.Timedelta(days=30)))].intersection(M.index)
    live = live[M.loc[live, [origin - 2, origin - 1, origin]].sum(axis=1) > 0]
    f0 = features(origin, 1).loc[live]; out = {}
    # B1 flat: mean of last (up to) 3 live months
    out["B1 flat last-3m mean"] = pd.DataFrame({t: f0.m3 for t in tgt})
    # B2 seasonal naive (same months last year) where item was live a year ago, else B1
    sn = M[[t - 12 for t in tgt]].loc[live]; sn.columns = tgt
    has = it.loc[live, "start"] < (tgt[0] - 12).to_timestamp()
    out["B2 seasonal naive (fallback flat)"] = sn.where(has, out["B1 flat last-3m mean"])
    # B3 level x type seasonal ratio (prior year, shrunk)
    out["B3 last-3m x type seasonal ratio"] = pd.DataFrame({origin + h: f0.m3 * np.exp(features(origin, h).loc[live, "type_sea"]) for h in range(1, H + 1)})
    # B4 LightGBM, pooled across items & horizons, target = log ratio to recent level
    tr = []
    for o in pd.period_range("2024-03", origin - 1, freq="M"):
        for h in range(1, H + 1):
            if o + h <= origin: tr.append(features(o, h))
    tr = pd.concat(tr); tr = tr[tr.m3 > 0]
    fcols = [c for c in tr.columns if c not in ("y", "origin")]
    tr["ptype"] = tr["ptype"].astype(str).astype("category"); cats = tr["ptype"].cat.categories
    mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8, reg_lambda=5, objective="l1", verbose=-1, random_state=0)
    mdl.fit(tr[fcols], np.log1p(tr.y) - tr.lvl, sample_weight=np.sqrt(tr.m3.clip(lower=1)))
    preds = {}
    for h in range(1, H + 1):
        te = features(origin, h).loc[live]; te["ptype"] = pd.Categorical(te["ptype"].astype(str), categories=cats)
        preds[origin + h] = np.expm1(mdl.predict(te[fcols]) + te.lvl).clip(lower=0)
    out["B4 LightGBM (log-ratio target)"] = pd.DataFrame(preds)
    out["B5 blend 0.5*B3 + 0.5*B4"] = 0.5 * out["B3 last-3m x type seasonal ratio"] + 0.5 * out["B4 LightGBM (log-ratio target)"]
    imp = pd.Series(mdl.booster_.feature_importance("gain"), index=fcols).sort_values(ascending=False)
    return live, act.loc[live], out, imp, len(tr)

for origin in [pd.Period("2025-06"), pd.Period("2026-02")]:
    live, act, out, imp, ntr = run_fold(origin)
    print(f"\n===== ORIGIN {origin} -> forecast {origin+1}..{origin+4} | live items {len(live)} | LGBM train rows {ntr}")
    segs = {"all live items": live, "the 100 targets (live at origin)": live[live.isin(T)],
            "young (<6m at origin)": live[(origin.to_timestamp(how='end') - it.loc[live, 'start']).dt.days < 183],
            "mature (>=12m)": live[(origin.to_timestamp(how='end') - it.loc[live, 'start']).dt.days >= 365]}
    for sname, idx in segs.items():
        if len(idx) == 0: continue
        print(f"-- {sname}")
        print(pd.DataFrame([metrics(k, v, act, idx) for k, v in out.items()]).set_index("model").round(3).to_string())
    print("LGBM gain importance top 10:", (imp / imp.sum()).round(3).head(10).to_dict())
    res = pd.concat({k: v.sum(axis=1) for k, v in out.items()}, axis=1); res["actual"] = act.sum(axis=1); res.to_csv(f"data/backtest_{origin}.csv")
