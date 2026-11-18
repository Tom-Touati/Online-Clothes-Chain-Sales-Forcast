"""D353 (user goal, 18 Nov: "experiment with different losses that can target our issues. log absolute error perhaps on some models. go
through tricks from m5"). The tree member ec_ns (eda/180q_models.py: D198 features + 15 events + launch curves - static descriptors, BASE and
FULL sets averaged, training origins o' + 4 <= o) refitted with different losses / M5 tricks. The current loss IS log absolute error: the
median (quantile 0.5) of z = log((total + 1) / (4 level + 1)). Variants (argv; data/q_backtest_ls_<v>[OSUF].parquet, column q50):
  l1log     baseline: quantile 0.5 on z (= absolute error in logs)                         [reproduces ec_ns]
  l2log     squared error on z (geometric mean)          huber   Huber on z (delta 0.5); huber03 / huber10 deltas 0.3 / 1.0; huber_w05 Huber + weights (4L + 1)^0.5      fair   Fair loss on z
  q55, q60  quantile 0.55 / 0.60 on z (shifts up: our bias is about -6%)
  tweedie   Tweedie (power 1.2) on units with offset log(4 level + 1) (M5's winning objective; predicts the mean)
  poisson   Poisson on units with the same offset
  wape      absolute error on units: L1 on (total + 1) / (4 level + 1) with weight (4 level + 1) = exactly WAPE's loss
  w05, w10  log L1 with weights (4 level + 1)^0.5 / ^1 (M5-style volume weights; the fork's w03 is ^0.3)
  recency   log L1 with time-decay weights 0.85^(months between the training origin and o) (M5: recent data matter more)
  permacro  log L1, one model per macro group (M5 winners: per store x department models), pooled model for groups < 300 rows
  mix_tw    the average in logs of l1log and tweedie (M5: blends of objectives)
  mix_q     the average in logs of l1log and q60
Env NJOBS threads, LYDROP=1 (the last-year-removed gate: training origins o - 12 +/- 1 dropped), OSUF file suffix.
Writes the files; stdout per variant."""
import os, sys, time, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
VARS = sys.argv[1:]; sys.argv = ["eda/180q_models.py"]; ns = {}
exec(compile(open("eda/180q_models.py", encoding="utf-8").read(), "eda/180q_models.py", "exec"), ns)
T, sets, P = ns["T"], ns["sets"], dict(ns["P"]); P["n_jobs"] = int(os.environ.get("NJOBS", 8))
B_, F_ = sets("ec_ns"); ORIG = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
_TF = pd.read_parquet("data/rc_temporal_feats.parquet"); TR = [c for c in _TF.columns if c.startswith("tr_")]   # D367: Fork 2's D347 rhythm block
T = T.merge(_TF[["origin", "style_color"] + TR], on=["origin", "style_color"], how="left")
if os.environ.get("ORIGS"): ORIG = os.environ["ORIGS"].split(",")   # D356: e.g. ORIGS=2026-06 OSUF=_2026 for the deliverable
OSUF = os.environ.get("OSUF", ""); LYD = os.environ.get("LYDROP") == "1"
if os.path.exists("data/tabpfn_wf.parquet") and os.path.exists("data/rc_recsys_preds_x_wf.parquet"):   # level-1 walk-forward predictions as z
    _w = pd.read_parquet("data/tabpfn_wf.parquet").rename(columns={"z_pf": "l1_pf"}); T = T.merge(_w, on=["origin", "style_color"], how="left")
    _r = pd.read_parquet("data/rc_recsys_preds_x_wf.parquet")[["origin", "style_color", "f_aug_none_gbm", "f_aug_plan_gbm"]]; T = T.merge(_r, on=["origin", "style_color"], how="left")
    _L = 4 * T.level.values + 1; T["l1_rc"] = np.log((T.f_aug_none_gbm.values + 1) / _L); T["l1_rcp"] = np.log((T.f_aug_plan_gbm.values + 1) / _L)
T["t_since"] =T.o_ord - pd.Period("2024-01", "M").ordinal   # months since the data start (known at the origin)
# Fork 2: last year's shape month by month (known at the origin: target month t - 12 is at or before the origin)
_M = pd.read_parquet("data/monthly_item.parquet"); _M.columns = [str(c) for c in _M.columns]
_PT = T.drop_duplicates("style_color").set_index("style_color").ptype.astype(str); _MT = _M.groupby(_PT.reindex(_M.index).fillna("na").values).sum()
def _lym(o):   # last year needs real volume (>= 10 units a month in its level window)
    P_ = pd.Period(o, "M"); lv = [str(P_ - 12 - j) for j in range(3)]; tg = [str(P_ - 12 + h) for h in (1, 2, 3, 4)]
    if not all(c in _M.columns for c in lv + tg): return None
    L0 = _M[lv].mean(1); W = _M[tg]; ok = _M[lv + [str(P_ - 15)]].sum(1).gt(0) if str(P_ - 15) in _M.columns else L0.gt(0)
    f = pd.DataFrame({f"ly_m{h}": np.log((W[tg[h - 1]] + 5) / (L0 + 5)) for h in (1, 2, 3, 4)}); f["ly_win"] = np.log((W.sum(1) + 5) / (4 * L0 + 5)); f["ly_topshare"] = (W.max(1) + 1) / (W.sum(1) + 4)
    f = f.where(ok & L0.ge(10), np.nan); LT = _MT[lv].mean(1); WT = _MT[tg]
    ft = pd.DataFrame({f"lyt_m{h}": np.log((WT[tg[h - 1]] + 5) / (LT + 5)) for h in (1, 2, 3, 4)}); return f, ft
_parts = []
for _o in T.origin.astype(str).unique():
    r_ = _lym(_o); m_ = (T.origin.astype(str) == _o).values
    if r_ is None: continue
    f_, ft_ = r_; sub = T.loc[m_, ["style_color", "ptype"]]
    _parts.append(pd.concat([f_.reindex(sub.style_color.values).set_index(sub.index), ft_.reindex(sub.ptype.astype(str).values).set_index(sub.index)], axis=1))
_LF = pd.concat(_parts); LYM = list(_LF.columns); T = T.join(_LF)
# Fork 2 D397 (user: "the ratio of the last year's last 4 months vs the next 4 months on the item and each one of its levels ... (historical or
# seasonal ratio) also we can impute that, or nan for early in the dataset"): hr_* = log((units in o-11..o-8 + 5) / (units in o-15..o-12 + 5)),
# i.e. last year's next four months over last year's previous four, for the item (when it sold through o-15..o-12), its product type, macro
# group, collection and style; hr_imp = the item's own when it exists, else imputed (eda/187 huber_hr fits it from the levels and neighbours).
_MAC = pd.read_parquet("data/features.parquet", columns=["style_color", "macro"]).drop_duplicates("style_color").set_index("style_color").macro.astype(str)
_ITM = pd.read_parquet("data/items.parquet").set_index("style_color")
_GRP = {"type": _PT, "macro": _MAC, "coll": _ITM.product_collection.astype(str), "occ": _ITM.occasion.astype(str), "style": pd.Series(_M.index.str.split("_").str[0], index=_M.index)}
def _hr(o):
    P_ = pd.Period(o, "M"); nx = [str(P_ - 12 + h) for h in (1, 2, 3, 4)]; pv = [str(P_ - 12 - j) for j in range(4)]
    if not all(c in _M.columns for c in nx + pv): return None
    N_, Pv = _M[nx].sum(1), _M[pv].sum(1); live = _M[pv].gt(0).all(1) & (Pv >= 20)
    out = pd.DataFrame({"hr_own": np.log((N_ + 5) / (Pv + 5)).where(live)})
    for g, ser in _GRP.items():
        if ser is None: continue
        key = ser.reindex(_M.index).fillna("na").values; Ng, Pg = N_.groupby(key).transform("sum"), Pv.groupby(key).transform("sum")
        out[f"hr_{g}"] = np.log((Ng + 5) / (Pg + 5)).where(Pg >= 50)
    return out
_hp = []
for _o in T.origin.astype(str).unique():
    r_ = _hr(_o); m_ = (T.origin.astype(str) == _o).values
    if r_ is not None: _hp.append(r_.reindex(T.style_color[m_].values).set_index(T.index[m_]))
_HRF = pd.concat(_hp); HRC = list(_HRF.columns); T = T.join(_HRF)
_PFO = pd.read_parquet("data/tabpfn_oof.parquet") if os.path.exists("data/tabpfn_oof.parquet") else None
# Fork 2 (user: "add features like which discounts and how many days for each of the horizon months ... use it directly in the models"):
# known-future campaign features (eda/227 -> data/campaign_feats.parquet); variants <base>_cmpA / _cmpB / _cmpC add them to both feature sets
if os.path.exists("data/campaign_feats.parquet"):
    _CF = pd.read_parquet("data/campaign_feats.parquet"); T = T.merge(_CF, on=["origin", "style_color"], how="left") if "cdays_h1" not in T.columns else T
CMP = {"A": [f"{k}_h{h}" for k in ("cdays", "cdisc", "sdisc") for h in (1, 2, 3, 4)] + ["cdays_tot", "cdisc_tot", "web_sh"],
       "B": [f"dcd_h{h}" for h in (1, 2, 3, 4)] + ["dcd_tot", "x_cmp"] + [f"x_h{h}" for h in (1, 2, 3, 4)] + ["web_sh"],
       "C": ["x_cmp"]}
def zfit(tr, te, obj, w=None, y="z", extra=None, init=None, init_te=None):
    out = []
    for cols in (B_, F_):
        m = lgb.LGBMRegressor(objective=obj, **(extra or {}), **P).fit(tr[cols], tr[y], sample_weight=w, init_score=init)
        out.append(m.predict(te[cols], raw_score=True) + (init_te if init_te is not None else 0) if init is not None else m.predict(te[cols]))
    return np.mean(out, axis=0)
def pred(tr, te, v, oo):
    global B_, F_
    if "_cmp" in v:   # campaign-feature variants: extend both feature sets, run the base variant, restore
        base, k = v.split("_cmp"); B0, F0_ = B_, F_; B_, F_ = B_ + CMP[k], F_ + CMP[k]
        try: return pred(tr, te, base, oo)
        finally: B_, F_ = B0, F0_
    L_tr, L_te = 4 * tr.level.values + 1, 4 * te.level.values + 1
    if v == "l1log": return zfit(tr, te, "quantile", extra={"alpha": 0.5})
    if v == "l2log": return zfit(tr, te, "l2")
    if v == "huber": return zfit(tr, te, "huber", extra={"alpha": 0.5})
    if v == "huber_tr":   # D367: Huber + the five rhythm inputs (daily ACF lags 1 / 7, weekly ACF lags 1 / 4, best-weekday share); global B_, F_ declared at the top of pred
        _b, _f = B_, F_; B_, F_ = _b + TR, _f + TR
        try: return zfit(tr, te, "huber", extra={"alpha": 0.5})
        finally: B_, F_ = _b, _f
    if v in ("huber03", "huber10"): return zfit(tr, te, "huber", extra={"alpha": {"huber03": 0.3, "huber10": 1.0}[v]})   # D353b: Huber threshold
    if v == "huber_w05": return zfit(tr, te, "huber", w=L_tr ** 0.5, extra={"alpha": 0.5})                              # D353b: Huber + volume weights
    if v.startswith(("huber_lin", "l1log_lin")) and not v.startswith("huber_linin"):   # Fork 2 (user: "try the linear tree"): linear model in each leaf, ridge linear_lambda = suffix
        lam = float(v.split("lin")[1] or 1); ex = {"linear_tree": True, "linear_lambda": lam}
        return zfit(tr, te, "huber", extra={"alpha": 0.5, **ex}) if v.startswith("huber") else zfit(tr, te, "quantile", extra={"alpha": 0.5, **ex})
    if v in ("l1log_new5", "huber_new5", "huber_new3"):   # Fork 2 (user: "retrain with 5 times the loss on new data (has less than 6 months, wasnt there from the start)")
        wt = np.where(tr.age_m.fillna(99).values < 6, 3.0 if v == "huber_new3" else 5.0, 1.0)
        return zfit(tr, te, "quantile" if v.startswith("l1log") else "huber", w=wt, extra={"alpha": 0.5})
    if v in ("huber_t", "huber_rw2", "huber_t_rw2", "huber_rw3"):   # Fork 2 (user: "add time since start of dataset. make the recent year be twice as important in the loss")
        # t_since: months from the data start (Jan 2024) to the origin; rw2: training origins within 12 months of o weigh 2, older 1 (rw3: 3)
        k = 3.0 if v == "huber_rw3" else 2.0; wt = np.where(oo - tr.o_ord.values <= 12, k, 1.0) if "rw" in v else None
        if "_t" in v:
            _b, _f = B_, F_; B_, F_ = _b + ["t_since"], _f + ["t_since"]
            try: return zfit(tr, te, "huber", w=wt, extra={"alpha": 0.5})
            finally: B_, F_ = _b, _f
        return zfit(tr, te, "huber", w=wt, extra={"alpha": 0.5})
    if v in ("huber_linin", "huber_linin_only"):   # Fork 2 (user: "predict totals with a linear model and give them as input"): ridge on z, out of fold
        from sklearn.linear_model import Ridge
        num = [c for c in dict.fromkeys(F_) if c in tr.columns and pd.api.types.is_numeric_dtype(tr[c])]
        med = tr[num].median().fillna(0); mu = tr[num].fillna(med).mean(); sd = tr[num].fillna(med).std().replace(0, 1)
        Z = lambda d: np.clip(((d[num].fillna(med) - mu) / sd).values, -5, 5)
        fold = tr.style_color.str.split("_").str[0].map(lambda s: sum(map(ord, s)) % 5).values; oof = np.zeros(len(tr))
        for k in range(5):
            m = fold == k; oof[m] = Ridge(alpha=10.0).fit(Z(tr[~m]), tr.z.values[~m]).predict(Z(tr[m]))
        tr = tr.assign(lin_z=oof); te = te.assign(lin_z=Ridge(alpha=10.0).fit(Z(tr), tr.z.values).predict(Z(te)))
        if v == "huber_linin_only": return te.lin_z.values
        _b, _f = B_, F_; B_, F_ = _b + ["lin_z"], _f + ["lin_z"]
        try: return zfit(tr, te, "huber", extra={"alpha": 0.5})
        finally: B_, F_ = _b, _f
    if v in ("huber_abs", "l1_abs"):   # Fork 2 (user: "train also on absolute numbers (log?) rather than relative to recent history"): target log(total + 1),
        # the recent level stays an input (lvl, lagr*) but is no longer the reference; back to z = log((F + 1) / (4L + 1))
        tr = tr.assign(za=np.log1p(tr.ytot.values))
        za = zfit(tr, te, "huber" if v == "huber_abs" else "quantile", y="za", extra={"alpha": 0.5})
        return za - np.log(L_te)
    if v in ("huber_pfinit", "huber_pfinit_half", "pf_oof_only"):   # Fork 2 (user: "use tabpfn as initial model, then stack the ensemble afterwards"):
        # the tree boosts from TabPFN's prediction (init_score); TabPFN out of fold on the training rows, fitted on all of them for the test (eda/210)
        o = str(te.origin.iloc[0]); PF_ = _PFO[_PFO.origin == o]
        ztr = tr[["origin", "style_color"]].merge(PF_[PF_.part == "train"].rename(columns={"row_origin": "origin", "origin": "_o"})[["origin", "style_color", "z_pf"]], on=["origin", "style_color"], how="left").z_pf.values
        zte = te[["style_color"]].merge(PF_[PF_.part == "test"][["style_color", "z_pf"]], on="style_color", how="left").z_pf.values
        if v == "pf_oof_only": return zte
        ok = np.isfinite(ztr); tr = tr[ok]; ztr = ztr[ok]
        P0 = dict(P)
        if v == "huber_pfinit_half": P["n_estimators"] = max(int(P.get("n_estimators", 100)) // 2, 50)
        try: return zfit(tr, te, "huber", extra={"alpha": 0.5}, init=ztr, init_te=zte)
        finally: P.clear(); P.update(P0)
    if v.startswith("res_"):   # Fork 2 (user: "ensembling the tabpfn and the recommender twin tree as first layer, then training the rest of the models as
        # secondary stacked on the residuals"): base = mean in z of TabPFN and the recommender twin (walk-forward: each row's prediction was made at
        # that row's own origin); the Huber tree boosts from the base (init_score) and learns the residual. res_base: the base alone.
        # suffixes: p = recommender with the business plan; half = half the rounds; lr = learning rate halved too
        rcol = "l1_rcp" if (v == "res_basep" or v.startswith("res_pfrcp")) else "l1_rc"
        bz = lambda d: np.nanmean(np.stack([d.l1_pf.values, d[rcol].values]), 0)
        btr, bte = bz(tr), bz(te); ok = np.isfinite(btr); bte = np.where(np.isfinite(bte), bte, 0.0)
        if v in ("res_base", "res_basep"): return bte
        if "_dm" in v:   # demeaned: remove each training origin's median residual, so the tree learns item-relative corrections, not date-wide shifts
            r_ = tr.z.values - btr; med = pd.Series(r_).groupby(tr.origin.values).transform("median").values; btr = btr + np.where(np.isfinite(med), med, 0)
            v = v.replace("_dm", "")
        P0 = dict(P)
        if v.endswith("_half"): P["n_estimators"] = 150
        if v.endswith("_lr"): P["n_estimators"] = 150; P["learning_rate"] = 0.025
        try: return zfit(tr[ok], te, "huber", extra={"alpha": 0.5}, init=btr[ok], init_te=bte)
        finally: P.clear(); P.update(P0)
    if v in ("l2o_pf", "l2o_both", "l2o_bothplan"):   # level 2 with TabPFN's cross-fitted (out-of-fold) predictions from eda/210 as the level-1 input
        o = str(te.origin.iloc[0]); PF_ = _PFO[_PFO.origin == o]
        tr = tr.drop(columns=["l1_pf"], errors="ignore").merge(PF_[PF_.part == "train"][["row_origin", "style_color", "z_pf"]].rename(columns={"row_origin": "origin", "z_pf": "l1_pf"}), on=["origin", "style_color"], how="left")
        te = te.drop(columns=["l1_pf"], errors="ignore").merge(PF_[PF_.part == "test"][["style_color", "z_pf"]].rename(columns={"z_pf": "l1_pf"}), on="style_color", how="left")
        add = {"l2o_pf": ["l1_pf"], "l2o_both": ["l1_pf", "l1_rc"], "l2o_bothplan": ["l1_pf", "l1_rcp"]}[v]
        _b, _f = B_, F_; B_, F_ = _b + add, _f + add
        try: return zfit(tr, te, "huber", extra={"alpha": 0.5})
        finally: B_, F_ = _b, _f
    if v.startswith("l2_"):   # Fork 2 (user: "tabpfn + twin tree recommender as beginning, with trees and nn stacked secondarily"): level-2 Huber tree,
        # its inputs + the level-1 walk-forward predictions (each row carries what TabPFN / the recommender predicted at that row's own origin), as z
        # l2_pf: + TabPFN; l2_rc: + recommender (plan missing); l2_both: both; l2_bothplan: both, recommender with the business plan
        add = {"l2_pf": ["l1_pf"], "l2_rc": ["l1_rc"], "l2_both": ["l1_pf", "l1_rc"], "l2_bothplan": ["l1_pf", "l1_rcp"]}[v]
        _b, _f = B_, F_; B_, F_ = _b + add, _f + add
        try: return zfit(tr, te, "huber", extra={"alpha": 0.5})
        finally: B_, F_ = _b, _f
    if v in ("l1_pf_only", "l1_rc_only", "l1_rcp_only"): return te[v[:-5]].values
    if v in ("huber_lym", "huber_seastart", "huber_lym_seastart"):   # Fork 2 (user: "objects that have a huge rise on september october november on both
        # years ... add features that cover that. if the last 2 years had a significant deviation at the same time, we need to start our modeling
        # from that prediction"). lym: last year's shape month by month (ly_m1..4 = log units in target month h a year earlier / last year's
        # level, plus the type's), the share of last year's window in its top month, and the window over the LY level (ly_win).
        # seastart: the tree boosts from last year's window ratio (init = s * ly_win where it exists, s = 0.5), fitting the residual.
        cols = LYM if "lym" in v else []
        _b, _f = B_, F_; B_, F_ = _b + cols, _f + cols
        try:
            if "seastart" not in v: return zfit(tr, te, "huber", extra={"alpha": 0.5})
            s = float(os.environ.get("SEA_S", 0.5)); it_, ite = s * np.clip(np.nan_to_num(tr.ly_win.values), -1, 1), s * np.clip(np.nan_to_num(te.ly_win.values), -1, 1)   # clipped: tiny LY levels give extreme ratios
            return zfit(tr, te, "huber", extra={"alpha": 0.5}, init=it_, init_te=ite)
        finally: B_, F_ = _b, _f
    if v in ("huber_lyres", "huber_lyres_noimp"):   # Fork 2 D396 (user: "we train a model to predict the residual from last year's log value. if the model
        # doesnt have a last year, impute it's last year from all it's features (basically higher level categories, knn, features, etc)").
        # Start = log(LY + 1): LY = the item's own units in the same four months a year earlier, when it was live then (sold >= 5 a month in
        # the 3 months before last year's origin); otherwise imputed as (4 level + 1) x exp(r_hat) - 1, r_hat = a LightGBM fitted on the
        # training rows that have a last year, predicting last year's window over last year's level from every input (type / macro /
        # occasion seasonal ratios, co-purchase neighbours' ratios, attributes, calendar, history). The Huber tree fits log(total + 1) - start.
        # _noimp: rows without a last year start from 4 level instead (no imputation).
        def ly(d):
            o_ = d.origin.astype(str).values; out = np.full(len(d), np.nan); lev = np.full(len(d), np.nan)
            for oo in np.unique(o_):
                P_ = pd.Period(oo, "M"); w_ = [str(P_ - 12 + h) for h in (1, 2, 3, 4)]; l_ = [str(P_ - 12 - j) for j in range(3)]
                if not all(c in _M.columns for c in w_ + l_): continue
                m_ = o_ == oo; sc = d.style_color.values[m_]; W_ = _M[w_].sum(1).reindex(sc).values; Lv = _M[l_].mean(1).reindex(sc).values
                live = _M[l_].gt(0).all(1).reindex(sc).fillna(False).values & (Lv >= 5); out[m_] = np.where(live, W_, np.nan); lev[m_] = np.where(live, Lv, np.nan)
            return out, lev
        (ytr_, ltr_), (yte_, lte_) = ly(tr), ly(te); r_tr = np.log((ytr_ + 5) / (4 * ltr_ + 5)); has = np.isfinite(r_tr)
        if v == "huber_lyres" and has.sum() < 300:   # the earliest origins: no training row has a last year yet; fall back to the type's ratio
            rt = lambda d: np.nan_to_num(d.lfl_type4.values.astype(float)) if "lfl_type4" in d else 0.0
            st_tr = np.where(has, np.log1p(ytr_), np.log(L_tr) + rt(tr)); st_te = np.where(np.isfinite(yte_), np.log1p(yte_), np.log(L_te) + rt(te))
        elif v == "huber_lyres":
            imp = lgb.LGBMRegressor(objective="huber", alpha=0.5, n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, verbose=-1, n_jobs=P["n_jobs"]).fit(tr.loc[has, F_], r_tr[has])
            st_tr = np.where(has, np.log1p(ytr_), np.log(L_tr * np.exp(imp.predict(tr[F_])))); st_te = np.where(np.isfinite(yte_), np.log1p(yte_), np.log(L_te * np.exp(imp.predict(te[F_]))))
        else: st_tr = np.where(has, np.log1p(ytr_), np.log(L_tr)); st_te = np.where(np.isfinite(yte_), np.log1p(yte_), np.log(L_te))
        tr = tr.assign(zr=np.log1p(tr.ytot.values) - st_tr, ly_has=has.astype(float), ly_start=st_tr - np.log(L_tr)); te = te.assign(ly_has=np.isfinite(yte_).astype(float), ly_start=st_te - np.log(L_te))
        print(f"   {te.origin.iloc[0]}: last year on {has.mean():.0%} of training rows, {np.isfinite(yte_).mean():.0%} of test rows", flush=True)
        _b, _f = B_, F_; B_, F_ = _b + ["ly_has", "ly_start"], _f + ["ly_has", "ly_start"]
        try: zr = zfit(tr, te, "huber", y="zr", extra={"alpha": 0.5})
        finally: B_, F_ = _b, _f
        return np.log(np.exp(zr + st_te)) - np.log(L_te)
    if v in ("huber_hr", "huber_hr_route", "huber_hr_route_only"):   # Fork 2 D397: historical ratios as inputs (+ imputed own ratio); _route: a specialist
        # Huber tree trained only on rows whose (own or imputed) historical ratio is significant (|hr| > log 1.3) predicts those test rows,
        # the main tree (with the same inputs) the rest; _route_only prints how many rows are routed
        imp_cols = [c for c in HRC if c != "hr_own"] + [c for c in ("cp_nb_sea4", "dy_nb_sea4", "lfl_type4", "lfl_macro4", "lfl_occ4") if c in tr]
        has = tr.hr_own.notna().values
        if has.sum() >= 300:
            mi = lgb.LGBMRegressor(objective="huber", alpha=0.5, n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, verbose=-1, n_jobs=P["n_jobs"]).fit(tr.loc[has, imp_cols + [c for c in ("tmonth",) if c in tr]], tr.hr_own.values[has])
            icol = imp_cols + [c for c in ("tmonth",) if c in tr]; fi = lambda d: np.where(d.hr_own.notna(), d.hr_own, mi.predict(d[icol]))
        else: fi = lambda d: np.where(d.hr_own.notna(), d.hr_own, d.hr_type if "hr_type" in d else np.nan)
        tr = tr.assign(hr_imp=fi(tr), hr_isimp=tr.hr_own.isna().astype(float)); te = te.assign(hr_imp=fi(te), hr_isimp=te.hr_own.isna().astype(float))
        add = HRC + ["hr_imp", "hr_isimp"]; _b, _f = B_, F_; B_, F_ = _b + add, _f + add
        try:
            z = zfit(tr, te, "huber", extra={"alpha": 0.5})
            if v == "huber_hr": return z
            sig_tr, sig_te = np.abs(np.nan_to_num(tr.hr_imp.values)) > np.log(1.3), np.abs(np.nan_to_num(te.hr_imp.values)) > np.log(1.3)
            print(f"   {te.origin.iloc[0]}: routed {sig_te.mean():.0%} of test rows; specialist trained on {sig_tr.sum()} rows ({tr.hr_own.notna()[sig_tr].mean() if sig_tr.any() else 0:.0%} with an own ratio)", flush=True)
            if sig_tr.sum() >= 300 and sig_te.any(): z[sig_te] = zfit(tr[sig_tr], te[sig_te], "huber", extra={"alpha": 0.5})
            return z
        finally: B_, F_ = _b, _f
    if v == "fair": return zfit(tr, te, "fair", extra={"fair_c": 0.5})
    if v in ("q55", "q60", "q25", "q75"): return zfit(tr, te, "quantile", extra={"alpha": int(v[1:]) / 100})   # D364: q25 / q75 (user: P25 / P75 for all models)
    if v in ("tweedie", "poisson"):
        ex = {"tweedie_variance_power": 1.2} if v == "tweedie" else {}
        raw = zfit(tr, te, v, y="ytot", extra=ex, init=np.log(L_tr), init_te=np.log(L_te))
        return np.log(np.exp(raw) + 1) - np.log(L_te)                     # back to z = log((F + 1) / (4L + 1))
    if v == "wape":
        tr = tr.assign(r=(tr.ytot + 1) / L_tr); return np.log(np.clip(zfit(tr, te, "quantile", w=L_tr, y="r", extra={"alpha": 0.5}), 1e-3, None))
    if v in ("w05", "w10"): return zfit(tr, te, "quantile", w=L_tr ** {"w05": 0.5, "w10": 1.0}[v], extra={"alpha": 0.5})
    if v == "recency": return zfit(tr, te, "quantile", w=0.85 ** (oo - tr.o_ord.values), extra={"alpha": 0.5})
    if v == "mix_tw": return (pred(tr, te, "l1log", oo) + pred(tr, te, "tweedie", oo)) / 2
    if v == "mix_q": return (pred(tr, te, "l1log", oo) + pred(tr, te, "q60", oo)) / 2
    raise ValueError(v)
MAC = pd.read_parquet("data/features.parquet", columns=["style_color", "macro"]).drop_duplicates("style_color").set_index("style_color").macro
T["macro_g"] = T.style_color.map(MAC).fillna("other").astype(str)
def pred_permacro(tr, te, oo):
    z = pred(tr, te, "l1log", oo)
    for g in te.macro_g.unique():
        a, b = tr[tr.macro_g == g], (te.macro_g == g).values
        if len(a) >= 300 and b.any(): z[b] = zfit(a, te[b], "quantile", extra={"alpha": 0.5})
    return z
for v in VARS:
    out, t0 = [], time.time()
    for o in ORIG:
        oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()].copy(); te = T[T.origin == o].copy()
        if LYD: tr = tr[~tr.o_ord.between(oo - 13, oo - 11)]
        assert (tr.o_ord + 4 <= oo).all() and len(te)
        z = pred_permacro(tr, te, oo) if v == "permacro" else pred(tr, te, v, oo)
        R = te[["origin", "style_color", "level", "ytot"]].copy(); R["q50"] = np.clip((4 * te.level.values + 1) * np.exp(z) - 1, 0, None); out.append(R)
    pd.concat(out).to_parquet(f"data/q_backtest_ls_{v}{OSUF}.parquet", index=False); print(v, OSUF, round(time.time() - t0), "s", flush=True)
