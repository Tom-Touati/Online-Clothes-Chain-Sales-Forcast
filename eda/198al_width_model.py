"""D370 (fork; attempt 1 of Fork 2's delegation, user in Fork 2's session: "delegate these attempts"): a width model for the item ranges.
Predict each item's absolute log miss |log((y + 1) / (P50 + 1))| of the candidate system (D358 / nx5 P50) from what is known at the origin, then
set the ranges as log bounds lP50 -/+ k * w_hat, with each side's k fitted on other origins (25% / 10% of actuals outside on each side).
  width sources   pf2c     the conformal TabPFN member's own log widths (the adopted D366 base; separate lower / upper widths)
                  const    one width for every item (the ranges are then the plain quantiles of the miss; per segment = segment quantiles)
                  spread   the five members' disagreement (fk_err_master 'spread')
                  model    LightGBM on |miss| (L2, 15 leaves, 80 rows per leaf, 400 rounds at 0.03): the members' spread and their own stated
                           widths (pf2c, pf2d32, tree, recommender), level, P50 vs R3, age, history, sales entropy / rhythm / shape (Fork 2's
                           eda/209 te_ / tr_ / tm_ / td_ features), discount, channel shares, stores, trend, peak, out-of-stock, size counts
  calibration     global (one k per side) or per segment (age band x volume third, as D328; global k below 50 training rows)
  protocols       loo     every other origin (as the adopted D366 factors)
                  purged  only origins at least 4 months away (no overlap of the 4-month windows; the width model's own protocol for the
                          like-for-like comparison)
                  honest  only closed origins (o' + 4 <= o); scored on 2025-07..2026-02, where at least two closed origins exist
  model k         the width model's k is fitted on its out-of-fold predictions for the training rows (3 origin folds inside the training set,
                  or leave-one-origin-out with fewer than 6 training origins), so k is not fitted on in-sample widths
Scores: P25-P75 and P10-P90 below / inside / above, pinball loss in units as a ratio to R3's, log interval score, mean log width, inside share
by segment (min / max / sd over the 12 age x volume cells), inside share by third of the predicted width, Spearman of each width with the miss.
Second pass (arguments obj=l1 tag=_l1): the width model fitted on the median of |miss| (L1), less pulled by the largest misses, plus
  cqr      conformalised quantile regression on the signed miss: LightGBM quantile models of log((y + 1) / (P50 + 1)) at 0.25 / 0.75 (0.10 / 0.90)
           on the same inputs, then each side shifted on the out-of-fold training predictions so that 25% (10%) fall outside (globally or per
           segment). Unlike a width, it can move the range up or down per item.
Writes data/fk_width_model_backtest{tag}.parquet; adds J["width_model{tag}"]; stdout -> eda/width_model_output{tag}.txt."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb
ARG = dict(a.split("=") for a in sys.argv[1:]); OBJ = ARG.get("obj", "l2"); TAG = ARG.get("tag", ""); CQR = OBJ != "l2"
sys.path.insert(0, "eda"); import fk_rangekit as K
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
B = K.load(); ORIG = sorted(B.origin.unique())
TF = pd.read_parquet("data/rc_temporal_feats.parquet"); B = B.merge(TF, on=["origin", "style_color"], how="left", validate="1:1")
lw = lambda a, b: np.log((B[b].values + 1) / (B[a].values + 1))
B["w_pf2c"] = B.pw_q25 + B.pw_q75; B["w_pf2d32"] = lw("pf2d32_q25", "pf2d32_q75"); B["w_tree"] = lw("tree_q25", "tree_q75"); B["w_rc"] = lw("rc_q25", "rc_q75")
B["l_lev"] = np.log1p(B.level); B["dev_r3"] = B.lP - np.log1p(B.R3); B["adev_r3"] = B.dev_r3.abs()
X = ["lP", "l_lev", "dev_r3", "adev_r3", "spread", "w_pf2c", "w_pf2d32", "w_tree", "w_rc", "age_m", "hist_months", "has_ly", "days_live", "pre2024", "newest_colour",
     "disc_item", "disc_rel", "disc_mom", "sh_tiktok", "sh_retail", "sh_partner", "sh_web", "n_stores", "n_partners", "trend91", "peak_r", "lagr1", "lagr3", "own_sea",
     "bf_window", "top_day_share", "oos_28", "cancel_rate", "n_sizes_life", "multi_share",
     "te_ent_d", "te_ent_w", "te_spec", "tr_acf1_d", "tr_acf7_d", "tr_acf1_w", "tr_acf4_w", "tr_wkday_max", "tm_cv_d", "tm_cv_w", "tm_zero_d", "tm_maxmean_d",
     "td_slope4", "td_slope13", "td_wk_since_peak"]
miss = [c for c in X if c not in B.columns]; assert not miss, miss
XM = B[X].astype(float).values; A = np.abs(B.e.values); e = B.e.values
PRM = dict(objective=OBJ, num_leaves=15, min_data_in_leaf=80, learning_rate=0.03, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=3, verbose=-1, seed=0)
fit = lambda m: lgb.train(PRM, lgb.Dataset(XM[m], A[m]), 400)
fitq = lambda m, al: lgb.train({**PRM, "objective": "quantile", "alpha": al}, lgb.Dataset(XM[m], e[m]), 400)


def cqr_preds(how, tail):
    """Test-origin quantile predictions of the signed miss (lower, upper) and their out-of-fold training versions, per test origin."""
    tl, th, oof = np.full(len(B), np.nan), np.full(len(B), np.nan), {}
    for o in ORIG:
        tr = K.train_mask(B, o, how); te = (B.origin == o).values; to = sorted(B.origin[tr].unique())
        if len(to) < 2: continue
        tl[te] = fitq(tr, tail).predict(XM[te]); th[te] = fitq(tr, 1 - tail).predict(XM[te])
        folds = [[x] for x in to] if len(to) < 6 else [to[i::3] for i in range(3)]; il, ih = np.full(len(B), np.nan), np.full(len(B), np.nan)
        for f in folds:
            mf = B.origin.isin(f).values & tr; il[mf] = fitq(tr & ~mf, tail).predict(XM[mf]); ih[mf] = fitq(tr & ~mf, 1 - tail).predict(XM[mf])
        oof[o] = (il, ih)
    return tl, th, oof


def cqr_bounds(how, per_seg, tail, P_):
    tl, th, oof = P_; lo, hi = np.full(len(B), np.nan), np.full(len(B), np.nan); lP = B.lP.values
    for o in ORIG:
        tr = K.train_mask(B, o, how); te = (B.origin == o).values
        if o not in oof: continue
        il, ih = oof[o]; sh = lambda m: (float(np.quantile(e[m] - il[m], tail)), float(np.quantile(e[m] - ih[m], 1 - tail)))
        dg = sh(tr)
        for sg in (B.seg[te].unique() if per_seg else [None]):
            ms = te & ((B.seg == sg).values if sg else True); mtr = tr & ((B.seg == sg).values if sg else True)
            d = sh(mtr) if (sg is None or mtr.sum() >= 50) else dg
            lo[ms] = lP[ms] + tl[ms] + d[0]; hi[ms] = lP[ms] + th[ms] + d[1]
    return lo, hi
pred = lambda mdl, m: np.maximum(mdl.predict(XM[m]), 0.03)


def model_widths(how):
    """Test-origin widths and out-of-fold training widths (for k) per test origin."""
    wt, oof = np.full(len(B), np.nan), {}
    for o in ORIG:
        tr = K.train_mask(B, o, how); te = (B.origin == o).values; to = sorted(B.origin[tr].unique())
        if len(to) < 2: continue
        wt[te] = pred(fit(tr), te)
        folds = [[x] for x in to] if len(to) < 6 else [to[i::3] for i in range(3)]; w_in = np.full(len(B), np.nan)
        for f in folds:
            mf = B.origin.isin(f).values & tr; w_in[mf] = pred(fit(tr & ~mf), mf)
        oof[o] = w_in
    return wt, oof


def bounds(how, src, per_seg, tail, wt=None, oof=None):
    lo, hi = np.full(len(B), np.nan), np.full(len(B), np.nan); lP = B.lP.values
    for o in ORIG:
        tr = K.train_mask(B, o, how); te = (B.origin == o).values
        if B.origin[tr].nunique() < 2: continue
        if src == "pf2c": wl_tr, wu_tr = (B["pw_q25" if tail == 0.25 else "pw_q10"].values, B["pw_q75" if tail == 0.25 else "pw_q90"].values); wl_te, wu_te = wl_tr, wu_tr
        elif src == "const": wl_tr = wu_tr = wl_te = wu_te = np.ones(len(B))
        elif src == "spread": wl_tr = wu_tr = wl_te = wu_te = np.maximum(B.spread.values, 0.03)
        elif src == "model":
            if np.isnan(wt[te]).all(): continue
            wl_tr = wu_tr = oof[o]; wl_te = wu_te = wt
        kg = K.factors(e[tr], wl_tr[tr], wu_tr[tr], tail)
        if not per_seg:
            lo[te] = lP[te] - kg[0] * wl_te[te]; hi[te] = lP[te] + kg[1] * wu_te[te]; continue
        for sg in B.seg[te].unique():
            ms = te & (B.seg == sg).values; mtr = tr & (B.seg == sg).values
            k = K.factors(e[mtr], wl_tr[mtr], wu_tr[mtr], tail) if mtr.sum() >= 50 else kg
            lo[ms] = lP[ms] - k[0] * wl_te[ms]; hi[ms] = lP[ms] + k[1] * wu_te[ms]
    return lo, hi


W = {h: model_widths(h) for h in ["purged", "honest"]}; print("width model fitted", flush=True)
P = {"recent 9": (B.origin >= "2025-06").values, "all 13": np.ones(len(B), bool), "2025-07..2026-02": (B.origin >= "2025-07").values}
METH = [("pf2c widths, per segment (adopted D366)", "pf2c", True), ("pf2c widths, global", "pf2c", False), ("one width, per segment", "const", True),
        ("member spread, global", "spread", False), ("member spread, per segment", "spread", True), ("width model, global", "model", False), ("width model, per segment", "model", True)]
if CQR:
    METH += [("conformalised quantile regression, global", "cqr", False), ("conformalised quantile regression, per segment", "cqr", True)]
    CQ = {(h, t): cqr_preds(h, t) for h in ["purged", "honest"] for t in [0.25, 0.10]}; print("quantile models fitted", flush=True)
rows, BND = [], {}
for how in ["loo", "purged", "honest"]:
    for lab, src, ps in METH:
        if src in ("model", "cqr") and how == "loo": continue
        wt, oof = W.get(how, (None, None))
        for rng, tail, a in [("P25-P75", 0.25, 0.5), ("P10-P90", 0.10, 0.2)]:
            lo, hi = cqr_bounds(how, ps, tail, CQ[(how, tail)]) if src == "cqr" else bounds(how, src, ps, tail, wt, oof); BND[(how, lab, rng)] = (lo, hi)
            for pn, m in P.items():
                if (how == "honest") != (pn == "2025-07..2026-02"): continue
                rows.append(dict(protocol=how, method=lab, range=rng, period=pn, **K.score(B, lo, hi, a, m & np.isfinite(lo))))
R = pd.DataFrame(rows)
for rng in ["P25-P75", "P10-P90"]:
    print(f"\n=== {rng} ==="); print(R[R.range == rng].drop(columns=["range", "rows_ok"]).to_string(index=False))
# how well each width ranks the misses (recent origins), and coverage by third of the purged width model's prediction
rec = P["recent 9"]; wm = W["purged"][0]
sp = {nm: round(float(pd.Series(v[rec]).corr(pd.Series(A[rec]), method="spearman")), 3) for nm, v in [("pf2c width", B.w_pf2c.values), ("member spread", B.spread.values), ("width model (purged)", wm), ("width model (honest)", W["honest"][0])]}
print("\nSpearman with the absolute log miss, recent origins:", sp)
B["wm_third"] = pd.Series(wm).groupby(B.origin).transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"])).astype(str).values
cond = []
for (how, lab, rng), (lo, hi) in BND.items():
    if how != "purged": continue
    ins = (B.ly.values >= lo) & (B.ly.values <= hi)
    cond.append(dict(method=lab, range=rng, **{f"inside_{t}": round(float(ins[rec & (B.wm_third == t).values].mean()), 3) for t in ["low", "mid", "high"]}))
C = pd.DataFrame(cond); print("\ninside share by third of the predicted width (purged, recent):"); print(C.to_string(index=False))
mA = fit(np.ones(len(B), bool)); imp = pd.Series(mA.feature_importance("gain"), index=X).sort_values(ascending=False); imp = (imp / imp.sum()).round(3)
print("\nwidth model, share of gain (fitted on all 13 origins):"); print(imp.head(15).to_string())
O = B[["origin", "style_color", "y", "P50", "R3", "seg"]].copy(); O["w_model_purged"] = wm; O["w_model_honest"] = W["honest"][0]
for (how, lab, rng), (lo, hi) in BND.items():
    if lab.startswith("width model") or lab.startswith("pf2c widths, per") or lab.startswith("conformalised"): tg = f"{how}|{lab}|{rng}"; O[tg + "|lo"] = np.exp(lo) - 1; O[tg + "|hi"] = np.exp(hi) - 1
O.to_parquet(f"data/fk_width_model_backtest{TAG}.parquet")
J = json.load(open("data/fk_err_analysis.json")); J["width_model" + TAG] = dict(rows=rows, spearman=sp, conditional=cond, importance=imp.head(20).to_dict())
json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0, default=float); print("saved")
