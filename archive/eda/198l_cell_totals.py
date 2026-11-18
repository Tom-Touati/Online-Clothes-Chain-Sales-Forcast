"""D281 (fork; user comment on the error study, at "only when the total of every product type within every look is known does it reach
56%": "perhaps we can estimate that and add as a feature"). Can an estimate of the type x look group totals raise the +/-20% hit rate?
  1 how good the group totals are today: the summed item forecasts against the actual group total (recent origins), and R3's
  2 Main's group-total models (D240, data/cells_preds.parquet: item forecasts rescaled fully / partly to a modelled group total): item
    hit rate and WAPE / R3 of each, against the system they were built on (p2c32a)
  3 as a feature: the modelled group total relative to the summed items, the group's forecast growth against R3, the group's size and
    the item's share of it, added to a time-honest residual correction of the submitted system (earlier origins only, origin total kept)
  4 how accurate would the estimate have to be? Items rescaled to the true group total times log-normal noise of a given size
Adds J["cells"] to data/fk_err_analysis.json; stdout -> eda/cell_totals_output.txt."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
D = pd.read_parquet("data/fk_err_master.parquet"); ORIG = sorted(D.origin.unique()); REC = [o for o in ORIG if o >= "2025-06"]; D["rec"] = D.origin.isin(REC); D["oo"] = D.origin.map(lambda o: pd.Period(o, "M").ordinal)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k); y = D.y.values; ar = (D.R3 - D.y).abs().values; rec = D.rec.values
hit = lambda f, m=rec: float((np.abs(f - y)[m] <= 0.2 * y[m]).mean()); wape = lambda f, m=rec: float(np.abs(f - y)[m].sum() / ar[m].sum())
K = ["origin", "ptype", "look"]; g = D.groupby(K); D["cF"] = g.F.transform("sum"); D["cY"] = g.y.transform("sum"); D["cR"] = g.R3.transform("sum"); D["cn"] = g.F.transform("size")
C = D[D.rec].groupby(K).agg(F=("F", "sum"), y=("y", "sum"), R3=("R3", "sum"), n=("F", "size")).reset_index(); C = C[C.y > 0]
OUT = dict(cells=dict(n=int(len(C)), per_origin=r(len(C) / len(REC), 0), med_items=r(C.n.median(), 0), single=r((C.n == 1).mean(), 3), hit=r(((C.F - C.y).abs() <= 0.2 * C.y).mean(), 3), hit_r3=r(((C.R3 - C.y).abs() <= 0.2 * C.y).mean(), 3),
                      wape=r((C.F - C.y).abs().sum() / C.y.sum(), 3), wape_r3=r((C.R3 - C.y).abs().sum() / C.y.sum(), 3), log_sd=r(float(np.log((C.F + 1) / (C.y + 1)).std()), 3),
                      log_sd_w=r(float(np.sqrt(np.average(np.log((C.F + 1) / (C.y + 1)) ** 2, weights=C.y))), 3)))
print("1 type x look groups, recent:", OUT["cells"])
# 2 Main's modelled group totals as rescaling targets (built on p2c32a)
CP = pd.read_parquet("data/cells_preds.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([D.origin, D.style_color]))
LAB = {"F": "system then (p2c32a), items as forecast", "type_look:c2_full": "rescaled fully to the modelled group total", "type_look:c2_p5": "rescaled half-way", "type_look:c2_p3": "rescaled 30% of the way",
       "type_look:c3_p5": "residual model of the group, half strength", "type_look:mix": "mix of modelled and summed total (inverse variance)"}
OUT["rescale"] = [dict(variant=lab, hit=r(hit(CP[c].values), 3), wape=r(wape(CP[c].values), 4)) for c, lab in LAB.items()]
D["F_or"] = D.F * D.cY / D.cF; OUT["oracle_hit"] = r(hit(D.F_or.values), 3); OUT["oracle_wape"] = r(wape(D.F_or.values), 4)
print("2 rescaling to a modelled group total:"); [print("   ", d) for d in OUT["rescale"]]; print("    oracle (true group totals):", OUT["oracle_hit"], OUT["oracle_wape"])
# 3 as features in a time-honest residual correction of the submitted system
D["res"] = np.log((D.y + 1) / (D.F + 1)); D["res_c"] = D.res - D.groupby("origin").res.transform("mean"); D["llv"] = np.log1p(D.level); D["lf"] = np.log((D.F + 1) / (D.R3 + 1)); D["macro_c"] = D.macro.astype("category").cat.codes
D["c_mult"] = np.log(CP["type_look:c2_full"].values / CP["F"].values); D["c_growth"] = np.log((D.cF + 1) / (D.cR + 1)); D["c_size"] = np.log1p(D.cn); D["c_share"] = D.F / D.cF; D["c_rel"] = D.lf - D.c_growth
lk = D.groupby(["origin", "look"]); D["l_growth"] = np.log((lk.F.transform("sum") + 1) / (lk.R3.transform("sum") + 1)); ty = D.groupby(["origin", "ptype"]); D["t_growth"] = np.log((ty.F.transform("sum") + 1) / (ty.R3.transform("sum") + 1))
BASEF = ["llv", "age_m", "lf", "spread", "bf_window", "macro_c", "disc_web", "size_run", "sh_tiktok", "sh_retail", "sh_partner", "n_stores", "trend91"]; CF = ["c_mult", "c_growth", "c_size", "c_share", "c_rel", "l_growth", "t_growth"]
PR = dict(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=6)
def th_corr(cols, shrink=0.5):
    f = D.F.values.copy()
    for o in REC:
        tr = D[D.oo + 4 <= pd.Period(o).ordinal]; te = D[D.origin == o]
        if len(tr) < 500: continue
        v = (te.F + 1) * np.exp(shrink * lgb.LGBMRegressor(**PR).fit(tr[cols], tr.res_c, sample_weight=np.log1p(tr.R3)).predict(te[cols])) - 1; f[(D.origin == o).values] = v * te.F.sum() / v.sum()
    return f
fb = th_corr(BASEF); fc = th_corr(BASEF + CF); rng = np.random.default_rng(0)
def boot(fa, fb_, B=1000):
    Z = pd.DataFrame({"s": D.style_color, "a": np.abs(fa - y), "b": np.abs(fb_ - y), "r": ar, "ha": np.abs(fa - y) <= 0.2 * y, "hb": np.abs(fb_ - y) <= 0.2 * y, "n": 1.0})[rec].groupby("s").sum().values; n = len(Z); T = Z[rng.integers(0, n, (B, n))].sum(1)
    d = T[:, 1] / T[:, 2] - T[:, 0] / T[:, 2]; hd = (T[:, 4] - T[:, 3]) / T[:, 5]; return [r(np.percentile(d, 2.5)), r(np.percentile(d, 97.5))], [r(np.percentile(hd, 2.5), 3), r(np.percentile(hd, 97.5), 3)]
ci, hci = boot(fb, fc)
OUT["feature"] = dict(system=dict(wape=r(wape(D.F.values)), hit=r(hit(D.F.values), 3)), ref=dict(wape=r(wape(fb)), hit=r(hit(fb), 3)), with_cells=dict(wape=r(wape(fc)), hit=r(hit(fc), 3)), ci=ci, hit_ci=hci,
                      rho={c: r(np.nanmean([__import__("scipy.stats").stats.spearmanr(gg[c], gg.res, nan_policy="omit")[0] for _, gg in D[D.rec].groupby("origin")]), 3) for c in CF})
print("3 as features (time-honest):", OUT["feature"])
# 4 how accurate must the estimate be?
cells = D.groupby(K).ngroup().values; nc = cells.max() + 1; sims = []
for sd in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
    hs, ws = [], []
    for _ in range(15 if sd > 0 else 1):
        f = D.F.values * (D.cY.values * np.exp(rng.normal(0, sd, nc))[cells]) / D.cF.values; hs.append(hit(f)); ws.append(wape(f))
    sims.append(dict(sd=sd, hit=r(np.mean(hs), 3), wape=r(np.mean(ws), 4)))
OUT["noise"] = sims; OUT["hit_now"] = r(hit(D.F.values), 3); print("4 rescaling to a noisy group total:", sims, "| now", OUT["hit_now"], "| our group totals' log error SD", OUT["cells"]["log_sd"])
J = json.load(open("data/fk_err_analysis.json")); J["cells"] = OUT; json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
