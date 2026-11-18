"""D285, D286 (fork; two user comments on the error study).
A  "we can pick the top models for each item based on the past"  For every item at every origin, the five members' errors on that
   same item at earlier origins whose windows have closed (o' + 4 <= o); then: the item's best member, its best two, its best three,
   weights by the item's past errors, the same shrunk towards equal weights, the best member at the latest closed origin only, and
   the same choices pooled by style and by product type. Items with no closed history keep the plain average. Also: how often an
   item's best member stays its best (persistence), and how much history there is.
B  "All five models too low, 25% of forecasts: how do we handle that"  What these forecasts are, how far above the highest member the
   actual lands, whether the stated range covers them, whether they can be flagged at the origin (time-honest classifier), and how much
   of their error the plans and a month-1 re-forecast remove (data/fk_err_voi_preds.parquet, D276).
Adds J["itempick"], J["alllow"] to data/fk_err_analysis.json; stdout -> eda/item_pick_all_low_output.txt."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb, warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); ORIG = sorted(D.origin.unique()); REC = [o for o in ORIG if o >= "2025-06"]; D["rec"] = D.origin.isin(REC); D["oo"] = D.origin.map(lambda o: pd.Period(o, "M").ordinal)
MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; X = D[MEM].values.astype(float); y = D.y.values; ar = (D.R3 - D.y).abs().values; rec = D.rec.values; corr = (D.F / D.F0).values; base = X.mean(1)
E = np.abs(np.log((X + 1) / (y[:, None] + 1))); D["tl"] = D.ptype.astype(str) + "|" + D.look.astype(str); rng = np.random.default_rng(0)
def past_err(key, o, last_only=False):
    """mean past error of each member per key (item / style / group), from origins whose windows have closed; n = closed origins used"""
    past = (D.oo + 4 <= pd.Period(o).ordinal).values.copy()
    if last_only and past.any(): past = past & (D.oo == D.oo[past].max()).values
    if not past.any(): return None
    P = pd.DataFrame(E[past], columns=MEM); P["k"] = D[key].values[past]; P["o"] = D.origin.values[past]; g = P.groupby("k"); M = g[MEM].mean(); M["n"] = g.o.nunique(); return M
def pick(key, mode, min_n=1, last_only=False, shrink=3.0):
    f = base.copy(); used = np.zeros(len(D), bool)
    for o in ORIG:
        M = past_err(key, o, last_only)
        if M is None: continue
        te = np.where((D.origin == o).values)[0]; J_ = M.reindex(D[key].values[te]); ok = (J_.n.values >= min_n); e = J_[MEM].values[ok]; ix = te[ok]
        if not len(ix): continue
        used[ix] = True; order = np.argsort(e, 1); xr = X[ix]
        if mode == "best": f[ix] = xr[np.arange(len(ix)), order[:, 0]]
        elif mode == "top2": f[ix] = np.take_along_axis(xr, order[:, :2], 1).mean(1)
        elif mode == "top3": f[ix] = np.take_along_axis(xr, order[:, :3], 1).mean(1)
        elif mode == "drop_worst": f[ix] = np.take_along_axis(xr, order[:, :4], 1).mean(1)
        else:
            w = (1 / np.maximum(e, 0.02)) ** 2; w = w / w.sum(1, keepdims=True)
            if mode == "weights_shrunk": n = J_.n.values[ok][:, None]; w = (n * w + shrink * 0.2) / (n + shrink)
            f[ix] = (xr * w).sum(1)
    return f, used
hit = lambda f, m=rec: float((np.abs(f - y)[m] <= 0.2 * y[m]).mean()); wape = lambda f, m=rec: float(np.abs(f - y)[m].sum() / ar[m].sum()); bias = lambda f, m=rec: float(f[m].sum() / y[m].sum() - 1)
def boot(fa, fb_, B=1000):
    Z = pd.DataFrame({"s": D.style_color, "a": np.abs(fa - y), "b": np.abs(fb_ - y), "r": ar, "ha": np.abs(fa - y) <= 0.2 * y, "hb": np.abs(fb_ - y) <= 0.2 * y, "n": 1.0})[rec].groupby("s").sum().values; n = len(Z); T = Z[rng.integers(0, n, (B, n))].sum(1)
    d = T[:, 1] / T[:, 2] - T[:, 0] / T[:, 2]; hd = (T[:, 4] - T[:, 3]) / T[:, 5]; return [r(np.percentile(d, 2.5)), r(np.percentile(d, 97.5))], [r(np.percentile(hd, 2.5), 3), r(np.percentile(hd, 97.5), 3)]
VAR = [("the item's best model so far", "style_color", "best", {}), ("the item's best two", "style_color", "top2", {}), ("the item's best three", "style_color", "top3", {}), ("drop the item's worst model", "style_color", "drop_worst", {}),
       ("weights from the item's past errors", "style_color", "weights", {}), ("the same weights, shrunk towards equal", "style_color", "weights_shrunk", {}), ("the item's best model at the latest closed origin", "style_color", "best", dict(last_only=True)),
       ("the item's best two, items with 3+ closed origins only", "style_color", "top2", dict(min_n=3)), ("the style's best two (all colours)", "style", "top2", {}), ("the product type's best two", "ptype", "top2", {}), ("the product type's best model", "ptype", "best", {})]   # looks are re-clustered at every origin, so a type x look group has no history of its own
rows = [dict(variant="plain average of the five (submitted stack)", wape=r(wape(base)), hit=r(hit(base), 3), bias=r(bias(base), 3), wape_all=r(wape(base, np.ones(len(D), bool))), cover=1.0)]; FV = {}
for lab, key, mode, kw in VAR:
    f, used = pick(key, mode, **kw); FV[lab] = (f, used); ci, hci = boot(base, f); u = used & rec
    rows.append(dict(variant=lab, wape=r(wape(f)), hit=r(hit(f), 3), bias=r(bias(f), 3), wape_all=r(wape(f, np.ones(len(D), bool))), cover=r(u.sum() / rec.sum(), 3), ci=ci, hit_ci=hci, wape_used=r(wape(f, u)), wape_used_base=r(wape(base, u)), wape_corr=r(wape(f * corr))))
bnd = X[np.arange(len(X)), np.abs(X - y[:, None]).argmin(1)]; rows.append(dict(variant="the item's best model, picked afterwards (bound)", wape=r(wape(bnd)), hit=r(hit(bnd), 3), bias=r(bias(bnd), 3), wape_all=r(wape(bnd, np.ones(len(D), bool))), cover=1.0))
print("A  picking each item's models from its own past (before the correction layer):"); print(pd.DataFrame(rows).drop(columns=["ci", "hit_ci"]).to_string(index=False)); [print("   ", d["variant"][:52], d.get("ci"), d.get("hit_ci")) for d in rows[1:-1]]
# persistence: does an item's best member stay its best?
bestnow = E.argmin(1); pers = []; rk = []
for o in REC:
    M = past_err("style_color", o, last_only=True)
    if M is None: continue
    te = np.where((D.origin == o).values)[0]; J_ = M.reindex(D.style_color.values[te]); ok = J_.n.notna().values; e = J_[MEM].values[ok]; pers.append((e.argmin(1) == bestnow[te[ok]]).astype(float)); rk.append(np.array([pd.Series(a).rank().corr(pd.Series(b).rank()) for a, b in zip(e, E[te[ok]])]))
    M2 = past_err("style_color", o); J2 = M2.reindex(D.style_color.values[te]); ok2 = J2.n.notna().values
pers = np.concatenate(pers); rk = np.concatenate(rk); nh = []
for o in REC:
    M = past_err("style_color", o); te = np.where((D.origin == o).values)[0]; n = M.reindex(D.style_color.values[te]).n.fillna(0).values; nh.append(n)
nh = np.concatenate(nh); same_side = float((((X > y[:, None]).all(1) | (X < y[:, None]).all(1)))[rec].mean()); best_share = pd.Series(bestnow[rec]).value_counts(normalize=True).reindex(range(5)).values
IP = dict(rows=rows, persist=r(pers.mean(), 3), chance=0.2, rank_corr=r(float(np.nanmean(rk)), 3), no_history=r(float((nh == 0).mean()), 3), med_history=r(float(np.median(nh[nh > 0])), 1), hist3=r(float((nh >= 3).mean()), 3), same_side=r(same_side, 3),
          best_share={m: r(v, 3) for m, v in zip(MEM, best_share)}, member_corr=r(float(np.corrcoef(np.log((X[rec] + 1) / (y[rec, None] + 1)).T)[np.triu_indices(5, 1)].mean()), 3))
print("persistence of the best member (chance 0.2):", IP["persist"], "| rank corr of member errors, latest closed origin vs now:", IP["rank_corr"], "| no closed history:", IP["no_history"], "| median closed origins:", IP["med_history"], "| mean pairwise corr of member log errors:", IP["member_corr"], "| best member shares:", IP["best_share"])

# ---------- B all five too low
allu = (X < y[:, None]).all(1); allo = (X > y[:, None]).all(1); R = D[rec]; au = allu[rec]; ae = np.abs(D.F.values - y)
V = pd.read_parquet("data/fk_err_voi_preds.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([D.origin, D.style_color])); fref, fplan, fall_ = V.f_ref.values, V.f_plan.values, V.f_all.values
m = rec & allu; top = X.max(1)
desc = lambda mm: dict(n=int(mm.sum()), share=r(mm.sum() / rec.sum(), 3), units=r(y[mm].sum() / y[rec].sum(), 3), err=r(ae[mm].sum() / ae[rec].sum(), 3), bias=r(D.F.values[mm].sum() / y[mm].sum() - 1, 3), young=r(float((D.age_m.values[mm] < 3).mean()), 3),
                       stores2=r(float((D.h_stores_chg.values[mm] >= 2).mean()), 3), partners1=r(float((D.h_partners_chg.values[mm] >= 1).mean()), 3), m1_up=r(float((D.h_m1.values[mm] > 1.2 * D.F.values[mm] / 4).mean()), 3),
                       disc_up=r(float((D.h_disc_chg.values[mm] > 0.03).mean()), 3), bf=r(float(D.bf_window.values[mm].mean()), 3), med_level=r(float(np.median(D.level.values[mm])), 0), in_range=r(float((y[mm] <= D.p90.values[mm]).mean()), 3),
                       hit=r(float((ae[mm] <= 0.2 * y[mm]).mean()), 3), med_over_top=r(float(np.median(y[mm] / np.maximum(top[mm], 1))), 3), med_over_F=r(float(np.median(y[mm] / np.maximum(D.F.values[mm], 1))), 3), over_top_20=r(float((y[mm] > 1.2 * top[mm]).mean()), 3))
AL = dict(all_low=desc(m), rest=desc(rec & ~allu), all_high=desc(rec & allo))
# what information removes (hindsight, cross-fitted, D276)
er = lambda f, mm: float(np.abs(f - y)[mm].sum())
AL["info"] = dict(ref_err_share=r(er(fref, m) / er(fref, rec), 3), plan_cut=r(1 - er(fplan, m) / er(fref, m), 3), all_cut=r(1 - er(fall_, m) / er(fref, m), 3), plan_bias=r(fplan[m].sum() / y[m].sum() - 1, 3), ref_bias=r(fref[m].sum() / y[m].sum() - 1, 3), all_bias=r(fall_[m].sum() / y[m].sum() - 1, 3),
                  hit_ref=r(hit(fref, m), 3), hit_plan=r(hit(fplan, m), 3), hit_all=r(hit(fall_, m), 3))
# can they be flagged at the origin? time-honest classifier
D["llv"] = np.log1p(D.level); D["lf"] = np.log((D.F + 1) / (D.R3 + 1)); D["macro_c"] = D.macro.astype("category").cat.codes; D["rng"] = np.log((D.p90 + 1) / (D.p10 + 1)); D["allu"] = allu.astype(int)
CF = ["llv", "age_m", "lf", "spread", "rng", "bf_window", "macro_c", "disc_web", "disc_mom", "size_run", "oos_28", "sh_tiktok", "sh_retail", "sh_partner", "n_stores", "stores_mom", "trend91", "top_day_share", "tiktok_mom", "n_siblings", "newest_colour", "has_ly", "hist_months"]
p = np.full(len(D), np.nan)
for o in REC:
    tr = D[D.oo + 4 <= pd.Period(o).ordinal]; te = D[D.origin == o]
    if len(tr) < 500: continue
    p[(D.origin == o).values] = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=6).fit(tr[CF], tr.allu).predict_proba(te[CF])[:, 1]
ok = rec & np.isfinite(p); D["p"] = p; D["dec"] = np.nan; D.loc[ok, "dec"] = D[ok].groupby("origin").p.transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
dec = [dict(decile=int(k) + 1, rate=r(float(allu[ok & (D.dec.values == k)].mean()), 3), err=r(ae[ok & (D.dec.values == k) & allu].sum() / ae[ok & allu].sum(), 3), units=r(y[ok & (D.dec.values == k)].sum() / y[ok].sum(), 3), bias=r(D.F.values[ok & (D.dec.values == k)].sum() / y[ok & (D.dec.values == k)].sum() - 1, 3)) for k in range(10)]
t2 = ok & (D.dec.values >= 8); AL["clf"] = dict(auc=r(roc_auc_score(allu[ok], p[ok]), 3), base=r(float(allu[ok].mean()), 3), deciles=dec, top20_rate=r(float(allu[t2].mean()), 3), top20_err=r(ae[t2 & allu].sum() / ae[ok & allu].sum(), 3), top20_bias=r(D.F.values[t2].sum() / y[t2].sum() - 1, 3),
                                         auc_by_origin=[dict(origin=o, auc=r(roc_auc_score(allu[ok & (D.origin == o).values], p[ok & (D.origin == o).values]), 3)) for o in REC if (ok & (D.origin == o).values).any()])
# a blind uplift of the flagged fifth: factor learnt on earlier origins' flagged fifth (their own time-honest flags), each origin's total not kept
fu = D.F.values.copy(); facs = []
for o in REC:
    pastm = ok & (D.oo + 4 <= pd.Period(o).ordinal).values & (D.dec.values >= 8); te = ok & (D.origin == o).values & (D.dec.values >= 8)
    if pastm.sum() < 150: continue
    k = y[pastm].sum() / D.F.values[pastm].sum(); facs.append(dict(origin=o, factor=r(k, 3))); fu[te] = D.F.values[te] * k
mu = rec & np.isin(D.origin.values, [f["origin"] for f in facs])
AL["uplift"] = dict(origins=[f["origin"] for f in facs], factors=facs, wape=r(wape(fu, mu)), wape_base=r(wape(D.F.values, mu)), hit=r(hit(fu, mu), 3), hit_base=r(hit(D.F.values, mu), 3), bias=r(bias(fu, mu), 3), bias_base=r(bias(D.F.values, mu), 3))
print("\nB  all five too low:"); [print("  ", k, v) for k, v in AL.items() if k != "clf"]; print("   clf", {k: v for k, v in AL["clf"].items() if k != "deciles"}); print(pd.DataFrame(dec).to_string(index=False))
J = json.load(open("data/fk_err_analysis.json")); J["itempick"] = IP; J["alllow"] = AL; json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
