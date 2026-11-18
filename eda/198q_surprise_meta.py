"""D289b, D289c (fork; user comments on the error study).
A  "let's add surprise metrics regarding model surprise from the months before that / like the surprise of the last forecasted month
   in the past"   At origin o the forecasts made at o-1 .. o-4 cover months that have since been sold. Surprise j = log of what the
   item sold in those j known months over the pace of the forecast made j origins earlier (its 4-month total x j / 4); the same
   against the highest of the five members; the same relative to the item's product type at that origin (so a seasonal swing shared
   by the type is not counted); and the closed miss of the forecast made 4 origins earlier. All known at the origin. Tested as inputs
   to the time-honest correction (as D282) and to the all-five-too-low flag (D286).
B  "maybe we can train a secondary model that picks models (tree based or bayesian) based on high level features like product type
   and perhaps we can run pca on the models and work with the independent models"
   Gate: a LightGBM classifier of which member will be closest, from high-level features (with and without the members' positions);
         pick its first choice / weight the members by its probabilities / average its top two.
   Bayes: member weights per product type from past errors, shrunk towards the overall weights by the number of past forecasts.
   PCA:  principal components of the five members' log forecasts (relative to the item's level), fitted on closed origins; the actual
         regressed on the first k components; a tree on the components and the high-level features.
   All trained on origins whose windows have closed; scored before the correction layer against the plain average.
Adds J["surprise"], J["metapick"] to data/fk_err_analysis.json; stdout -> eda/surprise_meta_output.txt."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb, warnings
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); ORIG = sorted(D.origin.unique()); REC = [o for o in ORIG if o >= "2025-06"]; D["rec"] = D.origin.isin(REC); D["oo"] = D.origin.map(lambda o: pd.Period(o, "M").ordinal)
J = json.load(open("data/fk_err_analysis.json")); MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; X = D[MEM].values.astype(float); y = D.y.values; F = D.F.values; ar = (D.R3 - D.y).abs().values; rec = D.rec.values; base = X.mean(1); rng = np.random.default_rng(0)
hit = lambda f, m=rec: float((np.abs(f - y)[m] <= 0.2 * y[m]).mean()); wape = lambda f, m=rec: float(np.abs(f - y)[m].sum() / ar[m].sum()); bias = lambda f, m=rec: float(f[m].sum() / y[m].sum() - 1)
def boot(fa, fb_, B=1000):
    Z = pd.DataFrame({"s": D.style_color, "a": np.abs(fa - y), "b": np.abs(fb_ - y), "r": ar, "ha": np.abs(fa - y) <= 0.2 * y, "hb": np.abs(fb_ - y) <= 0.2 * y, "n": 1.0})[rec].groupby("s").sum().values; n = len(Z); T = Z[rng.integers(0, n, (B, n))].sum(1)
    d = T[:, 1] / T[:, 2] - T[:, 0] / T[:, 2]; hd = (T[:, 4] - T[:, 3]) / T[:, 5]; return [r(np.percentile(d, 2.5)), r(np.percentile(d, 97.5))], [r(np.percentile(hd, 2.5), 3), r(np.percentile(hd, 97.5), 3)]

# ---------- A surprise of earlier forecasts
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; P = lambda o, k: str(pd.Period(o, "M") + k)
K = pd.MultiIndex.from_arrays([D.origin, D.style_color]); Fs = pd.Series(F, index=K); Ts = pd.Series(X.max(1), index=K); D["res"] = np.log((D.y + 1) / (D.F + 1)); Rs = pd.Series(D.res.values, index=K)
for j in (1, 2, 3, 4):
    prev = pd.MultiIndex.from_arrays([D.origin.map(lambda o: P(o, -j)), D.style_color]); fp = Fs.reindex(prev).values; tp = Ts.reindex(prev).values
    act = np.zeros(len(D))
    for o in ORIG:
        m = (D.origin == o).values; act[m] = M.reindex(D.style_color.values[m]).reindex(columns=[P(o, -k) for k in range(j)]).fillna(0).values.sum(1)
    D[f"s{j}"] = np.log((act + 1) / (fp * j / 4 + 1)); D[f"st{j}"] = np.log((act + 1) / (tp * j / 4 + 1)); D[f"s{j}_rel"] = D[f"s{j}"] - D.groupby(["origin", "ptype"])[f"s{j}"].transform("mean")
D["s_closed"] = Rs.reindex(pd.MultiIndex.from_arrays([D.origin.map(lambda o: P(o, -4)), D.style_color])).values; D["n_prev"] = D[["s1", "s2", "s3", "s4"]].notna().sum(1)
SUR = ["s1", "s2", "s3", "s4", "s1_rel", "s2_rel", "s3_rel", "st1", "st2", "s_closed", "n_prev"]
D["res_c"] = D.res - D.groupby("origin").res.transform("mean"); D["llv"] = np.log1p(D.level); D["lf"] = np.log((D.F + 1) / (D.R3 + 1)); D["macro_c"] = D.macro.astype("category").cat.codes; D["ptype_c"] = D.ptype.astype("category").cat.codes
allu = (X < y[:, None]).all(1); D["allu"] = allu.astype(int); D["rng"] = np.log((D.p90 + 1) / (D.p10 + 1))
LAB = {"s1": "latest month against the forecast made one origin earlier", "s2": "latest two months against the forecast made two origins earlier", "s3": "latest three months against the forecast made three origins earlier", "s_closed": "the closed miss of the forecast made four origins earlier",
       "s1_rel": "latest month's surprise, relative to the product type", "s2_rel": "two-month surprise, relative to the product type", "st1": "latest month against the highest model's pace", "trend91": "for comparison: the 91-day trend already in the models"}
sig = []
for c, lab in LAB.items():
    rr = D[D.rec].groupby("origin").apply(lambda g: spearmanr(g[c], g.res, nan_policy="omit")[0], include_groups=False); ra = D[D.rec].groupby("origin").apply(lambda g: spearmanr(g[c], g.allu, nan_policy="omit")[0], include_groups=False)
    sig.append(dict(key=c, label=lab, rho=r(rr.mean(), 3), same=int((np.sign(rr) == np.sign(rr.mean())).sum()), rho_allu=r(ra.mean(), 3), cover=r(float(D[D.rec][c].notna().mean()), 3)))
print("A  surprise features: rank correlation with the miss | origins with the same sign | with all-five-too-low | coverage"); [print(f"   {d['label']:68s} {d['rho']:+.3f} {d['same']}/9 {d['rho_allu']:+.3f} {d['cover']:.2f}") for d in sig]
BASEF = ["llv", "age_m", "lf", "spread", "bf_window", "macro_c", "disc_web", "size_run", "sh_tiktok", "sh_retail", "sh_partner", "n_stores", "trend91"]
PR = dict(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=6)
def th_corr(cols, shrink=0.5):
    f = F.copy()
    for o in REC:
        tr = D[D.oo + 4 <= pd.Period(o).ordinal]; te = D[D.origin == o]
        if len(tr) < 500: continue
        v = (te.F + 1) * np.exp(shrink * lgb.LGBMRegressor(**PR).fit(tr[cols], tr.res_c, sample_weight=np.log1p(tr.R3)).predict(te[cols])) - 1; f[(D.origin == o).values] = v * te.F.sum() / v.sum()
    return f
fb = th_corr(BASEF); rows = [dict(variant="submitted system", wape=r(wape(F)), hit=r(hit(F), 3)), dict(variant="correction on origin-known features (reference)", wape=r(wape(fb)), hit=r(hit(fb), 3))]
for lab, cols in [("+ surprise of the latest month (one origin earlier)", ["s1", "s1_rel", "st1"]), ("+ all surprise inputs (one to four origins earlier)", SUR), ("+ all surprise inputs, full strength", SUR)]:
    f = th_corr(BASEF + cols, shrink=1.0 if "full" in lab else 0.5); ci, hci = boot(fb, f); rows.append(dict(variant=lab, wape=r(wape(f)), hit=r(hit(f), 3), ci=ci, hit_ci=hci, wape_allu=r(wape(f, rec & allu)), wape_allu_ref=r(wape(fb, rec & allu))))
print(pd.DataFrame(rows).to_string(index=False))
CF = ["llv", "age_m", "lf", "spread", "rng", "bf_window", "macro_c", "disc_web", "disc_mom", "size_run", "oos_28", "sh_tiktok", "sh_retail", "sh_partner", "n_stores", "stores_mom", "trend91", "top_day_share", "tiktok_mom", "n_siblings", "newest_colour", "has_ly", "hist_months"]
def flag(cols):
    p = np.full(len(D), np.nan)
    for o in REC:
        tr = D[D.oo + 4 <= pd.Period(o).ordinal]; te = D[D.origin == o]
        if len(tr) < 500: continue
        p[(D.origin == o).values] = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=6).fit(tr[cols], tr.allu).predict_proba(te[cols])[:, 1]
    ok = rec & np.isfinite(p); dec = pd.Series(p[ok]).groupby(D.origin.values[ok]).transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)).values; ae = np.abs(F - y)[ok]; au = allu[ok]
    return dict(auc=r(roc_auc_score(au, p[ok]), 3), top20_rate=r(float(au[dec == 4].mean()), 3), top20_err=r(ae[(dec == 4) & au].sum() / ae[au].sum(), 3))
fl0, fl1 = flag(CF), flag(CF + SUR); print("all-five-too-low flag: without", fl0, "| with surprise", fl1)
J["surprise"] = dict(sig=sig, rows=rows, flag_base=fl0, flag_sur=fl1, n_inputs=len(SUR))

# ---------- B a second model that picks or re-mixes the five
L = np.log((X + 1) / (4 * D.level.values[:, None] + 1)); t = np.log((y + 1) / (4 * D.level.values + 1)); E = np.abs(np.log((X + 1) / (y[:, None] + 1))); best = E.argmin(1); dev = np.log((X + 1) / (base[:, None] + 1))
for i, m in enumerate(MEM): D["d_" + m] = dev[:, i]
HI = ["ptype_c", "macro_c", "age_m", "llv", "spread", "rng", "bf_window", "n_stores", "trend91", "has_ly", "hist_months"]; DV = ["d_" + m for m in MEM]; ones = np.ones(len(D), bool); OUT = {}
def gate(cols):
    Pm = np.full((len(D), 5), np.nan)
    for o in REC:
        trm = (D.oo + 4 <= pd.Period(o).ordinal).values; tem = (D.origin == o).values
        if trm.sum() < 500: continue
        c = lgb.LGBMClassifier(objective="multiclass", n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=6).fit(D.loc[trm, cols], best[trm])
        pr = c.predict_proba(D.loc[tem, cols]); Pm[np.ix_(np.where(tem)[0], c.classes_)] = pr
    return np.nan_to_num(Pm, nan=0.2)
for tag, cols in [("high-level features only", HI), ("high-level features and the models' positions", HI + DV)]:
    Pm = gate(cols); o2 = np.argsort(-Pm, 1); acc = float((Pm.argmax(1) == best)[rec].mean())
    OUT[f"gate, {tag}: its first choice"] = (X[np.arange(len(X)), Pm.argmax(1)], acc); OUT[f"gate, {tag}: weights = its probabilities"] = ((X * Pm).sum(1) / Pm.sum(1), acc); OUT[f"gate, {tag}: average of its top two"] = (np.take_along_axis(X, o2[:, :2], 1).mean(1), acc)
# empirical-Bayes weights per product type
fbay = base.copy()
for o in REC:
    trm = (D.oo + 4 <= pd.Period(o).ordinal).values; tem = np.where((D.origin == o).values)[0]
    if trm.sum() < 500: continue
    g = pd.DataFrame(E[trm], columns=MEM).assign(k=D.ptype.values[trm]).groupby("k"); me = g[MEM].mean(); n = g.size(); glob = E[trm].mean(0); k0 = 200.0
    sh = (me.mul(n, axis=0).values + k0 * glob) / (n.values[:, None] + k0); w = pd.DataFrame((1 / sh) ** 2, index=me.index); w = w.div(w.sum(1), axis=0); W = w.reindex(D.ptype.values[tem]).fillna(0.2).values; fbay[tem] = (X[tem] * W).sum(1) / W.sum(1)
OUT["Bayes: weights per product type, shrunk to the overall weights"] = (fbay, None)
# PCA of the members
pca_info = None
def pca_stack(k, tree=False, smear=True):
    global pca_info
    f = base.copy()
    for o in REC:
        trm = (D.oo + 4 <= pd.Period(o).ordinal).values; tem = (D.origin == o).values
        if trm.sum() < 500: continue
        mu = L[trm].mean(0); U, S, Vt = np.linalg.svd(L[trm] - mu, full_matrices=False); pc_tr = (L[trm] - mu) @ Vt.T; pc_te = (L[tem] - mu) @ Vt.T
        if o == REC[-1]: pca_info = dict(share=[r(v, 4) for v in (S ** 2 / (S ** 2).sum())], load1=[r(v, 3) for v in Vt[0]], load2=[r(v, 3) for v in Vt[1]])
        if tree:
            A = np.column_stack([pc_tr, D.loc[trm, HI].values]); B = np.column_stack([pc_te, D.loc[tem, HI].values]); pred = lgb.LGBMRegressor(**PR).fit(A, t[trm]).predict(B); fit_tr = None
        else:
            A = np.column_stack([np.ones(trm.sum()), pc_tr[:, :k]]); beta = np.linalg.lstsq(A, t[trm], rcond=None)[0]; pred = np.column_stack([np.ones(tem.sum()), pc_te[:, :k]]) @ beta; fit_tr = A @ beta
        v = (4 * D.level.values[tem] + 1) * np.exp(pred) - 1
        if smear and fit_tr is not None: v = v * y[trm].sum() / ((4 * D.level.values[trm] + 1) * np.exp(fit_tr) - 1).sum()   # restore the level lost by averaging in logs
        f[tem] = np.maximum(v, 0)
    return f
for k in (1, 2, 3, 5): OUT[f"PCA: the actual regressed on the first {k} component{'s' if k > 1 else ''}"] = (pca_stack(k), None)
OUT["PCA: a tree on the five components and the high-level features"] = (pca_stack(5, tree=True), None)
mrows = [dict(variant="plain average of the five (submitted stack)", wape=r(wape(base)), hit=r(hit(base), 3), bias=r(bias(base), 3), wape_all=r(wape(base, ones)))]
for lab, (f, acc) in OUT.items():
    ci, hci = boot(base, f); mrows.append(dict(variant=lab, wape=r(wape(f)), hit=r(hit(f), 3), bias=r(bias(f), 3), ci=ci, hit_ci=hci, acc=r(acc, 3) if acc is not None else None))
print("\nB  a second model that picks or re-mixes the five (before the correction layer):"); print(pd.DataFrame(mrows).to_string(index=False)); print("PCA at the last origin:", pca_info)
J["metapick"] = dict(rows=mrows, pca=pca_info, chance=0.2, majority=r(float(pd.Series(best[rec]).value_counts(normalize=True).max()), 3)); json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
