"""D310 (fork; user comments on the error study: "use that as known future" on the tag 'web discount deepened 3+ pts vs the site', and
"anything not related to sales but is a result of new partnerships or other business decisions can be added as a known future").
Each business decision of the forecast window as a known-future input to the time-honest correction of the submitted system, the same
test as D282 / D283 (earlier origins only, whose windows have closed; origin total kept; reference on origin-known features):
  discount plan   the item's web discount in the window against the three months before, and the same against the site's
  partners        partners added in the window
  launches        new colours of the style and same-type launches in the window
  stores          own stores added and reached (D283, repeated here for one table)
  all decisions   the four together
Also where the discount plan sits: the miss by how the item's discount moved against the site's.
Adds J["future_blocks"] to data/fk_err_analysis.json; stdout -> eda/future_blocks_corr_output.txt."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); ORIG = sorted(D.origin.unique()); REC = [o for o in ORIG if o >= "2025-06"]; D["rec"] = D.origin.isin(REC); D["oo"] = D.origin.map(lambda o: pd.Period(o, "M").ordinal)
y = D.y.values; F = D.F.values; ar = (D.R3 - D.y).abs().values; rec = D.rec.values; rng = np.random.default_rng(0)
D["res"] = np.log((D.y + 1) / (D.F + 1)); D["res_c"] = D.res - D.groupby("origin").res.transform("mean"); D["llv"] = np.log1p(D.level); D["lf"] = np.log((D.F + 1) / (D.R3 + 1)); D["macro_c"] = D.macro.astype("category").cat.codes; D["stores_end"] = D.n_stores + D.h_stores_chg
hit = lambda f, m=rec: float((np.abs(f - y)[m] <= 0.2 * y[m]).mean()); wape = lambda f, m=rec: float(np.abs(f - y)[m].sum() / ar[m].sum()); bias = lambda f, m=rec: float(f[m].sum() / y[m].sum() - 1)
BASEF = ["llv", "age_m", "lf", "spread", "bf_window", "macro_c", "disc_web", "size_run", "sh_tiktok", "sh_retail", "sh_partner", "n_stores", "trend91"]
PR = dict(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=3)   # user: do not lag the computer
def th_corr(cols, shrink=0.5):
    f = F.copy()
    for o in REC:
        tr = D[D.oo + 4 <= pd.Period(o).ordinal]; te = D[D.origin == o]
        if len(tr) < 500: continue
        v = (te.F + 1) * np.exp(shrink * lgb.LGBMRegressor(**PR).fit(tr[cols], tr.res_c, sample_weight=np.log1p(tr.R3)).predict(te[cols])) - 1; f[(D.origin == o).values] = v * te.F.sum() / v.sum()
    return f
def boot(fa, fb_, B=1000):
    Z = pd.DataFrame({"s": D.style_color, "a": np.abs(fa - y), "b": np.abs(fb_ - y), "r": ar, "ha": np.abs(fa - y) <= 0.2 * y, "hb": np.abs(fb_ - y) <= 0.2 * y, "n": 1.0})[rec].groupby("s").sum().values; n = len(Z); T_ = Z[rng.integers(0, n, (B, n))].sum(1)
    d = T_[:, 1] / T_[:, 2] - T_[:, 0] / T_[:, 2]; hd = (T_[:, 4] - T_[:, 3]) / T_[:, 5]; return [r(np.percentile(d, 2.5)), r(np.percentile(d, 97.5))], [r(np.percentile(hd, 2.5), 3), r(np.percentile(hd, 97.5), 3)]
BL = {"the discount plan": ["h_disc_chg", "h_disc_chg_rel"], "partners added": ["h_partners_chg"], "new colours of the style and same-type launches": ["h_new_style", "h_new_type"], "own stores added and reached": ["h_stores_chg", "stores_end"]}
BL["every business decision together"] = sum(BL.values(), [])
fb = th_corr(BASEF); rows = [dict(variant="submitted system", wape=r(wape(F)), hit=r(hit(F), 3)), dict(variant="correction on origin-known features (reference)", wape=r(wape(fb)), hit=r(hit(fb), 3))]
for lab, cols in BL.items():
    f = th_corr(BASEF + cols); ci, hci = boot(fb, f); rows.append(dict(variant="+ " + lab, n=len(cols), wape=r(wape(f)), hit=r(hit(f), 3), ci=ci, hit_ci=hci))
print(pd.DataFrame(rows).to_string(index=False))
# the miss by how the item's discount moved against the site's in the window
R = D[D.rec]; bins = [(-9, -0.03, "discount fell 3+ points against the site"), (-0.03, 0.03, "within 3 points of the site"), (0.03, 9, "discount deepened 3+ points against the site")]; dz = []
for lo, hi, lab in bins:
    m = (R.h_disc_chg_rel > lo) & (R.h_disc_chg_rel <= hi); dz.append(dict(g=lab, n=r(float(m.mean()), 3), units=r(R.y[m].sum() / R.y.sum(), 3), bias=r(R.F[m].sum() / R.y[m].sum() - 1, 3), err=r((R.F - R.y).abs()[m].sum() / (R.F - R.y).abs().sum(), 3), hit=r(float(((R.F - R.y).abs()[m] <= 0.2 * R.y[m]).mean()), 3)))
print(pd.DataFrame(dz).to_string(index=False)); J = json.load(open("data/fk_err_analysis.json")); J["future_blocks"] = dict(rows=rows, disc=dz, cover_disc=r(float(R.h_disc_chg_rel.notna().mean()), 3)); json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
