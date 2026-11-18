"""D304, D305 (fork; user in the session: "focus also on adding more recent features from the last month. also lets have an R^2 metric").
A  Last-month inputs, from the daily sales of the 28 days before the origin (and the 28 days before those):
     pace      last 7 and last 14 days against the 28-day pace; last 28 days against the previous 28; slope of the four weekly totals;
               last 28 days against the recency level
     rhythm    share of the 28 days with a sale; days since the last sale; days since the best day; best day over the average day;
               weekend share
     basket    units per order
     price     average discount of the last 28 days and its change against the previous 28
     channel   TikTok, own-store and partner share of the last 28 days and their change against the previous 28
     returns   cancelled share of the last 28 days
   Tested like the other inputs: rank correlation with the miss, the time-honest correction (as D282) and the all-too-low flag (D286).
B  R-squared of the forecasts (recent origins, and by origin): in units, in logs, and on the change against R3 (the part a forecast has
   to add to the recency baseline), pooled and within origin; for R3, R3 + rules, the five members and the submitted system.
Adds J["lastmonth"], J["r2"] to data/fk_err_analysis.json; stdout -> eda/last_month_r2_output.txt."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb, warnings
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); ORIG = sorted(D.origin.unique()); REC = [o for o in ORIG if o >= "2025-06"]; D["rec"] = D.origin.isin(REC); D["oo"] = D.origin.map(lambda o: pd.Period(o, "M").ordinal)
J = json.load(open("data/fk_err_analysis.json")); MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; X = D[MEM].values.astype(float); y = D.y.values; F = D.F.values; ar = (D.R3 - D.y).abs().values; rec = D.rec.values; rng = np.random.default_rng(0)
hit = lambda f, m=rec: float((np.abs(f - y)[m] <= 0.2 * y[m]).mean()); wape = lambda f, m=rec: float(np.abs(f - y)[m].sum() / ar[m].sum())

# ---------- A last-month inputs
dd = pd.read_parquet("data/daily_item_chan.parquet"); dd["d"] = pd.to_datetime(dd.ny_date); ch = dd.chan.astype(str).str.lower(); dd["k"] = np.where(dd["class"].astype(str).str.upper() == "RETAIL", "ret", np.where(ch.str.contains("tiktok"), "tt", np.where(ch == "edikted.com", "web", "par")))
print("channels:", dd.groupby("k").qty.sum().round(0).to_dict()); ok_ = dd[~dd.cancelled]; can = dd[dd.cancelled]
lr = lambda a, b: np.log((a + 1) / (b + 1)); parts = []
for o in ORIG:
    idx = np.where((D.origin == o).values)[0]; it = D.style_color.values[idx]; end = pd.Period(o, "M").end_time.normalize(); days = pd.date_range(end - pd.Timedelta(days=55), end)
    w = ok_[(ok_.d >= days[0]) & (ok_.d <= end) & ok_.style_color.isin(it)]; piv = lambda df, v: df.pivot_table(index="style_color", columns="d", values=v, aggfunc="sum").reindex(index=it, columns=days).fillna(0).values
    U = piv(w, "qty"); a, b = U[:, 28:], U[:, :28]; u28, p28 = a.sum(1), b.sum(1); f = pd.DataFrame(index=idx)
    f["lm_r7"] = lr(a[:, -7:].sum(1) * 4, u28); f["lm_r14"] = lr(a[:, -14:].sum(1) * 2, u28); f["lm_acc"] = lr(u28, p28); wk = np.log1p(np.stack([a[:, i * 7:(i + 1) * 7].sum(1) for i in range(4)], 1)); xw = np.arange(4) - 1.5; f["lm_slope"] = (wk * xw).sum(1) / (xw ** 2).sum()
    f["lm_vs_level"] = lr(u28 * 30.4 / 28, D.level.values[idx]); f["lm_days_sold"] = (a > 0).mean(1); last = np.where((a > 0).any(1), 27 - (a > 0)[:, ::-1].argmax(1), -1); f["lm_days_since"] = np.where(last >= 0, 27 - last, 28)
    f["lm_peak_ago"] = np.where(u28 > 0, 27 - a.argmax(1), np.nan); f["lm_peak_r"] = np.where(u28 > 0, a.max(1) / np.maximum(a.mean(1), 1e-9), np.nan); wd = np.array([d.dayofweek for d in days[28:]]); f["lm_wkend"] = np.where(u28 > 0, a[:, wd >= 5].sum(1) / np.maximum(u28, 1), np.nan)
    O = piv(w, "n_orders"); f["lm_basket"] = np.where(O[:, 28:].sum(1) > 0, u28 / np.maximum(O[:, 28:].sum(1), 1), np.nan); Dw = piv(w, "disc_w"); da = np.where(u28 > 0, Dw[:, 28:].sum(1) / np.maximum(u28, 1), np.nan); db = np.where(p28 > 0, Dw[:, :28].sum(1) / np.maximum(p28, 1), np.nan)
    f["lm_disc"] = da; f["lm_disc_chg"] = da - db
    for k in ("tt", "ret", "par"):
        Uk = piv(w[w.k == k], "qty"); sa = np.where(u28 > 0, Uk[:, 28:].sum(1) / np.maximum(u28, 1), np.nan); sb = np.where(p28 > 0, Uk[:, :28].sum(1) / np.maximum(p28, 1), np.nan); f[f"lm_sh_{k}"] = sa; f[f"lm_sh_{k}_chg"] = sa - sb
    C = piv(can[(can.d >= days[0]) & (can.d <= end) & can.style_color.isin(it)], "qty"); f["lm_cancel"] = C[:, 28:].sum(1) / np.maximum(C[:, 28:].sum(1) + u28, 1); parts.append(f)
Fe = pd.concat(parts).reindex(D.index); D = pd.concat([D, Fe], axis=1); LM = list(Fe.columns); print("last-month inputs:", len(LM), LM)
D["res"] = np.log((D.y + 1) / (D.F + 1)); D["res_c"] = D.res - D.groupby("origin").res.transform("mean"); D["llv"] = np.log1p(D.level); D["lf"] = np.log((D.F + 1) / (D.R3 + 1)); D["macro_c"] = D.macro.astype("category").cat.codes
allu = (X < y[:, None]).all(1); D["allu"] = allu.astype(int); D["rng"] = np.log((D.p90 + 1) / (D.p10 + 1)); D["abs_res"] = D.res.abs()
LAB = {"lm_r7": "last 7 days against the 28-day pace", "lm_r14": "last 14 days against the 28-day pace", "lm_acc": "last 28 days against the previous 28", "lm_slope": "slope of the last four weeks", "lm_vs_level": "last 28 days against the recency level",
       "lm_days_sold": "share of the last 28 days with a sale", "lm_days_since": "days since the last sale", "lm_peak_ago": "days since the best day", "lm_peak_r": "best day over the average day", "lm_wkend": "weekend share", "lm_basket": "units per order",
       "lm_disc": "average discount, last 28 days", "lm_disc_chg": "change in discount against the previous 28 days", "lm_sh_tt": "TikTok share, last 28 days", "lm_sh_tt_chg": "change in TikTok share", "lm_sh_ret": "own-store share, last 28 days",
       "lm_sh_ret_chg": "change in own-store share", "lm_sh_par": "partner share, last 28 days", "lm_sh_par_chg": "change in partner share", "lm_cancel": "cancelled share, last 28 days", "trend91": "for comparison: the 91-day trend already in the models", "f_r28": "for comparison: the 28-day rate already in the models"}
sig = []
for c, lab in LAB.items():
    g = D[D.rec].groupby("origin"); rr = g.apply(lambda z: spearmanr(z[c], z.res, nan_policy="omit")[0], include_groups=False); ra = g.apply(lambda z: spearmanr(z[c], z.abs_res, nan_policy="omit")[0], include_groups=False); ru = g.apply(lambda z: spearmanr(z[c], z.allu, nan_policy="omit")[0], include_groups=False)
    sig.append(dict(key=c, label=lab, rho=r(rr.mean(), 3), rho_size=r(ra.mean(), 3), rho_allu=r(ru.mean(), 3), same=int((np.sign(rr) == np.sign(rr.mean())).sum()), cover=r(float(D[D.rec][c].notna().mean()), 3)))
sig = sorted(sig, key=lambda d: -abs(d["rho"])); print("rank correlation with the miss (direction) | origins with that sign | with its size | with all-five-too-low"); [print(f"   {d['label']:56s} {d['rho']:+.3f} {d['same']}/9 {d['rho_size']:+.3f} {d['rho_allu']:+.3f}") for d in sig]
BASEF = ["llv", "age_m", "lf", "spread", "bf_window", "macro_c", "disc_web", "size_run", "sh_tiktok", "sh_retail", "sh_partner", "n_stores", "trend91"]
PR = dict(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=6)
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
fb = th_corr(BASEF); rows = [dict(variant="submitted system", wape=r(wape(F)), hit=r(hit(F), 3)), dict(variant="correction on origin-known features (reference)", wape=r(wape(fb)), hit=r(hit(fb), 3))]
GR = {"pace of the last month": ["lm_r7", "lm_r14", "lm_acc", "lm_slope", "lm_vs_level"], "rhythm, basket and returns": ["lm_days_sold", "lm_days_since", "lm_peak_ago", "lm_peak_r", "lm_wkend", "lm_basket", "lm_cancel"],
      "price and channel shifts in the last month": ["lm_disc", "lm_disc_chg", "lm_sh_tt", "lm_sh_tt_chg", "lm_sh_ret", "lm_sh_ret_chg", "lm_sh_par", "lm_sh_par_chg"], "all last-month inputs": LM}
for lab, cols in GR.items():
    f = th_corr(BASEF + cols); ci, hci = boot(fb, f); rows.append(dict(variant="+ " + lab, n=len(cols), wape=r(wape(f)), hit=r(hit(f), 3), ci=ci, hit_ci=hci))
f = th_corr(BASEF + LM, shrink=1.0); ci, hci = boot(fb, f); rows.append(dict(variant="+ all last-month inputs, full strength", n=len(LM), wape=r(wape(f)), hit=r(hit(f), 3), ci=ci, hit_ci=hci)); print(pd.DataFrame(rows).to_string(index=False))
CF = ["llv", "age_m", "lf", "spread", "rng", "bf_window", "macro_c", "disc_web", "disc_mom", "size_run", "oos_28", "sh_tiktok", "sh_retail", "sh_partner", "n_stores", "stores_mom", "trend91", "top_day_share", "tiktok_mom", "n_siblings", "newest_colour", "has_ly", "hist_months"]
def flag(cols):
    p = np.full(len(D), np.nan)
    for o in REC:
        tr = D[D.oo + 4 <= pd.Period(o).ordinal]; te = D[D.origin == o]
        if len(tr) < 500: continue
        p[(D.origin == o).values] = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=6).fit(tr[cols], tr.allu).predict_proba(te[cols])[:, 1]
    ok = rec & np.isfinite(p); q5 = pd.Series(p[ok]).groupby(D.origin.values[ok]).transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)).values; ae = np.abs(F - y)[ok]; au = allu[ok]
    return dict(auc=r(roc_auc_score(au, p[ok]), 3), top20_rate=r(float(au[q5 == 4].mean()), 3), top20_err=r(ae[(q5 == 4) & au].sum() / ae[au].sum(), 3))
fl0, fl1 = flag(CF), flag(CF + LM); print("all-five-too-low flag: without", fl0, "| with the last-month inputs", fl1)
J["lastmonth"] = dict(sig=sig, rows=rows, flag_base=fl0, flag_lm=fl1, n_inputs=len(LM))

# ---------- B R-squared
def r2s(f, m):
    yy, ff, r3 = y[m], f[m], D.R3.values[m]; ly, lf_ = np.log(yy + 1), np.log(ff + 1); g, pg = np.log((yy + 1) / (r3 + 1)), np.log((ff + 1) / (r3 + 1)); o = D.origin.values[m]
    dm = lambda v: v - pd.Series(v).groupby(o).transform("mean").values; e = g - pg
    return dict(units=r(1 - ((yy - ff) ** 2).sum() / ((yy - yy.mean()) ** 2).sum(), 3), log=r(1 - ((ly - lf_) ** 2).sum() / ((ly - ly.mean()) ** 2).sum(), 3), change=r(1 - (e ** 2).sum() / ((g - g.mean()) ** 2).sum(), 3),
                change_within=r(1 - (dm(e) ** 2).sum() / (dm(g) ** 2).sum(), 3), corr2_change=r(float(np.corrcoef(g, pg)[0, 1] ** 2) if pg.std() > 0 else 0.0, 3))
SYS = [("R3", D.R3.values), ("R3 + rules", D.rules.values), ("LightGBM member", D.qdyn.values), ("Neural net member", D.nn.values), ("TabPFN member (conformal)", D.pf2c.values), ("TabPFN member (32 estimators)", D.pf2d32.values), ("Plain average of the five", X.mean(1)), ("Submitted system", F)]
R2 = [dict(system=lab, **r2s(f, rec), all13=r2s(f, np.ones(len(D), bool))) for lab, f in SYS]; by_o = [dict(origin=o, **r2s(F, (D.origin == o).values), r3=r2s(D.R3.values, (D.origin == o).values)) for o in ORIG]
T = pd.DataFrame([{k: v for k, v in d.items() if k != "all13"} for d in R2]); print("\nB  R-squared, recent origins:"); print(T.to_string(index=False)); print(pd.DataFrame([{k: v for k, v in d.items() if k != "r3"} for d in by_o]).to_string(index=False))
J["r2"] = dict(systems=R2, by_origin=by_o); json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
