"""D289e-D289g (fork; user comments on the error study).
A  "what is the reason it scored so poorly on june"   The June 2025 origin (window July to October 2025) against the other origins:
   the system's and R3's error, how much of each is the level (the origin's total) and how much the split between items (both scaled
   to the true total), how often the system is closer than R3, how far the window moved away from the recent level, the history the
   models had, and where the June under-forecast sits.
B  "how do we identify the flag"   The all-five-too-low flag (D286): what it is trained on, which inputs carry it (gain importance and
   the rate in the top against the bottom fifth of each input), and the flag applied to the June 2026 origin: which of the 100
   targets fall in the highest-risk fifth.
C  "let's add the entropy of the specific item, and info on past peaks"   Entropy: how evenly the item's sales are spread over the
   last 91 days, over its last 12 months, and over channels (1 = even, 0 = one day / month / channel). Past peaks: months since the
   item's best month, best month of the last year over the level, months above 1.5x the item's median, whether last year's peak month
   falls in the forecast window, last year's same window over the level. Tested in the time-honest correction and in the flag.
   (Main tested a PEAK block inside the tree member in D253: 0.4332 against 0.4315, level.)
Adds J["june"], J["flaginfo"], J["entropy"] to data/fk_err_analysis.json; stdout -> eda/june_flag_entropy_output.txt."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb, warnings, openpyxl
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); ORIG = sorted(D.origin.unique()); REC = [o for o in ORIG if o >= "2025-06"]; D["rec"] = D.origin.isin(REC); D["oo"] = D.origin.map(lambda o: pd.Period(o, "M").ordinal)
J = json.load(open("data/fk_err_analysis.json")); MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; X = D[MEM].values.astype(float); y = D.y.values; F = D.F.values; ar = (D.R3 - D.y).abs().values; rec = D.rec.values; rng = np.random.default_rng(0)
hit = lambda f, m=rec: float((np.abs(f - y)[m] <= 0.2 * y[m]).mean()); wape = lambda f, m=rec: float(np.abs(f - y)[m].sum() / ar[m].sum())

# ---------- A the June 2025 origin
po = []
for o in ORIG:
    Z = D[D.origin == o]; fs = Z.F * Z.y.sum() / Z.F.sum(); rs = Z.R3 * Z.y.sum() / Z.R3.sum(); tot = Z.y.sum()
    po.append(dict(origin=o, sys_abs=r((Z.F - Z.y).abs().sum() / tot, 3), r3_abs=r((Z.R3 - Z.y).abs().sum() / tot, 3), ratio=r((Z.F - Z.y).abs().sum() / (Z.R3 - Z.y).abs().sum(), 3), sys_split=r((fs - Z.y).abs().sum() / tot, 3), r3_split=r((rs - Z.y).abs().sum() / tot, 3),
                   split_ratio=r((fs - Z.y).abs().sum() / (rs - Z.y).abs().sum(), 3), closer=r(float(((Z.F - Z.y).abs() < (Z.R3 - Z.y).abs()).mean()), 3), growth=r(tot / Z.R3.sum(), 3), sys_bias=r(Z.F.sum() / tot - 1, 3), r3_bias=r(Z.R3.sum() / tot - 1, 3),
                   has_ly=r(float(Z.has_ly.mean()), 3), hit=r(float(((Z.F - Z.y).abs() <= 0.2 * Z.y).mean()), 3), move=r(float(np.abs(np.log((Z.y + 1) / (Z.R3 + 1)) - np.log(tot / Z.R3.sum())).median()), 3)))
PO = pd.DataFrame(po); print("A  by origin:"); print(PO.to_string(index=False)); o6 = PO[PO.origin == "2025-06"].iloc[0]; oth = PO[PO.origin.isin(REC) & (PO.origin != "2025-06")]
Z6 = D[D.origin == "2025-06"]; coh6 = []
for lab, m in [("gained 2+ own stores in the window", Z6.h_stores_chg >= 2), ("launches under 3 months", Z6.age_m < 3), ("first month ran 20%+ above the forecast's pace", Z6.h_m1 > 1.2 * Z6.F / 4), ("all other forecasts", ~((Z6.h_stores_chg >= 2) | (Z6.age_m < 3) | (Z6.h_m1 > 1.2 * Z6.F / 4)))]:
    coh6.append(dict(g=lab, n=r(float(m.mean()), 3), units=r(Z6.y[m].sum() / Z6.y.sum(), 3), bias=r(Z6.F[m].sum() / Z6.y[m].sum() - 1, 3), err=r((Z6.F - Z6.y).abs()[m].sum() / (Z6.F - Z6.y).abs().sum(), 3)))
ty6 = Z6.assign(ae=(Z6.F - Z6.y).abs(), sgn=Z6.F - Z6.y).groupby("ptype").agg(ae=("ae", "sum"), F=("F", "sum"), y=("y", "sum")).sort_values("ae", ascending=False).head(6); ty6["bias"] = ty6.F / ty6.y - 1; ty6["err"] = ty6.ae / (Z6.F - Z6.y).abs().sum()
JU = dict(by_origin=po, june={k: (v if isinstance(v, str) else r(float(v), 3)) for k, v in o6.items()}, others={k: [r(float(oth[k].min()), 3), r(float(oth[k].max()), 3), r(float(oth[k].median()), 3)] for k in PO.columns if k != "origin"}, cohorts=coh6,
          types=[dict(ptype=k, err=r(v.err, 3), bias=r(v.bias, 3), units=r(v.y / Z6.y.sum(), 3)) for k, v in ty6.iterrows()], corr_ratio_r3=r(float(np.corrcoef(PO.ratio, PO.r3_abs)[0, 1]), 3), corr_ratio_split=r(float(np.corrcoef(PO.ratio, PO.r3_split)[0, 1]), 3),
          members=J.get("members", {}).get("season") if isinstance(J.get("members"), dict) else None)
print("June:", JU["june"]); print("other recent origins [min, max, median]:", JU["others"]); print(pd.DataFrame(coh6).to_string(index=False)); print(ty6[["err", "bias"]].round(3).to_string())

# ---------- C entropy and past peaks (built first: the flag in B is also re-run with them)
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; P = lambda o, k: str(pd.Period(o, "M") + k)
dd = pd.read_parquet("data/daily_item_chan.parquet", columns=["style_color", "ny_date", "cancelled", "qty"]); dd = dd[~dd.cancelled]; dd["d"] = pd.to_datetime(dd.ny_date); dd = dd.groupby(["style_color", "d"], as_index=False).qty.sum()
def nent(a):
    """normalised Shannon entropy of the positive part of each row: 1 = even over the live span, 0 = one cell"""
    a = np.maximum(np.asarray(a, float), 0); s = a.sum(1, keepdims=True); p = np.divide(a, s, out=np.zeros_like(a), where=s > 0); h = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(1)
    first = np.where((a > 0).any(1), (a > 0).argmax(1), a.shape[1]); span = a.shape[1] - first; return np.where((s[:, 0] > 0) & (span > 1), h / np.log(np.maximum(span, 2)), np.nan)
parts = []
for o in ORIG:
    it = D.style_color.values[(D.origin == o).values]; end = pd.Period(o, "M").end_time.normalize(); w = dd[(dd.d > end - pd.Timedelta(days=91)) & (dd.d <= end) & dd.style_color.isin(it)]
    dm = w.pivot_table(index="style_color", columns="d", values="qty", aggfunc="sum").reindex(index=it, columns=pd.date_range(end - pd.Timedelta(days=90), end)).fillna(0).values
    mm = M.reindex(it).reindex(columns=[P(o, -k) for k in range(11, -1, -1)]).fillna(0).values; life = M.reindex(it).reindex(columns=[c for c in M.columns if c <= o]).fillna(0); lv = life.values
    f = pd.DataFrame(index=np.where((D.origin == o).values)[0]); f["ent_d91"] = nent(dm); f["ent_m12"] = nent(mm); f["top7_share"] = np.sort(dm, 1)[:, -7:].sum(1) / np.maximum(dm.sum(1), 1)
    pk = lv.argmax(1); f["m_since_peak"] = lv.shape[1] - 1 - pk; lvl = D.level.values[f.index]; f["peak12_r"] = np.log((mm.max(1) + 1) / (lvl + 1)); med = np.median(np.where(mm > 0, mm, np.nan), 1) if mm.size else np.nan
    f["n_spikes"] = (mm > 1.5 * np.nan_to_num(np.nanmedian(np.where(mm > 0, mm, np.nan), axis=1), nan=np.inf)[:, None]).sum(1); pkm = np.array([int(str(life.columns[i])[5:]) for i in pk]); win = [(pd.Period(o, "M") + k).month for k in (1, 2, 3, 4)]
    f["peak_in_win"] = (np.isin(pkm, win) & (f.m_since_peak.values >= 6)).astype(float); ly = M.reindex(it).reindex(columns=[P(o, k - 12) for k in (1, 2, 3, 4)]).fillna(0).values.sum(1); f["ly_win_r"] = np.where(ly > 0, np.log((ly + 1) / (4 * lvl + 1)), np.nan); parts.append(f)
Fe = pd.concat(parts).reindex(D.index); D = pd.concat([D, Fe], axis=1); sh = D[["sh_web", "sh_tiktok", "sh_retail", "sh_partner"]].clip(lower=0).fillna(0).values; D["ent_chan"] = nent(sh) * 0 + (-(np.where(sh > 0, sh * np.log(np.where(sh > 0, sh, 1)), 0)).sum(1) / np.log(4))
ENT = ["ent_d91", "ent_m12", "ent_chan", "top7_share"]; PEAK = ["m_since_peak", "peak12_r", "n_spikes", "peak_in_win", "ly_win_r", "peak_r"]
D["res"] = np.log((D.y + 1) / (D.F + 1)); D["res_c"] = D.res - D.groupby("origin").res.transform("mean"); D["llv"] = np.log1p(D.level); D["lf"] = np.log((D.F + 1) / (D.R3 + 1)); D["macro_c"] = D.macro.astype("category").cat.codes
allu = (X < y[:, None]).all(1); D["allu"] = allu.astype(int); D["rng"] = np.log((D.p90 + 1) / (D.p10 + 1)); D["abs_res"] = D.res.abs()
LAB = {"ent_d91": "evenness of sales over the last 91 days", "ent_m12": "evenness of sales over the last 12 months", "ent_chan": "evenness over channels", "top7_share": "share of the 91-day units sold on the best 7 days",
       "m_since_peak": "months since the item's best month", "peak12_r": "best month of the last year over the level", "n_spikes": "months above 1.5× the item's median", "peak_in_win": "last year's peak month falls in the forecast window", "ly_win_r": "last year's same window over the level"}
sig = []
for c, lab in LAB.items():
    g = D[D.rec].groupby("origin"); rr = g.apply(lambda z: spearmanr(z[c], z.res, nan_policy="omit")[0], include_groups=False); ra = g.apply(lambda z: spearmanr(z[c], z.abs_res, nan_policy="omit")[0], include_groups=False); ru = g.apply(lambda z: spearmanr(z[c], z.allu, nan_policy="omit")[0], include_groups=False)
    sig.append(dict(key=c, label=lab, rho=r(rr.mean(), 3), rho_size=r(ra.mean(), 3), rho_allu=r(ru.mean(), 3), same=int((np.sign(rr) == np.sign(rr.mean())).sum()), cover=r(float(D[D.rec][c].notna().mean()), 3)))
print("\nC  entropy and past peaks: rank correlation with the miss (direction) | with its size | with all-five-too-low | coverage"); [print(f"   {d['label']:56s} {d['rho']:+.3f} {d['rho_size']:+.3f} {d['rho_allu']:+.3f} {d['cover']:.2f}") for d in sig]
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
for lab, cols in [("+ entropy inputs", ENT), ("+ past-peak inputs", PEAK), ("+ entropy and past peaks", ENT + PEAK)]:
    f = th_corr(BASEF + cols); ci, hci = boot(fb, f); rows.append(dict(variant=lab, wape=r(wape(f)), hit=r(hit(f), 3), ci=ci, hit_ci=hci))
print(pd.DataFrame(rows).to_string(index=False))

# ---------- B the flag
CF = ["llv", "age_m", "lf", "spread", "rng", "bf_window", "macro_c", "disc_web", "disc_mom", "size_run", "oos_28", "sh_tiktok", "sh_retail", "sh_partner", "n_stores", "stores_mom", "trend91", "top_day_share", "tiktok_mom", "n_siblings", "newest_colour", "has_ly", "hist_months"]
CN = {"llv": "sales level", "age_m": "age in months", "lf": "forecast against R3", "spread": "disagreement between the five models", "rng": "width of the stated range", "bf_window": "Black Friday in the window", "macro_c": "product group", "disc_web": "web discount",
      "disc_mom": "change in discount", "size_run": "share of sizes still selling", "oos_28": "days out of stock", "sh_tiktok": "TikTok share", "sh_retail": "own-store share", "sh_partner": "partner share", "n_stores": "own stores selling it", "stores_mom": "stores added in 3 months",
      "trend91": "91-day trend", "top_day_share": "best day's share of sales", "tiktok_mom": "change in TikTok share", "n_siblings": "live colours in the style", "newest_colour": "newest colour of the style", "has_ly": "sold a year ago", "hist_months": "months of history"}
CP = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=6)
def flag(cols):
    p = np.full(len(D), np.nan)
    for o in REC:
        tr = D[D.oo + 4 <= pd.Period(o).ordinal]; te = D[D.origin == o]
        if len(tr) < 500: continue
        p[(D.origin == o).values] = lgb.LGBMClassifier(**CP).fit(tr[cols], tr.allu).predict_proba(te[cols])[:, 1]
    ok = rec & np.isfinite(p); q5 = pd.Series(p[ok]).groupby(D.origin.values[ok]).transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)).values; ae = np.abs(F - y)[ok]; au = allu[ok]
    return dict(auc=r(roc_auc_score(au, p[ok]), 3), top20_rate=r(float(au[q5 == 4].mean()), 3), low20_rate=r(float(au[q5 == 0].mean()), 3), top20_err=r(ae[(q5 == 4) & au].sum() / ae[au].sum(), 3))
f_base = flag(CF); f_ent = flag(CF + ENT + PEAK); CF2 = [c for c in CF if c != "rng"]; f_norng = flag(CF2); print("\nB  flag: base", f_base, "| with entropy and peaks", f_ent, "| without the range (as usable for 2026)", f_norng)
full = lgb.LGBMClassifier(**CP).fit(D[CF], D.allu); gain = pd.Series(full.booster_.feature_importance("gain"), index=CF); gain = gain / gain.sum(); imp = []
for c in gain.sort_values(ascending=False).index[:10]:
    Z = D[D.rec]; q = Z.groupby("origin")[c].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)) if Z[c].nunique() > 5 else None
    lo, hi = (float(Z.allu[q == 0].mean()), float(Z.allu[q == 4].mean())) if q is not None else (float(Z.allu[Z[c] == Z[c].min()].mean()), float(Z.allu[Z[c] == Z[c].max()].mean()))
    imp.append(dict(key=c, label=CN[c], gain=r(gain[c], 3), low=r(lo, 3), high=r(hi, 3)))
print(pd.DataFrame(imp).to_string(index=False))
# the flag at the June 2026 origin: trained on all 13 origins (all closed by June 2026), without the range (not stored for 2026)
T = pd.read_parquet("data/fk_err_master_2026.parquet").reset_index(drop=True); T["llv"] = np.log1p(T.level); T["lf"] = np.log((T.F + 1) / (T.R3 + 1)); cats = D.macro.astype("category").cat.categories; T["macro_c"] = pd.Categorical(T.macro, categories=cats).codes
m26 = lgb.LGBMClassifier(**CP).fit(D[CF2], D.allu); T["p"] = m26.predict_proba(T[CF2])[:, 1]; T["q5"] = pd.qcut(T.p.rank(method="first"), 5, labels=False); ws = openpyxl.load_workbook("Edikted_Forecasting_Exercise_filled.xlsx", data_only=True)["Forecast"]; NM = {row[1]: row[2] for row in ws.iter_rows(min_row=2, values_only=True) if row[1]}
TG = T[T.style_color.isin(list(NM))].copy()   # the workbook's targets (the table's own target mark misses the 15 youngest); S16079_DARK-GRAY is ruled and has no model row
top = TG.sort_values("p", ascending=False).head(12); FI = dict(base=f_base, with_ent=f_ent, no_range=f_norng, n_inputs=len(CF), importance=imp, base_rate=r(float(D.allu[D.rec].mean()), 3),
          t26=dict(n=int(len(TG)), flagged=int((TG.q5 == 4).sum()), flagged_units=r(TG.F[TG.q5 == 4].sum() / TG.F.sum(), 3), mean_p=r(float(TG.p.mean()), 3), mean_p_all=r(float(T.p.mean()), 3), low=int((TG.q5 == 0).sum()), flagged_young=int(((TG.q5 == 4) & (TG.age_m < 6)).sum()), young=int((TG.age_m < 6).sum()), flagged_items=sorted(TG.style_color[TG.q5 == 4].tolist()),
                   top=[dict(item=x.style_color, name=NM.get(x.style_color, ""), ptype=x.ptype, F=r(x.F, 0), p=r(x.p, 3), age=r(x.age_m, 1), spread=r(x.spread, 2), trend=r(x.trend91, 2)) for x in top.itertuples()]))
print("2026 targets:", {k: v for k, v in FI["t26"].items() if k != "top"}); print(pd.DataFrame(FI["t26"]["top"]).to_string(index=False))
J["june"] = JU; J["flaginfo"] = FI; J["entropy"] = dict(sig=sig, rows=rows, flag_base=f_base, flag_ent=f_ent, n_ent=len(ENT), n_peak=len(PEAK)); json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
