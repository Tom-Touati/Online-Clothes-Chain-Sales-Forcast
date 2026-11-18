"""D312 (fork; user comment on the error study, on the tag 'one day carried 15%+ of a month (spike)': "add feature that states the max sales
day in recent history"). From the daily sales up to the origin (cancelled lines excluded):
  MAXDAY  the best day over the last 91 and the last 365 days and over the item's life, each against the average day at the recency level;
          days since each best day; the best day's share of the 365-day units; the best day in units (log); whether the 365-day best day
          fell in the last 28 days
Tests (as D307 / D311): rank correlation with the submitted system's miss; the time-honest correction (as D282); Main's tree member 'ec_ns'
refitted with the block, alone and inside the five-member average, also on top of the D307 origin-known signals.
Writes data/fk_maxday_feats.parquet, data/fk_maxday_member_preds.parquet; adds J["maxday"]; stdout -> eda/max_day_output.txt."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb, warnings
from scipy.stats import spearmanr
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
argv0 = sys.argv; sys.argv = ["eda/180q_models.py"]; ns = {}; exec(compile(open("eda/180q_models.py", encoding="utf-8").read(), "eda/180q_models.py", "exec"), ns); sys.argv = argv0
ns["P"]["n_jobs"] = 3   # user: "you can use the laptop but dont let it lag the computer"
T = ns["T"].reset_index(drop=True); sets = ns["sets"]; fit = ns["fit"]; ORIG = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
dd = pd.read_parquet("data/daily_item_chan.parquet", columns=["style_color", "ny_date", "cancelled", "qty"]); dd = dd[~dd.cancelled]; dd["d"] = pd.to_datetime(dd.ny_date); dd = dd.groupby(["style_color", "d"], as_index=False).qty.sum()
d0 = dd.d.min(); parts = []
for o in sorted(T.origin.unique()):
    idx = np.where((T.origin == o).values)[0]; s = T.style_color.values[idx]; end = pd.Period(o, "M").end_time.normalize(); days = pd.date_range(d0, end)
    w = dd[(dd.d <= end) & dd.style_color.isin(s)]; U = w.pivot_table(index="style_color", columns="d", values="qty", aggfunc="sum").reindex(index=s, columns=days).fillna(0).values
    avg = np.maximum(T.level.values[idx], 0) / 30.4; f = pd.DataFrame(index=idx); n = U.shape[1]
    for nm, k in [("91", 91), ("365", 365), ("life", n)]:
        k = min(k, n); W = U[:, -k:]; mx = W.max(1); am = W.argmax(1)
        f[f"md_{nm}_r"] = np.where(mx > 0, np.log((mx + 1) / (avg + 1)), np.nan); f[f"md_{nm}_since"] = np.where(mx > 0, k - 1 - am, np.nan)
    W = U[:, -min(365, n):]; tot = W.sum(1); f["md_365_share"] = np.where(tot > 0, W.max(1) / np.maximum(tot, 1), np.nan); f["md_365_units"] = np.log1p(W.max(1)); f["md_365_recent"] = (f.md_365_since.values <= 27).astype(float)
    parts.append(f)
FE = pd.concat(parts).reindex(T.index); MD = list(FE.columns); T = pd.concat([T, FE], axis=1); T[["origin", "style_color"] + MD].to_parquet("data/fk_maxday_feats.parquet"); print("MAXDAY", len(MD), MD, "| coverage", round(float(T[MD].notna().mean().mean()), 3), flush=True)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); K = pd.MultiIndex.from_arrays([D.origin, D.style_color]); D = pd.concat([D, T.set_index(["origin", "style_color"])[MD].reindex(K).reset_index(drop=True)], axis=1)
REC = [o for o in ORIG if o >= "2025-06"]; rec = D.origin.isin(REC).values; D["oo"] = D.origin.map(lambda o: pd.Period(o, "M").ordinal); y = D.y.values; F = D.F.values; ar = (D.R3 - D.y).abs().values; rng = np.random.default_rng(0)
D["res"] = np.log((D.y + 1) / (D.F + 1)); D["res_c"] = D.res - D.groupby("origin").res.transform("mean"); D["llv"] = np.log1p(D.level); D["lf"] = np.log((D.F + 1) / (D.R3 + 1)); D["macro_c"] = D.macro.astype("category").cat.codes; D["abs_res"] = D.res.abs()
hit = lambda f, m=rec: float((np.abs(f - y)[m] <= 0.2 * y[m]).mean()); wape = lambda f, m=rec: float(np.abs(f - y)[m].sum() / ar[m].sum())
LAB = {"md_91_r": "best day of the last 91 days over the average day", "md_365_r": "best day of the last year over the average day", "md_life_r": "best day ever over the average day", "md_91_since": "days since the best day of the last 91",
       "md_365_since": "days since the best day of the last year", "md_365_share": "best day's share of the year's units", "md_365_units": "best day of the last year, units (log)", "md_365_recent": "the year's best day fell in the last 28 days"}
sig = []
for c, lab in LAB.items():
    g = D[rec].groupby("origin"); rr = g.apply(lambda z: spearmanr(z[c], z.res, nan_policy="omit")[0], include_groups=False); ra = g.apply(lambda z: spearmanr(z[c], z.abs_res, nan_policy="omit")[0], include_groups=False)
    sig.append(dict(key=c, label=lab, rho=r(rr.mean(), 3), rho_size=r(ra.mean(), 3), same=int((np.sign(rr) == np.sign(rr.mean())).sum())))
sig = sorted(sig, key=lambda d: -abs(d["rho"])); print("rank correlation with the miss (direction | size | origins with that sign)"); [print(f"   {d['label']:52s} {d['rho']:+.3f} {d['rho_size']:+.3f} {d['same']}/9") for d in sig]
BASEF = ["llv", "age_m", "lf", "spread", "bf_window", "macro_c", "disc_web", "size_run", "sh_tiktok", "sh_retail", "sh_partner", "n_stores", "trend91"]
PR = dict(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=3)
def th_corr(cols, shrink=0.5):
    f = F.copy()
    for o in REC:
        tr = D[D.oo + 4 <= pd.Period(o).ordinal]; te = D[D.origin == o]
        if len(tr) < 500: continue
        v = (te.F + 1) * np.exp(shrink * lgb.LGBMRegressor(**PR).fit(tr[cols], tr.res_c, sample_weight=np.log1p(tr.R3)).predict(te[cols])) - 1; f[(D.origin == o).values] = v * te.F.sum() / v.sum()
    return f
def boot(fa, fb_, B=1000):
    Zb = pd.DataFrame({"s": D.style_color, "a": np.abs(fa - y), "b": np.abs(fb_ - y), "r": ar, "ha": np.abs(fa - y) <= 0.2 * y, "hb": np.abs(fb_ - y) <= 0.2 * y, "n": 1.0})[rec].groupby("s").sum().values; n = len(Zb); T_ = Zb[rng.integers(0, n, (B, n))].sum(1)
    d = T_[:, 1] / T_[:, 2] - T_[:, 0] / T_[:, 2]; hd = (T_[:, 4] - T_[:, 3]) / T_[:, 5]; return [r(np.percentile(d, 2.5)), r(np.percentile(d, 97.5))], [r(np.percentile(hd, 2.5), 3), r(np.percentile(hd, 97.5), 3)]
fb = th_corr(BASEF); f1 = th_corr(BASEF + MD); ci, hci = boot(fb, f1); crow = [dict(variant="correction on origin-known features (reference)", wape=r(wape(fb)), hit=r(hit(fb), 3)), dict(variant="+ best-day inputs", n=len(MD), wape=r(wape(f1)), hit=r(hit(f1), 3), ci=ci, hit_ci=hci)]
print(pd.DataFrame(crow).to_string(index=False), flush=True)
src = open("eda/198d_error_master.py", encoding="utf-8").read(); src = src[src.index("import sys, glob"):src.index("\ndef build(")]; ax = {}; exec(compile(src, "eda/198d_error_master.py", "exec"), ax)
AX = pd.concat([ax["axes"](g, o, hindsight=False) for o, g in T.groupby("origin")]).reindex(T.index).add_prefix("ax_"); SIG = ["ax_" + c for c in ["oos_28", "newest_colour", "size_run", "size_hhi", "sh_partner", "n_partners", "share_of_style", "n_siblings", "disc_web", "disc_vs_site", "top_day_share"]]; T = pd.concat([T, AX[SIG]], axis=1)
B0, F0_ = sets("ec_ns"); PRED = {}
for lab, add in [("+ best-day inputs", MD), ("+ origin-known signals (D307) and best-day inputs", SIG + MD)]:
    t0 = time.time(); out = []
    for o in ORIG:
        oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()]; te = T[T.origin == o]; assert (tr.o_ord + 4 <= oo).all()
        z = (fit(tr, te, B0 + add) + fit(tr, te, F0_ + add)) / 2; out.append(pd.DataFrame({"origin": o, "style_color": te.style_color.values, "q50": np.clip((4 * te.level.values + 1) * np.exp(z) - 1, 0, None)}))
    PRED[lab] = pd.concat(out).set_index(["origin", "style_color"]).q50; print(lab, "|", round(time.time() - t0), "s", flush=True)
S = pd.read_parquet("data/fk_member_refit_preds.parquet").set_index(["origin", "style_color"]).reindex(K); MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; X = D[MEM].values.astype(float); corr = (D.F / D.F0).values; allm = np.ones(len(D), bool)
m0 = S["ec_ns (refit, check)"].values; s0 = (X.sum(1) - D.qdyn.values + m0) / 5; mS = S["+ origin-known signals"].values; sS = (X.sum(1) - D.qdyn.values + mS) / 5
mrow = [dict(variant="tree member as submitted", alone=r(wape(m0)), stack=r(wape(s0)), corrected=r(wape(s0 * corr)), corrected_hit=r(hit(s0 * corr), 3), corrected_all=r(wape(s0 * corr, allm))),
        dict(variant="+ origin-known signals (D307)", alone=r(wape(mS)), stack=r(wape(sS)), corrected=r(wape(sS * corr)), corrected_hit=r(hit(sS * corr), 3), corrected_all=r(wape(sS * corr, allm)))]; sv = D[["origin", "style_color"]].copy()
for lab in PRED:
    m = PRED[lab].reindex(K).values; assert np.isfinite(m).all(); st = (X.sum(1) - D.qdyn.values + m) / 5; sv[lab] = m; ref = sS if "D307" in lab else s0; ci, _ = boot(ref, st); po = [wape(st, (D.origin == o).values) - wape(ref, (D.origin == o).values) for o in sorted(D.origin.unique())]
    mrow.append(dict(variant=lab, alone=r(wape(m)), stack=r(wape(st)), stack_ci=ci, corrected=r(wape(st * corr)), corrected_hit=r(hit(st * corr), 3), corrected_all=r(wape(st * corr, allm)), against="D307" if "D307" in lab else "submitted member", wins13=int(sum(v < 0 for v in po))))
print(pd.DataFrame(mrow).to_string(index=False)); sv.to_parquet("data/fk_maxday_member_preds.parquet")
J = json.load(open("data/fk_err_analysis.json")); J["maxday"] = dict(sig=sig, corr=crow, member=mrow, n=len(MD)); json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
