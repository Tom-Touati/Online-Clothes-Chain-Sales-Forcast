"""D279 (fork; user comment on the error study: "we have multiple launch curves, for each store the item arrives to first. we can also
use data from other stores as features"). Per-location launch curves and what they add, all known at the origin.
For every item x location (each own store, each wholesale partner, edikted.com, TikTok): arrival = first sale there; launch index =
log of its first-14-day units over the median first-14-day units of the items that arrived at the same location in the 180 days
before (so a strong store and a weak store are comparable). At each origin, from arrivals and sales up to the origin only:
  st_n, st_share, st_gap   own stores reached, as a share of the stores trading that month, and stores not yet reached
  st_new28                 own stores first reached in the last 28 days (roll-out in progress)
  st_idx_mean / first / last / sd   launch index over its stores, at the first store it reached, at the latest, and their spread
  st_rate                  units per reached store per day in the last 28 days, against the fleet's median for that month (log)
  pa_n, pa_gap, pa_idx     the same for wholesale partners
  web_idx, tt_idx          launch index on edikted.com and on TikTok
  lead_web                 days from the first web sale to the first own-store sale (negative = stores first)
  potential                st_gap x st_idx_mean: stores still to come, weighted by how it sells where it already is
Tests against the submitted system's backtest (data/fk_err_master.parquet):
  1 rank correlation with the log residual within origin (all items, launches under 6 months, items with stores still to reach)
  2 do they predict the roll-out itself (stores added in the window >= 2)? time-honest classifier AUC
  3 time-honest residual correction (trained on earlier origins only, origin total kept): origin-known reference vs + launch block
Writes data/fk_store_launch_feats.parquet, J["launch"] in data/fk_err_analysis.json, stdout -> eda/store_launch_output.txt."""
import sys, json, numpy as np, pandas as pd, pyarrow.parquet as pq, lightgbm as lgb, warnings
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
T = pq.read_table("Dataset.parquet", columns=["ny_timestamp", "style_color", "qty_ordered", "class", "o_type", "cancelled"], read_dictionary=["style_color", "class", "o_type"]).to_pandas()
T = T[~T.cancelled]; T["day"] = T.ny_timestamp.dt.floor("D")
DL = T.groupby(["style_color", "o_type", "day"], observed=True).qty_ordered.sum().reset_index(); DL.columns = ["item", "loc", "day", "u"]; DL["item"] = DL.item.astype(str); DL["loc"] = DL["loc"].astype(str)
kind = T.groupby("o_type", observed=True)["class"].first().astype(str); kind.index = kind.index.astype(str); del T
KIND = {l: ("store" if k == "RETAIL" else "partner" if k == "DROP" else ("web" if l == "edikted.com" else "tiktok" if l.lower() == "tiktok" else "other")) for l, k in kind.items()}
DL["kind"] = DL["loc"].map(KIND); DL = DL[DL.kind != "other"]
AR = DL.groupby(["item", "loc"]).day.min().rename("a").reset_index(); AR["kind"] = AR["loc"].map(KIND)
X = DL.merge(AR[["item", "loc", "a"]], on=["item", "loc"]); X["d"] = (X.day - X.a).dt.days
AR = AR.merge(X[X.d < 14].groupby(["item", "loc"]).u.sum().rename("u14").reset_index(), on=["item", "loc"], how="left")
START = pd.Timestamp("2024-01-01"); AR["censored"] = AR.a < START + pd.Timedelta(days=21)      # arrival cut by the start of the data: no launch index
AR["idx"] = np.nan
for loc, g in AR[~AR.censored].groupby("loc"):
    g = g.sort_values("a"); av = g.a.values.astype("datetime64[D]").astype(int); uv = g.u14.values; out = np.full(len(g), np.nan)
    for i in range(len(g)):
        m = (av <= av[i] - 14) & (av >= av[i] - 194)
        if m.sum() < 8: m = av <= av[i] - 14
        if m.sum() >= 5: out[i] = np.log((uv[i] + 1) / (np.median(uv[m]) + 1))
    AR.loc[g.index, "idx"] = out
print("locations", AR.groupby("kind")["loc"].nunique().to_dict(), "| arrivals", len(AR), "| with a launch index", int(AR.idx.notna().sum()), flush=True)
D = pd.read_parquet("data/fk_err_master.parquet"); J26 = pd.read_parquet("data/fk_err_master_2026.parquet"); ORIG = sorted(D.origin.unique()); REC = [o for o in ORIG if o >= "2025-06"]
DLs = DL[DL.kind == "store"]; rows = []
for o in ORIG + ["2026-06"]:
    E = pd.Period(o, "M").end_time.normalize(); items = (D[D.origin == o] if o != "2026-06" else J26).style_color.values
    A = AR[AR.a <= E].copy(); A.loc[A.a + pd.Timedelta(days=13) > E, "idx"] = np.nan       # the 14-day window must be complete at the origin
    mon = DLs[(DLs.day > E - pd.offsets.MonthBegin(1) - pd.Timedelta(days=1)) & (DLs.day <= E)]; fleet = mon[mon.u > 0]["loc"].nunique(); parts = DL[(DL.kind == "partner") & (DL.day > E - pd.Timedelta(days=60)) & (DL.day <= E)]["loc"].nunique()
    S = A[A.kind == "store"].sort_values("a"); gs = S.groupby("item")
    f = pd.DataFrame(index=pd.Index(items, name="item"))
    f["st_n"] = gs.size().reindex(items).fillna(0).values; f["st_share"] = f.st_n / max(fleet, 1); f["st_gap"] = np.maximum(fleet - f.st_n, 0)
    f["st_new28"] = S[S.a > E - pd.Timedelta(days=28)].groupby("item").size().reindex(items).fillna(0).values
    f["st_idx_mean"] = gs.idx.mean().reindex(items).values; f["st_idx_sd"] = gs.idx.std().reindex(items).values
    Sn = S[S.idx.notna()]; f["st_idx_first"] = Sn.groupby("item").idx.first().reindex(items).values; f["st_idx_last"] = Sn.groupby("item").idx.last().reindex(items).values
    u28 = DLs[(DLs.day > E - pd.Timedelta(days=28)) & (DLs.day <= E)].groupby("item").u.sum().reindex(items).fillna(0).values; rate = np.where(f.st_n > 0, u28 / np.maximum(f.st_n, 1) / 28, np.nan)
    f["st_rate"] = np.log((rate + 0.01) / (np.nanmedian(rate) + 0.01))
    Pp = A[A.kind == "partner"]; f["pa_n"] = Pp.groupby("item").size().reindex(items).fillna(0).values; f["pa_gap"] = np.maximum(parts - f.pa_n, 0); f["pa_idx"] = Pp.groupby("item").idx.mean().reindex(items).values
    f["web_idx"] = A[A.kind == "web"].set_index("item").idx.reindex(items).values; f["tt_idx"] = A[A.kind == "tiktok"].set_index("item").idx.reindex(items).values
    wa = A[A.kind == "web"].set_index("item").a.reindex(items); sa = gs.a.min().reindex(items); f["lead_web"] = (sa.values - wa.values) / np.timedelta64(1, "D")
    f["potential"] = f.st_gap * f.st_idx_mean.fillna(0); f["fleet"] = fleet; f["origin"] = o; rows.append(f.reset_index().rename(columns={"item": "style_color"}))
F = pd.concat(rows, ignore_index=True); F.to_parquet("data/fk_store_launch_feats.parquet", index=False)
LF = ["st_n", "st_share", "st_gap", "st_new28", "st_idx_mean", "st_idx_sd", "st_idx_first", "st_idx_last", "st_rate", "pa_n", "pa_gap", "pa_idx", "web_idx", "tt_idx", "lead_web", "potential"]
LAB = {"st_n": "own stores reached", "st_share": "share of the fleet reached", "st_gap": "stores not yet reached", "st_new28": "stores first reached in the last 28 days", "st_idx_mean": "launch index, mean over its stores",
       "st_idx_sd": "launch index, spread between stores", "st_idx_first": "launch index at the first store", "st_idx_last": "launch index at the latest store", "st_rate": "units per store per day, vs the fleet",
       "pa_n": "partners reached", "pa_gap": "partners not yet reached", "pa_idx": "launch index at partners", "web_idx": "launch index on edikted.com", "tt_idx": "launch index on TikTok", "lead_web": "days from web launch to first store",
       "potential": "stores to come x launch index"}
D = D.merge(F, on=["origin", "style_color"], how="left"); D["res"] = np.log((D.y + 1) / (D.F + 1)); D["ae"] = (D.F - D.y).abs(); D["ar"] = (D.R3 - D.y).abs(); D["rec"] = D.origin.isin(REC); D["oo"] = D.origin.map(lambda o: pd.Period(o, "M").ordinal)
OUT = dict(coverage={c: r(D[c].notna().mean(), 3) for c in LF}, fleet=[dict(origin=o, fleet=int(F[F.origin == o].fleet.iloc[0])) for o in ORIG + ["2026-06"]])

# 1 rank correlation with the residual
def sig(mask, lab):
    out = []
    for c in LF:
        rr = D[mask].groupby("origin").apply(lambda g: spearmanr(g[c], g.res, nan_policy="omit")[0] if g[c].notna().sum() > 30 and g[c].nunique() > 1 else np.nan, include_groups=False).dropna()
        out.append(dict(key=c, label=LAB[c], rho=r(rr[rr.index.isin(REC)].mean(), 3), rho_all=r(rr.mean(), 3), same=int((np.sign(rr) == np.sign(rr.mean())).sum()), n=int(len(rr))))
    print(f"\n1 {lab}: rank correlation with the residual (recent | all | origins with the same sign)"); [print(f"   {d['label']:44s} {d['rho'] if d['rho'] is not None else float('nan'):+.3f} {d['rho_all'] if d['rho_all'] is not None else float('nan'):+.3f}  {d['same']}/{d['n']}") for d in sorted(out, key=lambda d: -abs(d['rho'] or 0))]
    return sorted(out, key=lambda d: -abs(d["rho"] or 0))
OUT["sig_all"] = sig(np.ones(len(D), bool), "all items"); OUT["sig_young"] = sig((D.age_m < 6).values, "launches under 6 months"); OUT["sig_gap"] = sig((D.st_gap > 0).values & (D.st_n > 0).values, "items in some stores, not all")

# 2 do they predict the roll-out?
D["roll"] = (D.h_stores_chg >= 2).astype(int); D["llv"] = np.log1p(D.level); D["lf"] = np.log((D.F + 1) / (D.R3 + 1)); D["macro_c"] = D.macro.astype("category").cat.codes
PC = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=6)
def th_auc(cols):
    p = np.full(len(D), np.nan)
    for o in REC:
        oo = pd.Period(o).ordinal; tr = D[D.oo + 4 <= oo]
        if tr.roll.nunique() > 1: p[(D.origin == o).values] = lgb.LGBMClassifier(**PC).fit(tr[cols], tr.roll).predict_proba(D.loc[D.origin == o, cols])[:, 1]
    m = ~np.isnan(p); return roc_auc_score(D.roll[m], p[m]), p
a0, _ = th_auc(["llv", "age_m", "n_stores", "trend91", "macro_c"]); a1, proll = th_auc(["llv", "age_m", "n_stores", "trend91", "macro_c"] + LF); D["p_roll"] = proll
OUT["rollout"] = dict(base_rate=r(D[D.rec].roll.mean(), 3), auc_base=r(a0, 3), auc_launch=r(a1, 3), rho_gap=r(np.nanmean([spearmanr(g.st_gap, g.h_stores_chg)[0] for _, g in D[D.rec].groupby("origin")]), 3),
                      by_gap=[dict(g=lab, n=int(m.sum()), roll=r(D[m].roll.mean(), 3), bias=r(D[m].F.sum() / D[m].y.sum() - 1, 3), units=r(D[m].y.sum() / D[D.rec].y.sum(), 3))
                              for lab, m in [("in no store yet", D.rec & (D.st_n == 0)), ("in under half the fleet", D.rec & (D.st_n > 0) & (D.st_share < 0.5)), ("in half to all but one", D.rec & (D.st_share >= 0.5) & (D.st_gap >= 1)), ("in every store", D.rec & (D.st_gap == 0) & (D.st_n > 0))]],
                      top=[dict(q=int(q) + 1, roll=r(g.roll.mean(), 3), bias=r(g.F.sum() / g.y.sum() - 1, 3)) for q, g in D[D.rec & D.p_roll.notna()].assign(q=lambda z: pd.qcut(z.p_roll.rank(method="first"), 5, labels=False)).groupby("q")])
print("\n2 roll-out (stores added in the window >= 2):", {k: v for k, v in OUT["rollout"].items() if k not in ("by_gap", "top")}); [print("   ", d) for d in OUT["rollout"]["by_gap"]]; print("   by predicted roll-out fifth:", OUT["rollout"]["top"])

# 3 time-honest residual correction
D["res_c"] = D.res - D.groupby("origin").res.transform("mean")
BASEF = ["llv", "age_m", "lf", "spread", "bf_window", "macro_c", "disc_web", "size_run", "sh_tiktok", "sh_retail", "sh_partner", "n_stores", "trend91"]
PR = dict(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=6)
def th_corr(cols, shrink=0.5, mask=None):
    f = D.F.values.copy()
    for o in REC:
        oo = pd.Period(o).ordinal; tr = D[D.oo + 4 <= oo]; te = D[D.origin == o]
        if len(tr) < 500: continue
        g = np.exp(shrink * lgb.LGBMRegressor(**PR).fit(tr[cols], tr.res_c, sample_weight=np.log1p(tr.R3)).predict(te[cols]))
        if mask is not None: g = np.where(mask[(D.origin == o).values], g, 1.0)
        v = (te.F + 1) * g - 1; f[(D.origin == o).values] = v * te.F.sum() / v.sum()
    return f
def score(f, m=None):
    m = D.rec.values if m is None else (D.rec.values & m); return (np.abs(f - D.y.values)[m]).sum() / D.ar.values[m].sum(), float((np.abs(f - D.y.values)[m] <= 0.2 * D.y.values[m]).mean()), f[m].sum() / D.y.values[m].sum() - 1
fb = th_corr(BASEF); fl = th_corr(BASEF + LF); fy = th_corr(BASEF + LF, mask=(D.age_m < 12).values); f1 = th_corr(BASEF + LF, shrink=1.0)
rng = np.random.default_rng(0)
def boot(fa, fbb, B=1000):
    Z = pd.DataFrame({"s": D.style_color, "a": np.abs(fa - D.y.values), "b": np.abs(fbb - D.y.values), "r": D.ar.values})[D.rec.values].groupby("s").sum().values; n = len(Z); ix = rng.integers(0, n, (B, n)); S = Z[ix].sum(1)
    d = S[:, 1] / S[:, 2] - S[:, 0] / S[:, 2]; return [r(np.percentile(d, 2.5)), r(np.percentile(d, 97.5))]
young = (D.age_m < 6).values; C = []
for lab, f in [("submitted system", D.F.values), ("+ correction on origin-known features (reference)", fb), ("+ correction with the launch block", fl), ("the same, items under 12 months only", fy), ("launch block at full strength", f1)]:
    w, hh, b = score(f); wy, hy, by = score(f, young); C.append(dict(system=lab, wape=r(w), hit=r(hh), bias=r(b, 3), wape_young=r(wy), hit_young=r(hy), bias_young=r(by, 3)))
OUT["correction"] = C; OUT["boot_launch_vs_ref"] = boot(fb, fl); OUT["boot_launch_vs_system"] = boot(D.F.values, fl); OUT["boot_ref_vs_system"] = boot(D.F.values, fb)
print("\n3 time-honest residual correction (trained on earlier origins, origin total kept):"); print(pd.DataFrame(C).to_string(index=False))
print("   launch block minus reference:", OUT["boot_launch_vs_ref"], "| launch block minus system:", OUT["boot_launch_vs_system"], "| reference minus system:", OUT["boot_ref_vs_system"])
imp = lgb.LGBMRegressor(**PR).fit(D[BASEF + LF], D.res_c, sample_weight=np.log1p(D.R3)); gi = pd.Series(imp.booster_.feature_importance("gain"), index=BASEF + LF); OUT["launch_gain_share"] = r(gi[LF].sum() / gi.sum(), 3)
OUT["top_gain"] = [dict(f=LAB.get(k, k), share=r(v / gi.sum(), 3)) for k, v in gi.sort_values(ascending=False).head(8).items()]; print("   share of the residual model's gain from the launch block:", OUT["launch_gain_share"], OUT["top_gain"][:6])
# 4 the direct use: an item's launch at a later store, predicted from the stores (and the web) it reached earlier
AR.to_parquet("data/fk_store_arrivals.parquet", index=False)
S = AR[(AR.kind == "store") & AR.idx.notna()].sort_values(["item", "a"]); W = AR[(AR.kind == "web")].set_index("item"); rows = []
for it_, g in S.groupby("item"):
    av = g.a.values; iv = g.idx.values; wa = W.a.get(it_, pd.NaT); wi = W.idx.get(it_, np.nan)
    for k in range(len(g)):
        m = av <= av[k] - np.timedelta64(14, "D")
        rows.append(dict(item=it_, loc=g["loc"].values[k], a=av[k], idx=iv[k], u14=g.u14.values[k], n_prev=int(m.sum()), prev_mean=float(iv[m].mean()) if m.any() else np.nan, prev_first=float(iv[m][0]) if m.any() else np.nan,
                         web=float(wi) if pd.notna(wa) and wa <= av[k] - np.timedelta64(14, "D") and pd.notna(wi) else np.nan))
L = pd.DataFrame(rows); L["norm"] = (L.u14 + 1) / np.exp(L.idx) - 1; L1 = L[L.n_prev >= 1]; cut = pd.Timestamp("2025-06-01"); tr, te = L1[L1.a < cut], L1[L1.a >= cut]
def fitpred(cols, trn, tst):
    A_ = np.column_stack([np.ones(len(trn))] + [trn[c].values for c in cols]); b = np.linalg.lstsq(A_, trn.idx.values, rcond=None)[0]
    return b, np.column_stack([np.ones(len(tst))] + [tst[c].values for c in cols]) @ b
def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
b1, p1 = fitpred(["prev_mean"], tr, te); tw, ew = tr[tr.web.notna()], te[te.web.notna()]; b2, p2 = fitpred(["prev_mean", "web"], tw, ew); b3, p3 = fitpred(["web"], tw, ew)
mdape = lambda u, f: float(np.median(np.abs(f - u) / np.maximum(u, 1)))
hit = lambda u, f, t=0.2: float((np.abs(f - u) <= t * np.maximum(u, 1)).mean())
OUT["cross_store"] = dict(n=int(len(L1)), items=int(L1.item.nunique()), corr=r(np.corrcoef(L1.idx, L1.prev_mean)[0, 1], 3), corr_spear=r(spearmanr(L1.idx, L1.prev_mean)[0], 3), corr_first=r(np.corrcoef(L1.idx, L1.prev_first)[0, 1], 3),
                          corr_web=r(np.corrcoef(L1[L1.web.notna()].idx, L1[L1.web.notna()].web)[0, 1], 3), corr_by_nprev=[dict(n_prev=lab, corr=r(np.corrcoef(g.idx, g.prev_mean)[0, 1], 3), n=int(len(g))) for lab, g in [("1 earlier store", L1[L1.n_prev == 1]), ("2-3", L1[(L1.n_prev >= 2) & (L1.n_prev <= 3)]), ("4 or more", L1[L1.n_prev >= 4])]],
                          slope=r(b1[1], 3), r2_out=r(r2(te.idx.values, p1), 3), r2_out_web_only=r(r2(ew.idx.values, p3), 3), r2_out_both=r(r2(ew.idx.values, p2), 3), n_test=int(len(te)),
                          mdape_store_median=r(mdape(te.u14.values, te.norm.values), 3), mdape_with_earlier=r(mdape(te.u14.values, (te.norm.values + 1) * np.exp(p1) - 1), 3),
                          within30_store_median=r(hit(te.u14.values, te.norm.values, 0.3), 3), within30_with_earlier=r(hit(te.u14.values, (te.norm.values + 1) * np.exp(p1) - 1, 0.3), 3), med_u14=r(float(te.u14.median()), 0),
                          gap_days_median=r(float(np.median([(g.a.values[-1] - g.a.values[0]) / np.timedelta64(1, "D") for _, g in S.groupby("item") if len(g) > 1])), 0),
                          first_to_second_days=r(float(np.median([(g.a.values[1] - g.a.values[0]) / np.timedelta64(1, "D") for _, g in S.groupby("item") if len(g) > 1])), 0),
                          share_with_14d_lead=r(float((L.n_prev >= 1).mean()), 3))
ar_all = AR[AR.kind == "store"].sort_values(["item", "a"]); gg = ar_all.groupby("item").a
OUT["cross_store"]["arrive_same_fortnight"] = r(float(((ar_all.a - gg.transform("min")) <= pd.Timedelta(days=13)).mean()), 3)
print("\n4 a later store's launch from earlier stores:", OUT["cross_store"])
J = json.load(open("data/fk_err_analysis.json")); J["launch"] = OUT; json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
