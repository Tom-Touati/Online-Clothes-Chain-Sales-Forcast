"""D311 (fork; user in the session: "how about features that signal the trends (each cluster, product type, and the 10 distinct products
that are on a hot trend, their embedding and their trend size, same for falling"). All from data up to the origin.
  GROUP    the item's cluster (per-origin multi-view cluster, data/fk_cluster_labels.parquet) and product type: 3-month momentum, like-for-like
           momentum (members already selling 3 months earlier), year on year, change of share of all units, and the item's own momentum
           against its cluster's and its type's
  HOTCOLD  the 10 distinct products (one colour per style) rising fastest and the 10 falling fastest at the origin, among established items
           (last 3 months >= 300 units and the 3 before >= 150): 3-month momentum = their trend size. Each item's closeness to them in the
           dynamic co-occurrence embedding of that origin (D193, passes the cart gate): highest and top-3 similarity, similarity weighted by
           trend size, the trend size of the closest one, similarity to their trend-weighted centroid, hot minus cold; whether the item is
           itself one of them; the average trend size of the two sets (the strength of the current trend)
Tests: rank correlation with the submitted system's miss; the time-honest correction (as D282); Main's tree member 'ec_ns' refitted with the
blocks (as D307), alone and inside the five-member average, also on top of the D307 origin-known signals.
Writes data/fk_trend_feats.parquet, data/fk_trend_member_preds.parquet; adds J["trend"]; stdout -> eda/trend_feats_output.txt."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb, openpyxl, warnings
from scipy.stats import spearmanr
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
argv0 = sys.argv; sys.argv = ["eda/180q_models.py"]; ns = {}; exec(compile(open("eda/180q_models.py", encoding="utf-8").read(), "eda/180q_models.py", "exec"), ns); sys.argv = argv0
ns["P"]["n_jobs"] = 3   # user: "you can use the laptop but dont let it lag the computer" (3 threads, low priority)
T = ns["T"].reset_index(drop=True); sets = ns["sets"]; fit = ns["fit"]; ORIG = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; Pm = lambda o, k: str(pd.Period(o, "M") + k); sty = lambda s: s.split("_")[0]
ITM = pd.read_parquet("data/items.parquet").set_index("style_color"); PT = ITM.product_type.reindex(M.index).fillna("na"); NM = ITM.product_name if "product_name" in ITM else pd.Series(dtype=str)
CL = pd.read_parquet("data/fk_cluster_labels.parquet"); Z = np.load("data/fk_cooc_dyn.npz", allow_pickle=True); EI = {s: i for i, s in enumerate(Z["items"])}
lg = lambda a, b: np.log((np.asarray(a, float) + 1) / (np.asarray(b, float) + 1)); parts = []; HOT = {}
def grp_feats(key, L3, P3, Y3, own_m):
    g = pd.DataFrame({"L3": L3, "P3": P3, "Y3": Y3, "k": key}); a = g.groupby("k")[["L3", "P3", "Y3"]].sum(); lf = g[g.P3 >= 30].groupby("k")[["L3", "P3"]].sum().reindex(a.index).fillna(0)
    tl, tp = g.L3.sum(), g.P3.sum(); mom = lg(a.L3, a.P3); out = pd.DataFrame({"mom": mom, "lfl": lg(lf.L3, lf.P3), "yoy": np.where(a.Y3 > 0, lg(a.L3, a.Y3), np.nan), "share_chg": np.log((a.L3 + 1) / tl) - np.log((a.P3 + 1) / tp)}, index=a.index)
    return out
for o in sorted(T.origin.unique()):
    idx = np.where((T.origin == o).values)[0]; s = T.style_color.values[idx]
    L3 = M.reindex(columns=[Pm(o, k) for k in (-2, -1, 0)]).fillna(0).sum(1); P3 = M.reindex(columns=[Pm(o, k) for k in (-5, -4, -3)]).fillna(0).sum(1); Y3 = M.reindex(columns=[Pm(o, k) for k in (-14, -13, -12)]).fillna(0).sum(1)
    live = L3.index[(L3 > 0) | (P3 > 0)]; own = pd.Series(lg(L3, P3), index=L3.index); f = pd.DataFrame(index=idx)
    cl = CL[CL.origin == o].set_index("style_color").cluster; cl = cl[~cl.index.duplicated()]
    G = grp_feats(cl.values, L3.reindex(cl.index).values, P3.reindex(cl.index).values, Y3.reindex(cl.index).values, own.reindex(cl.index).values); ck = cl.reindex(s).values
    for c in ["mom", "lfl", "yoy", "share_chg"]: f["cl_" + c] = G[c].reindex(ck).values
    f["cl_rel"] = own.reindex(s).values - f.cl_mom.values
    G = grp_feats(PT.reindex(live).values, L3.reindex(live).values, P3.reindex(live).values, Y3.reindex(live).values, own.reindex(live).values); tk = PT.reindex(s).values
    for c in ["mom", "lfl", "yoy", "share_chg"]: f["ty_" + c] = G[c].reindex(tk).values
    f["ty_rel"] = own.reindex(s).values - f.ty_mom.values
    # the 10 distinct products rising and falling fastest
    el = own[(L3 >= 300) & (P3 >= 150)].to_frame("m"); el["sty"] = [sty(x) for x in el.index]; el = el.loc[np.array([x in EI for x in el.index], bool)]
    hot = el.sort_values("m", ascending=False).drop_duplicates("sty").head(10); cold = el.sort_values("m").drop_duplicates("sty").head(10); HOT[o] = (hot, cold)
    E = Z[o] if o in Z.files and len(hot) >= 5 and len(cold) >= 5 else None   # the first origins have no history 3 months back: block left empty
    if E is not None:
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9); ii = np.array([EI.get(x, -1) for x in s]); V = np.where(ii[:, None] >= 0, E[np.maximum(ii, 0)], np.nan); ss = np.array([sty(x) for x in s], dtype=object)
        for nm, S_, w_ in [("hot", hot, hot.m.clip(lower=0).values), ("cold", cold, (-cold.m).clip(lower=0).values)]:
            H = E[[EI[x] for x in S_.index]]; sim = V @ H.T; sv_ = np.array(S_.sty.tolist(), dtype=object); same = ss[:, None] == sv_[None, :]; sim = np.where(same, np.nan, sim)
            f[nm + "_max"] = np.nanmax(sim, 1); f[nm + "_top3"] = np.nanmean(-np.sort(-np.nan_to_num(sim, nan=-9), 1)[:, :3], 1); sp = np.nan_to_num(np.clip(sim, 0, None)); f[nm + "_w"] = (sp * w_).sum(1) / max(w_.sum(), 1e-9)
            f[nm + "_nn_trend"] = S_.m.values[np.nanargmax(np.nan_to_num(sim, nan=-9), 1)]; c = (H * w_[:, None]).sum(0); f[nm + "_cent"] = V @ (c / max(np.linalg.norm(c), 1e-9)); f["is_" + nm] = np.isin(ss, sv_).astype(float); f[nm + "_size"] = float(S_.m.mean())
        f["hot_minus_cold"] = f.hot_w - f.cold_w
    parts.append(f)
FE = pd.concat(parts).reindex(T.index); GROUP = [c for c in FE.columns if c.startswith(("cl_", "ty_"))]; HC = [c for c in FE.columns if c not in GROUP]; T = pd.concat([T, FE], axis=1)
T[["origin", "style_color"] + GROUP + HC].to_parquet("data/fk_trend_feats.parquet"); print("GROUP", len(GROUP), "| HOTCOLD", len(HC), "| coverage", round(float(T[GROUP + HC].notna().mean().mean()), 3), flush=True)
# ---------- signal and correction on the submitted system
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); K = pd.MultiIndex.from_arrays([D.origin, D.style_color]); D = pd.concat([D, T.set_index(["origin", "style_color"])[GROUP + HC].reindex(K).reset_index(drop=True)], axis=1)
REC = [o for o in ORIG if o >= "2025-06"]; rec = D.origin.isin(REC).values; D["oo"] = D.origin.map(lambda o: pd.Period(o, "M").ordinal); y = D.y.values; F = D.F.values; ar = (D.R3 - D.y).abs().values; rng = np.random.default_rng(0)
D["res"] = np.log((D.y + 1) / (D.F + 1)); D["res_c"] = D.res - D.groupby("origin").res.transform("mean"); D["llv"] = np.log1p(D.level); D["lf"] = np.log((D.F + 1) / (D.R3 + 1)); D["macro_c"] = D.macro.astype("category").cat.codes
hit = lambda f, m=rec: float((np.abs(f - y)[m] <= 0.2 * y[m]).mean()); wape = lambda f, m=rec: float(np.abs(f - y)[m].sum() / ar[m].sum()); bias = lambda f, m=rec: float(f[m].sum() / y[m].sum() - 1)
LAB = {"cl_mom": "cluster: 3-month momentum", "cl_lfl": "cluster: like-for-like momentum", "cl_yoy": "cluster: year on year", "cl_share_chg": "cluster: change of share", "cl_rel": "item's momentum against its cluster's",
       "ty_mom": "product type: 3-month momentum", "ty_lfl": "product type: like-for-like momentum", "ty_yoy": "product type: year on year", "ty_share_chg": "product type: change of share", "ty_rel": "item's momentum against its type's",
       "hot_max": "closest of the 10 rising products", "hot_w": "closeness to the 10 rising, weighted by trend size", "hot_cent": "closeness to the rising centroid", "hot_nn_trend": "trend size of the closest rising product",
       "cold_max": "closest of the 10 falling products", "cold_w": "closeness to the 10 falling, weighted by trend size", "cold_cent": "closeness to the falling centroid", "hot_minus_cold": "rising minus falling closeness", "is_hot": "is one of the 10 rising", "is_cold": "is one of the 10 falling"}
sig = []
for c, lab in LAB.items():
    g = D[rec].groupby("origin"); rr = g.apply(lambda z: spearmanr(z[c], z.res, nan_policy="omit")[0] if z[c].nunique() > 1 else np.nan, include_groups=False).dropna()
    sig.append(dict(key=c, label=lab, rho=r(rr.mean(), 3), same=int((np.sign(rr) == np.sign(rr.mean())).sum()), n=int(len(rr)), cover=r(float(D[rec][c].notna().mean()), 3)))
sig = sorted(sig, key=lambda d: -abs(d["rho"] or 0)); print("rank correlation with the miss | origins with that sign"); [print(f"   {d['label']:52s} {d['rho']:+.3f} {d['same']}/{d['n']}") for d in sig]
hc = D[rec & (D.is_hot == 1)]; cc = D[rec & (D.is_cold == 1)]; flagged = dict(hot_bias=r(hc.F.sum() / hc.y.sum() - 1, 3), hot_n=int(len(hc)), cold_bias=r(cc.F.sum() / cc.y.sum() - 1, 3), cold_n=int(len(cc)), hot_growth=r(hc.y.sum() / hc.R3.sum(), 3), cold_growth=r(cc.y.sum() / cc.R3.sum(), 3))
print("the 10 rising / falling themselves (recent):", flagged)
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
fb = th_corr(BASEF); crow = [dict(variant="correction on origin-known features (reference)", wape=r(wape(fb)), hit=r(hit(fb), 3))]
for lab, cols in [("+ cluster and product-type trends", GROUP), ("+ closeness to the 10 rising and 10 falling products", HC), ("+ both", GROUP + HC)]:
    f = th_corr(BASEF + cols); ci, hci = boot(fb, f); crow.append(dict(variant=lab, n=len(cols), wape=r(wape(f)), hit=r(hit(f), 3), ci=ci, hit_ci=hci))
print(pd.DataFrame(crow).to_string(index=False), flush=True)
# ---------- inside the tree member (as D307)
src = open("eda/198d_error_master.py", encoding="utf-8").read(); src = src[src.index("import sys, glob"):src.index("\ndef build(")]; ax = {}; exec(compile(src, "eda/198d_error_master.py", "exec"), ax)
AX = pd.concat([ax["axes"](g, o, hindsight=False) for o, g in T.groupby("origin")]).reindex(T.index).add_prefix("ax_"); SIG = ["ax_" + c for c in ["oos_28", "newest_colour", "size_run", "size_hhi", "sh_partner", "n_partners", "share_of_style", "n_siblings", "disc_web", "disc_vs_site", "top_day_share"]]; T = pd.concat([T, AX[SIG]], axis=1)
B0, F0_ = sets("ec_ns"); PRED = {}
for lab, add in [("+ cluster and type trends", GROUP), ("+ closeness to the rising and falling products", HC), ("+ both trend blocks", GROUP + HC), ("+ origin-known signals (D307) and both trend blocks", SIG + GROUP + HC)]:
    t0 = time.time(); out = []
    for o in ORIG:
        oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()]; te = T[T.origin == o]; assert (tr.o_ord + 4 <= oo).all()
        z = (fit(tr, te, B0 + add) + fit(tr, te, F0_ + add)) / 2; out.append(pd.DataFrame({"origin": o, "style_color": te.style_color.values, "q50": np.clip((4 * te.level.values + 1) * np.exp(z) - 1, 0, None)}))
    PRED[lab] = pd.concat(out).set_index(["origin", "style_color"]).q50; print(lab, "|", round(time.time() - t0), "s", flush=True)
S = pd.read_parquet("data/fk_member_refit_preds.parquet").set_index(["origin", "style_color"]).reindex(K); MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; X = D[MEM].values.astype(float); corr = (D.F / D.F0).values; allm = np.ones(len(D), bool)
m0 = S["ec_ns (refit, check)"].values; s0 = (X.sum(1) - D.qdyn.values + m0) / 5; mS = S["+ origin-known signals"].values; sS = (X.sum(1) - D.qdyn.values + mS) / 5
mrow = [dict(variant="tree member as submitted", alone=r(wape(m0)), stack=r(wape(s0)), corrected=r(wape(s0 * corr)), corrected_hit=r(hit(s0 * corr), 3), corrected_all=r(wape(s0 * corr, allm))),
        dict(variant="+ origin-known signals (D307)", alone=r(wape(mS)), stack=r(wape(sS)), corrected=r(wape(sS * corr)), corrected_hit=r(hit(sS * corr), 3), corrected_all=r(wape(sS * corr, allm)))]
sv = D[["origin", "style_color"]].copy()
for lab in PRED:
    m = PRED[lab].reindex(K).values; assert np.isfinite(m).all(); st = (X.sum(1) - D.qdyn.values + m) / 5; sv[lab] = m; ref = sS if "D307" in lab else s0; ci, hci = boot(ref, st); po = [wape(st, (D.origin == o).values) - wape(ref, (D.origin == o).values) for o in sorted(D.origin.unique())]
    mrow.append(dict(variant=lab, alone=r(wape(m)), stack=r(wape(st)), stack_ci=ci, corrected=r(wape(st * corr)), corrected_hit=r(hit(st * corr), 3), corrected_all=r(wape(st * corr, allm)), against="D307" if "D307" in lab else "submitted member", wins13=int(sum(v < 0 for v in po))))
print(pd.DataFrame(mrow).to_string(index=False)); sv.to_parquet("data/fk_trend_member_preds.parquet")
# the two sets at the June 2026 origin, for the page
ws = openpyxl.load_workbook("Edikted_Forecasting_Exercise_filled.xlsx", data_only=True)["Forecast"]; WN = {row[1]: row[2] for row in ws.iter_rows(min_row=2, values_only=True) if row[1]}; nm = lambda x: WN.get(x) or (NM.get(x) if len(NM) else "") or ""
h26, c26 = HOT["2026-06"]; sets26 = dict(hot=[dict(item=x, name=nm(x), ptype=str(PT.get(x, "")), trend=r(v, 2)) for x, v in h26.m.items()], cold=[dict(item=x, name=nm(x), ptype=str(PT.get(x, "")), trend=r(v, 2)) for x, v in c26.m.items()])
print("June 2026 rising:", [(d["item"], d["trend"]) for d in sets26["hot"]]); print("June 2026 falling:", [(d["item"], d["trend"]) for d in sets26["cold"]])
J = json.load(open("data/fk_err_analysis.json")); J["trend"] = dict(sig=sig, corr=crow, member=mrow, flagged=flagged, sets26=sets26, n_group=len(GROUP), n_hc=len(HC)); json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
