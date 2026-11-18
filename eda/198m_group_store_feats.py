"""D282 (fork; user comment on the error study: "can we have features saying what was the recent group total? and if more stores had
opened"). Two feature blocks, tested on the submitted system's backtest errors.
GROUP  recent actuals of the item's group, for three groupings (type x look, look, product type); members = the items in the group at
       the origin, history from their monthly units up to the origin:
         level (log of the last 3 months), momentum (last 3 months over the 3 before), like-for-like momentum (members at least 6
         months old only), newest-month turn, year-on-year, number of members, the item's share of the group's last 3 months
STORE  store openings: own stores trading at the origin (fleet), opened in the last 3 months, and opening in the forecast window
       (the opening calendar is known to the business ahead; in the backtest it is read from each store's first sales);
       each also weighted by the item's own-store share of units (an opening matters to an item in proportion to what it sells in
       stores), and the stores the item has not reached yet plus the openings to come
Tests: rank correlation with the log residual within origin; a time-honest residual correction (trained on earlier origins only, origin
total kept): origin-known reference vs + GROUP vs + STORE (known at the origin) vs + STORE with the openings to come vs all; and, across
origins, whether own-store units track the number of stores.
Adds J["groupstore"] to data/fk_err_analysis.json; stdout -> eda/group_store_output.txt."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb, warnings
from scipy.stats import spearmanr
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
D = pd.read_parquet("data/fk_err_master.parquet"); ORIG = sorted(D.origin.unique()); REC = [o for o in ORIG if o >= "2025-06"]; D["rec"] = D.origin.isin(REC); D["oo"] = D.origin.map(lambda o: pd.Period(o, "M").ordinal)
J = json.load(open("data/fk_err_analysis.json")); FLEET = {d["m"]: d["stores"] for d in J["fleet"]}
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; P = lambda o, k: str(pd.Period(o, "M") + k)
SL = pd.read_parquet("data/fk_store_launch_feats.parquet")[["origin", "style_color", "st_n", "st_gap", "st_share"]]; D = D.merge(SL, on=["origin", "style_color"], how="left")
win = lambda items, o, ks: M.reindex(items).reindex(columns=[P(o, k) for k in ks]).fillna(0).values.sum(1)
parts = []
for o, g in D.groupby("origin"):
    it = g.style_color.values; l3, p3, y3, m0, m12 = win(it, o, [-2, -1, 0]), win(it, o, [-5, -4, -3]), win(it, o, [-14, -13, -12]), win(it, o, [0]), win(it, o, [-2, -1]) / 2
    f = pd.DataFrame(index=g.index); old = (g.age_m.values >= 6)
    for nm, keys in [("tl", ["ptype", "look"]), ("lk", ["look"]), ("ty", ["ptype"])]:
        code = g.groupby(keys).ngroup().values; k = code.max() + 1; bc = lambda v: np.bincount(code, weights=v, minlength=k)[code]
        L3, P3, Y3 = bc(l3), bc(p3), bc(y3)
        f[f"g_{nm}_lvl"] = np.log1p(L3); f[f"g_{nm}_mom"] = np.log((L3 + 1) / (P3 + 1)); f[f"g_{nm}_lfl"] = np.log((bc(l3 * old) + 1) / (bc(p3 * old) + 1))
        f[f"g_{nm}_acc"] = np.log((bc(m0) + 1) / (bc(m12) + 1)); f[f"g_{nm}_yoy"] = np.where(Y3 > 0, np.log((L3 + 1) / (Y3 + 1)), np.nan); f[f"g_{nm}_n"] = bc(np.ones(len(g))); f[f"g_{nm}_share"] = l3 / np.maximum(L3, 1)
        # the same, relative to all other groups at the origin (second comment: "perhaps show that also relative to all other groups")
        lr = lambda a, b: np.log((a + 1) / (b + 1)); M0, M12, LO, PO = bc(m0), bc(m12), bc(l3 * old), bc(p3 * old); T3, TP, TY, T0, T12, TLO, TPO = l3.sum(), p3.sum(), y3.sum(), m0.sum(), m12.sum(), (l3 * old).sum(), (p3 * old).sum()
        f[f"r_{nm}_share"] = lr(L3, T3); f[f"r_{nm}_mom"] = lr(L3, P3) - lr(T3 - L3, TP - P3); f[f"r_{nm}_lfl"] = lr(LO, PO) - lr(TLO - LO, TPO - PO); f[f"r_{nm}_acc"] = lr(M0, M12) - lr(T0 - M0, T12 - M12)
        f[f"r_{nm}_yoy"] = np.where(Y3 > 0, lr(L3, Y3) - lr(T3 - L3, TY - Y3), np.nan); f[f"r_{nm}_rank"] = pd.Series(lr(L3, P3)).rank(pct=True).values; f[f"r_{nm}_yrank"] = pd.Series(np.where(Y3 > 0, lr(L3, Y3), np.nan)).rank(pct=True).values
    fl0, flm3, flp4 = FLEET.get(o, np.nan), FLEET.get(P(o, -3), np.nan), FLEET.get(P(o, 4), np.nan)
    f["fleet"] = fl0; f["open_recent"] = fl0 - flm3; f["open_future"] = flp4 - fl0
    f["expo_recent"] = g.sh_retail.fillna(0).values * (fl0 - flm3) / max(flm3, 1); f["expo_future"] = g.sh_retail.fillna(0).values * (flp4 - fl0) / max(fl0, 1)
    f["reach_future"] = g.st_gap.fillna(0).values + (flp4 - fl0); f["in_stores"] = (g.st_n.fillna(0).values > 0).astype(float)
    parts.append(f)
F = pd.concat(parts).reindex(D.index); D = pd.concat([D, F], axis=1)
GF = [c for c in F.columns if c.startswith("g_")]; RF = [c for c in F.columns if c.startswith("r_")]; S0 = ["fleet", "open_recent", "expo_recent", "st_n", "st_gap", "st_share", "in_stores"]; S1 = S0 + ["open_future", "expo_future", "reach_future"]
D["res"] = np.log((D.y + 1) / (D.F + 1)); D["res_c"] = D.res - D.groupby("origin").res.transform("mean"); D["llv"] = np.log1p(D.level); D["lf"] = np.log((D.F + 1) / (D.R3 + 1)); D["macro_c"] = D.macro.astype("category").cat.codes
y = D.y.values; ar = (D.R3 - D.y).abs().values; rec = D.rec.values
LAB = {"g_tl_lvl": "type x look: recent total", "g_tl_mom": "type x look: momentum", "g_tl_lfl": "type x look: like-for-like momentum", "g_tl_acc": "type x look: newest-month turn", "g_tl_yoy": "type x look: year on year",
       "g_tl_share": "item's share of its type x look group", "g_lk_mom": "look: momentum", "g_lk_lfl": "look: like-for-like momentum", "g_lk_yoy": "look: year on year", "g_ty_mom": "product type: momentum", "g_ty_lfl": "product type: like-for-like momentum",
       "g_ty_yoy": "product type: year on year", "g_lk_lvl": "look: recent total", "g_ty_lvl": "product type: recent total", "expo_recent": "recent openings x own-store share", "expo_future": "openings to come x own-store share",
       "reach_future": "stores it could still reach (gap + openings to come)", "st_gap": "stores not yet reached", "in_stores": "sold in own stores",
       "r_tl_share": "type x look: share of all units", "r_tl_mom": "type x look: momentum against all other groups", "r_tl_acc": "type x look: newest-month turn against all other groups", "r_tl_yoy": "type x look: year on year against all other groups",
       "r_tl_rank": "type x look: rank of its momentum among groups", "r_tl_yrank": "type x look: rank of its year on year among groups", "r_lk_mom": "look: momentum against all other looks", "r_lk_yoy": "look: year on year against all other looks",
       "r_ty_mom": "product type: momentum against all other types", "r_ty_yoy": "product type: year on year against all other types"}
sg = []
for c in LAB:
    rr = D.groupby("origin").apply(lambda g: spearmanr(g[c], g.res, nan_policy="omit")[0] if g[c].notna().sum() > 30 and g[c].nunique() > 1 else np.nan, include_groups=False).dropna()
    if len(rr): sg.append(dict(key=c, label=LAB[c], rho=r(rr[rr.index.isin(REC)].mean(), 3), rho_all=r(rr.mean(), 3), same=int((np.sign(rr) == np.sign(rr.mean())).sum()), n=int(len(rr))))
sg = sorted(sg, key=lambda d: -abs(d["rho"] or 0)); print("rank correlation with the residual (recent | all | origins with the same sign):"); [print(f"   {d['label']:52s} {d['rho']:+.3f} {d['rho_all']:+.3f}  {d['same']}/{d['n']}") for d in sg]
BASEF = ["llv", "age_m", "lf", "spread", "bf_window", "macro_c", "disc_web", "size_run", "sh_tiktok", "sh_retail", "sh_partner", "n_stores", "trend91"]
PR = dict(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=6)
def th_corr(cols, shrink=0.5):
    f = D.F.values.copy()
    for o in REC:
        tr = D[D.oo + 4 <= pd.Period(o).ordinal]; te = D[D.origin == o]
        if len(tr) < 500: continue
        v = (te.F + 1) * np.exp(shrink * lgb.LGBMRegressor(**PR).fit(tr[cols], tr.res_c, sample_weight=np.log1p(tr.R3)).predict(te[cols])) - 1; f[(D.origin == o).values] = v * te.F.sum() / v.sum()
    return f
hit = lambda f, m=rec: float((np.abs(f - y)[m] <= 0.2 * y[m]).mean()); wape = lambda f, m=rec: float(np.abs(f - y)[m].sum() / ar[m].sum()); rng = np.random.default_rng(0)
def boot(fa, fb_, B=1000):
    Z = pd.DataFrame({"s": D.style_color, "a": np.abs(fa - y), "b": np.abs(fb_ - y), "r": ar, "ha": np.abs(fa - y) <= 0.2 * y, "hb": np.abs(fb_ - y) <= 0.2 * y, "n": 1.0})[rec].groupby("s").sum().values; n = len(Z); T = Z[rng.integers(0, n, (B, n))].sum(1)
    d = T[:, 1] / T[:, 2] - T[:, 0] / T[:, 2]; hd = (T[:, 4] - T[:, 3]) / T[:, 5]; return [r(np.percentile(d, 2.5)), r(np.percentile(d, 97.5))], [r(np.percentile(hd, 2.5), 3), r(np.percentile(hd, 97.5), 3)]
fb = th_corr(BASEF); rows = [dict(variant="submitted system", wape=r(wape(D.F.values)), hit=r(hit(D.F.values), 3)), dict(variant="correction on origin-known features (reference)", wape=r(wape(fb)), hit=r(hit(fb), 3), ci=[0, 0], hit_ci=[0, 0])]
for lab, cols in [("+ recent group totals (type x look, look, type)", GF), ("+ store openings known at the origin", S0), ("+ store openings, including those to come", S1), ("+ group totals and all store openings", GF + S1),
                  ("+ group totals relative to all other groups", RF), ("+ group totals, as they are and relative", GF + RF)]:
    f = th_corr(BASEF + cols); ci, hci = boot(fb, f); rows.append(dict(variant=lab, wape=r(wape(f)), hit=r(hit(f), 3), ci=ci, hit_ci=hci))
print("\ntime-honest residual correction (earlier origins only, origin total kept):"); print(pd.DataFrame(rows).to_string(index=False))
# own-store units against the number of stores, across origins
AX = pd.read_parquet("data/fk_axes_month.parquet"); ret = AX.groupby("m").uc_retail.sum(); tot = AX.groupby("m").units.sum()
ch = [dict(m=m, stores=int(FLEET[m]), retail=int(ret[m]), per_store=r(ret[m] / FLEET[m], 0), share=r(ret[m] / tot[m], 3)) for m in sorted(FLEET) if m >= "2024-07" and m in ret.index]
df = pd.DataFrame(ch); c_lvl = float(np.corrcoef(df.stores, df.retail)[0, 1]); g1 = np.log(df.retail / df.retail.shift(4)).dropna(); g2 = np.log(df.stores / df.stores.shift(4)).dropna(); c_gr = float(np.corrcoef(g1, g2)[0, 1])
bo = [dict(origin=o, fleet=int(FLEET[o]), open_future=int(FLEET[P(o, 4)] - FLEET[o]), retail_growth=r(float(D[D.origin == o].y_ret.sum() / max(D[D.origin == o].b_ret.sum(), 1)) if "y_ret" in D else np.nan, 3)) for o in ORIG]
ex = D[D.rec & (D.sh_retail >= 0.15)]; byo = [dict(g=lab, n=int(m.sum()), bias=r(ex[m].F.sum() / ex[m].y.sum() - 1, 3), units=r(ex[m].y.sum() / D[D.rec].y.sum(), 3)) for lab, m in [("0-1 openings to come", ex.open_future <= 1), ("2-3", (ex.open_future >= 2) & (ex.open_future <= 3)), ("4-5", ex.open_future >= 4)]]
OUT = dict(sig=sg, rows=rows, per_store=ch, corr_level=r(c_lvl, 3), corr_growth4=r(c_gr, 3), per_store_first=ch[0]["per_store"], per_store_last=ch[-1]["per_store"], by_openings=byo, n_group=len(GF), n_rel=len(RF), n_store0=len(S0), n_store1=len(S1))
print("\nown-store units and stores by month:", [(d["m"], d["stores"], d["per_store"]) for d in ch[::3]]); print("correlation of levels", OUT["corr_level"], "| of 4-month growth", OUT["corr_growth4"]); print("items with 15%+ own-store share, bias by openings to come:", byo)
J["groupstore"] = OUT; json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
