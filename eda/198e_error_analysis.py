"""D276 (fork; user: "full thorough error analysis ... cohorts, axes of information, overfits, confounds"): the analysis of the
submitted system's backtest errors (data/fk_err_master.parquet, eda/198d). Everything is on 4-month totals; "recent" = the nine
origins from June 2025; WAPE / R3 = sum|F - y| / sum|R3 - y|. Parts:
  1 headline, concentration, direction, shared misses          6 overfit audit: each adopted step at item / style / origin level
  2 by origin (absolute WAPE of the system and of R3)          7 confounds: metric, overlap, survivorship, censoring, mix
  3 cohorts on ~35 axes, each with concentration, bias spread,  8 causes of the large misses (hindsight tags) and the worst misses
    stability and a time-honest extractability test            9 interval coverage by cohort
  4 heat maps (age x volume, age x season, group x season)     10 the 100 targets at June 2026 against the risky cohorts
  5 information audit: origin-known signals, hindsight signals, value of information of each hindsight block
Writes data/fk_err_analysis.json (read by the artifact builder eda/198g) and eda/error_analysis_output.txt (stdout)."""
import sys, json, glob, numpy as np, pandas as pd, lightgbm as lgb, warnings
from scipy.stats import spearmanr
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
D = pd.read_parquet("data/fk_err_master.parquet"); J26 = pd.read_parquet("data/fk_err_master_2026.parquet")
it = pd.read_parquet("data/items.parquet").set_index("style_color")
ORIG = sorted(D.origin.unique()); REC = [o for o in ORIG if o >= "2025-06"]; EARLY = [o for o in ORIG if o < "2025-06"]
D["oo"] = D.origin.map(lambda o: pd.Period(o, "M").ordinal); D["rec"] = D.origin.isin(REC)
D["ae"] = (D.F - D.y).abs(); D["ar"] = (D.R3 - D.y).abs(); D["hit"] = D.ae <= 0.2 * D.y; D["hit_r3"] = D.ar <= 0.2 * D.y; D["res"] = np.log((D.y + 1) / (D.F + 1)); D["lf"] = np.log((D.F + 1) / (D.R3 + 1))
R = D[D.rec].copy(); OUT = {}
rnd = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)


def stats(Z, tot_y=None, tot_e=None):
    y, e, r, f = Z.y.sum(), Z.ae.sum(), Z.ar.sum(), Z.F.sum()
    d = dict(n=int(len(Z)), ratio=rnd(e / r) if r > 0 else None, abs=rnd(e / y) if y > 0 else None, bias=rnd(f / y - 1) if y > 0 else None, r3abs=rnd(r / y) if y > 0 else None, r3bias=rnd(Z.R3.sum() / y - 1) if y > 0 else None)
    d["hit"] = rnd(Z.hit.mean()); d["hit_u"] = rnd(Z.y[Z.hit].sum() / y) if y > 0 else None; d["hit_r3"] = rnd(Z.hit_r3.mean())
    if tot_y: d["unit_share"] = rnd(y / tot_y); d["err_share"] = rnd(e / tot_e)
    return d


# ---------- 1 headline
H = dict(recent=stats(R), all=stats(D), n_items=int(D.style_color.nunique()), n_origins=len(ORIG))
s = R.ae.sort_values(ascending=False).values; cs = np.cumsum(s) / s.sum(); n = len(s)
H["top_share"] = {str(p): rnd(cs[int(n * p / 100) - 1]) for p in [1, 5, 10, 20, 50]}
OUT["lorenz"] = [[rnd((i + 1) / 40, 3), rnd(cs[int(n * (i + 1) / 40) - 1])] for i in range(40)]
H["under_err_share"] = rnd(R.ae[R.F < R.y].sum() / R.ae.sum()); H["under_n_share"] = rnd((R.F < R.y).mean())
mem = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; allu = (R[mem].values < R.y.values[:, None]).all(1); allo = (R[mem].values > R.y.values[:, None]).all(1)
H["all_under_err"] = rnd(R.ae[allu].sum() / R.ae.sum()); H["all_over_err"] = rnd(R.ae[allo].sum() / R.ae.sum()); H["all_under_n"] = rnd(allu.mean()); H["all_over_n"] = rnd(allo.mean())
best_mem = np.abs(R[mem].values - R.y.values[:, None]).min(1); H["oracle_member_ratio"] = rnd(best_mem.sum() / R.ar.sum())
lr = np.abs(np.log((R.F + 1) / (R.y + 1))); lr3 = np.abs(np.log((R.R3 + 1) / (R.y + 1)))
H["within20"] = rnd((R.ae <= 0.2 * R.y).mean()); H["within50"] = rnd((R.ae <= 0.5 * R.y).mean()); H["beyond2x"] = rnd((lr > np.log(2)).mean())
H["within20_r3"] = rnd((R.ar <= 0.2 * R.y).mean()); H["within50_r3"] = rnd((R.ar <= 0.5 * R.y).mean()); H["beyond2x_r3"] = rnd((lr3 > np.log(2)).mean())
H["beyond2x_err"] = rnd(R.ae[lr > np.log(2)].sum() / R.ae.sum())
H["early"] = stats(D[~D.rec]); OUT["headline"] = H

# ---------- 2 by origin
OO = []
for o in ORIG:
    Z = D[D.origin == o]; d = stats(Z); d.update(origin=o, window=f"{str(pd.Period(o) + 1)[2:]}..{str(pd.Period(o) + 4)[2:]}", units=int(Z.y.sum()), train_origins=int(pd.Period(o).ordinal - 4 - pd.Period("2024-03").ordinal + 1),
             young=rnd((Z.age_m < 3).mean()), bf=int(Z.bf_window.iloc[0]), stack_ratio=rnd((Z.F0 - Z.y).abs().sum() / Z.ar.sum())); OO.append(d)
OUT["origins"] = OO

# ---------- 3 cohorts
def band(x, edges, labels): return pd.cut(x, edges, labels=labels, right=False).astype(object).where(x.notna(), "n/a")
def qband(col, k=5, fmt="{:.2f}"):
    x = D[col]; qs = np.unique(np.nanquantile(D.loc[D.rec, col].dropna(), np.linspace(0, 1, k + 1)))
    if len(qs) < 3: return x.map(lambda v: "n/a" if pd.isna(v) else fmt.format(v))
    lab = [f"{fmt.format(qs[i])} to {fmt.format(qs[i + 1])}" for i in range(len(qs) - 1)]; e = qs.copy(); e[0] = -np.inf; e[-1] = np.inf
    return pd.cut(x, e, labels=lab, include_lowest=True).astype(object).where(x.notna(), "n/a")
def topk(col, k): v = D.groupby(col).y.sum().sort_values(ascending=False).index[:k]; return D[col].where(D[col].isin(v), "other")
AX = {}
AX["age"] = ("Item age at the origin", band(D.age_m, [-1, 3, 6, 12, 24, 999], ["under 3 months", "3 to 6 months", "6 to 12 months", "12 to 24 months", "24 months +"]))
D["vq"] = D.groupby("origin")["level"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
vlab = {q: f"fifth {q + 1} (median {int(D[D.vq == q].level.median())}/month)" for q in range(5)}; AX["volume"] = ("Sales level at the origin (fifths within origin)", D.vq.map(vlab))
AX["macro"] = ("Product group", D.macro); AX["ptype"] = ("Product type (top 12 by units)", topk("ptype", 12))
AX["colour_family"] = ("Colour family", topk("colour_family", 8)); AX["aesthetic"] = ("Aesthetic tag (first)", topk("aesthetic", 8)); AX["occasion"] = ("Occasion tag", D.occasion.where(D.occasion == "none", "tagged"))
AX["target"] = ("One of the 100 targets", D.target.map({True: "target", False: "other item"}))
sm = D.origin.str[-2:].astype(int)
AX["season"] = ("Season of the forecast window", sm.map(lambda m: "spring / early summer (origins Feb-Apr)" if m in (2, 3, 4) else "summer into autumn (origins May-Jun)" if m in (5, 6) else "window with November (origins Jul-Oct)" if m in (7, 8, 9, 10) else "holiday and post-holiday (origins Nov-Jan)"))
AX["direction"] = ("Forecast against R3", band(D.lf, [-99, -0.35, -0.1, 0.1, 0.35, 99], ["far below R3 (under 0.7x)", "below (0.7-0.9x)", "about R3 (0.9-1.1x)", "above (1.1-1.4x)", "far above (over 1.4x)"]))
D["mom"] = np.log((D.F0 * 0 + D.R3 / 4 * 0 + 1)); mo = np.log((D.lagr1.fillna(0) * 0 + 1))
AX["momentum"] = ("91-day sales trend at the origin", band(D.trend91, [-99, -0.5, -0.15, 0.15, 0.5, 99], ["falling fast", "falling", "flat", "rising", "rising fast"]).replace("n/a", "no trend yet (young)"))
AX["spread"] = ("Disagreement between the five models (fifths)", qband("spread"))
AX["disc_level"] = ("Web discount in the origin month", band(D.disc_web, [-1, 0.45, 0.55, 0.62, 0.7, 9], ["under 45%", "45-55%", "55-62%", "62-70%", "70% +"]))
AX["disc_mom"] = ("Web discount change into the origin month", band(D.disc_mom, [-9, -0.04, -0.01, 0.01, 0.04, 9], ["cut 4+ pts", "cut 1-4 pts", "flat", "up 1-4 pts", "up 4+ pts"]))
AX["channel"] = ("Main channel mix (last 3 months)", pd.Series(np.select([D.sh_tiktok >= 0.25, D.sh_retail >= 0.3, D.sh_partner >= 0.3], ["TikTok 25% +", "own stores 30% +", "partners 30% +"], "mostly edikted.com"), index=D.index))
AX["tiktok"] = ("TikTok share of units", band(D.sh_tiktok, [-1, 0.005, 0.05, 0.15, 0.3, 9], ["none", "under 5%", "5-15%", "15-30%", "30% +"]))
AX["stores"] = ("Own stores selling it in the origin month", band(D.n_stores, [-1, 1, 4, 8, 12, 99], ["none", "1-3", "4-7", "8-11", "12 +"]))
AX["partners"] = ("Wholesale / marketplace partners in the origin month", band(D.n_partners, [-1, 1, 3, 5, 99], ["none", "1-2", "3-4", "5 +"]))
AX["size_run"] = ("Sizes sold in the origin month / sizes ever sold", band(D.size_run, [-1, 0.6, 0.99, 9], ["under 60% (broken run)", "60-99%", "full run"]))
AX["siblings"] = ("Live colours in the style", band(D.n_siblings, [0, 2, 4, 7, 99], ["1", "2-3", "4-6", "7 +"]))
AX["newest"] = ("Newest colour of its style", D.newest_colour.map({1.0: "newest colour", 0.0: "older colour"}))
AX["style_share"] = ("Share of its style's units", band(D.share_of_style, [-1, 0.15, 0.4, 0.8, 9], ["under 15%", "15-40%", "40-80%", "80% + (dominant)"]))
fy = pd.to_datetime(it.first_sale).reindex(D.style_color).values; pre = D.pre2024 == 1
AX["cohort"] = ("Launch cohort", pd.Series(np.where(pre, "before 2024 (history cut)", pd.Series(pd.to_datetime(fy)).dt.year.astype(str).values + np.where(pd.Series(pd.to_datetime(fy)).dt.month.values <= 6, " H1", " H2")), index=D.index))
AX["has_ly"] = ("Sold in the same months last year", D.has_ly.map({1.0: "yes", 0.0: "no"}))
AX["launch_q"] = ("Launch quarter", D.launch_month.map(lambda m: f"Q{(int(m) - 1) // 3 + 1}"))
AX["burst_day"] = ("Largest day's share of the origin month (fifths)", qband("top_day_share")); AX["burst_hour"] = ("Largest hour's share of online units (fifths)", qband("top_hour_share", fmt="{:.3f}"))
AX["oos"] = ("Days flagged as stock gaps, last 28 days", band(D.oos_28, [-1, 0.001, 0.15, 9], ["none", "some (under 15%)", "15% +"]))
AX["cancel"] = ("Cancelled share of ordered units", band(D.cancel_rate, [-1, 0.005, 0.015, 0.03, 9], ["under 0.5%", "0.5-1.5%", "1.5-3%", "3% +"]))
AX["warm"] = ("Warm-state skew of buyers (thirds)", qband("clim_warm_rel", 3))
AX["summer"] = ("LLM season score", pd.Series(np.select([(D.summer - D.winter) >= 2, (D.summer - D.winter) <= -2], ["summer item", "winter item"], "all-season"), index=D.index))
AX["statement"] = ("LLM statement vs staple", pd.Series(np.select([(D.statement - D.staple) >= 2, (D.statement - D.staple) <= -2], ["statement piece", "staple"], "in between"), index=D.index))
D["iw"] = np.log((D.p90 + 1) / (D.p10 + 1)); AX["interval"] = ("Width of the forecast interval (fifths, P90/P10 in logs)", qband("iw"))
D["mcorr"] = np.log(D.F / D.F0); AX["correction"] = ("Correction applied by the last layer", band(D.mcorr, [-9, -0.05, -0.015, 0.015, 0.05, 9], ["down 5% +", "down 1.5-5%", "none", "up 1.5-5%", "up 5% +"]))
R = D[D.rec].copy(); TY, TE = R.y.sum(), R.ae.sum()
# reference cells for the mix-adjusted index: age x volume x season
cell = AX["age"][1].astype(str) + "|" + D.vq.astype(str) + "|" + AX["season"][1].astype(str); cr = (D[D.rec].groupby(cell[D.rec]).ae.sum() / D[D.rec].groupby(cell[D.rec]).y.sum())
D["exp_e"] = cell.map(cr).fillna(TE / TY) * D.y
half1 = D.origin.isin(ORIG[:7]); CO = {}; LB1 = np.log(D[half1].F.sum() / D[half1].y.sum()); LB2 = np.log(D[~half1].F.sum() / D[~half1].y.sum())
def extract(col):
    """Time-honest: at each recent origin the group's multiplier (sum y / sum F on earlier origins with known outcome)^0.5, total kept."""
    num = 0.0
    for o in REC:
        oo = pd.Period(o).ordinal; past = D[D.oo + 4 <= oo]; te = D[D.origin == o]
        if len(past) == 0: num += te.ae.sum(); continue
        g = past.groupby(col[past.index]); m = ((g.y.sum() + 50) / (g.F.sum() + 50)) ** 0.5
        f = te.F * col[te.index].map(m).fillna(1.0).astype(float); f = f * te.F.sum() / f.sum(); num += (f - te.y).abs().sum()
    return num / D[D.rec].ar.sum() - D[D.rec].ae.sum() / D[D.rec].ar.sum()
for k, (lab, col) in AX.items():
    col = col.astype(str); G = []
    for g, Z in D[D.rec].groupby(col[D.rec]):
        d = stats(Z, TY, TE); d["g"] = g; d["under"] = rnd((Z.F < Z.y).mean()); Za = D[col == g]; d["ratio_all"] = rnd(Za.ae.sum() / Za.ar.sum()); d["adj"] = rnd(Z.ae.sum() / Z.exp_e.sum())
        b1 = D[half1 & (col == g)]; b2 = D[~half1 & (col == g)]
        d["b1"] = rnd(np.log(b1.F.sum() / b1.y.sum()) - LB1) if b1.y.sum() > 0 and len(b1) >= 30 else None; d["b2"] = rnd(np.log(b2.F.sum() / b2.y.sum()) - LB2) if b2.y.sum() > 0 and len(b2) >= 30 else None
        cv = Z[(Z.p10.notna())]; d["cover"] = rnd(((cv.y >= cv.p10) & (cv.y <= cv.p90)).mean()) if len(cv) else None; G.append(d)
    G = sorted(G, key=lambda d: -d["unit_share"]); us = np.array([d["unit_share"] for d in G]); es = np.array([d["err_share"] for d in G]); lb = np.array([np.log1p(d["bias"]) for d in G])
    ok = [d for d in G if d["b1"] is not None and d["b2"] is not None and d["unit_share"] > 0.01]
    agree = sum(d["unit_share"] for d in ok if np.sign(d["b1"]) == np.sign(d["b2"])) / max(sum(d["unit_share"] for d in ok), 1e-9)
    CO[k] = dict(label=lab, groups=G, conc=rnd(0.5 * np.abs(us - es).sum()), bias_sd=rnd(np.sqrt(np.average((lb - np.average(lb, weights=us)) ** 2, weights=us))), sign_agree=rnd(agree), th_gain=rnd(extract(col)))
    print(f"{k:14s} conc {CO[k]['conc']:.3f} bias_sd {CO[k]['bias_sd']:.3f} sign_agree {CO[k]['sign_agree']:.2f} time-honest gain {CO[k]['th_gain']:+.4f}", flush=True)
OUT["cohorts"] = CO

# ---------- 4 heat maps
def heat(a, b, oa=None, ob=None):
    A, B = AX[a][1].astype(str), AX[b][1].astype(str); ra = oa or list(pd.unique(A)); cb = ob or list(pd.unique(B)); M = []
    for x in ra:
        row = []
        for z in cb:
            Z = D[D.rec & (A == x) & (B == z)]; row.append(dict(n=int(len(Z)), hit=rnd(Z.hit.mean(), 3) if len(Z) >= 15 else None, ratio=rnd(Z.ae.sum() / Z.ar.sum(), 3) if Z.ar.sum() > 0 and len(Z) >= 15 else None, bias=rnd(Z.F.sum() / Z.y.sum() - 1, 3) if Z.y.sum() > 0 and len(Z) >= 15 else None, es=rnd(Z.ae.sum() / TE, 4)))
        M.append(row)
    return dict(rows=ra, cols=cb, cells=M)
AGE = ["under 3 months", "3 to 6 months", "6 to 12 months", "12 to 24 months", "24 months +"]; VOL = [vlab[q] for q in range(5)]
SEA = ["spring / early summer (origins Feb-Apr)", "summer into autumn (origins May-Jun)", "window with November (origins Jul-Oct)", "holiday and post-holiday (origins Nov-Jan)"]
DIRS = ["far below R3 (under 0.7x)", "below (0.7-0.9x)", "about R3 (0.9-1.1x)", "above (1.1-1.4x)", "far above (over 1.4x)"]
OUT["heat"] = dict(dir_season=heat("direction", "season", DIRS, SEA), age_vol=heat("age", "volume", AGE, VOL), age_season=heat("age", "season", AGE, SEA), macro_season=heat("macro", "season", None, SEA), age_dir=heat("age", "direction", AGE, ["far below R3 (under 0.7x)", "below (0.7-0.9x)", "about R3 (0.9-1.1x)", "above (1.1-1.4x)", "far above (over 1.4x)"]))

# ---------- 5 information audit
D["sw"] = D.summer - D.winter; D["llv"] = np.log1p(D.level); D["h_m1r"] = np.log((D.h_m1 + 1) / (D.F / 4 + 1))
SIG = {"age_m": "item age", "llv": "sales level (log)", "lf": "forecast vs R3 (log)", "spread": "model disagreement", "iw": "interval width", "trend91": "91-day trend", "sh_tiktok": "TikTok share", "tiktok_mom": "TikTok share change, last month",
       "sh_retail": "own-store share", "sh_partner": "partner share", "n_stores": "own stores selling it", "stores_mom": "stores added in 3 months", "n_partners": "partners selling it", "size_run": "size-run completeness", "size_hhi": "size concentration",
       "state_hhi": "state concentration", "top_day_share": "largest day's share", "top_hour_share": "largest hour's share", "multi_share": "units in multi-unit lines", "wkend_share": "weekend share", "disc_web": "web discount level", "disc_mom": "web discount change",
       "disc_vs_site": "discount vs the site's", "oos_28": "stock-gap days, 28 d", "cancel_rate": "cancelled share", "n_siblings": "live colours in style", "share_of_style": "share of style's units", "newest_colour": "newest colour", "hist_months": "months of history",
       "clim_warm_rel": "warm-state skew", "sw": "summer minus winter score", "statement": "statement score", "peak_r": "peak ratio"}
HSIG = {"h_m1r": "month-1 sales vs forecast pace (early read)", "h_disc_chg": "web discount change into the window", "h_disc_chg_rel": "the same, relative to the site", "h_stores_chg": "own stores added in the window", "h_partners_chg": "partners added in the window",
        "h_retail_chg": "own-store share change", "h_partner_chg": "partner share change", "h_tiktok_chg": "TikTok share change", "h_size_run": "size-run completeness in the window", "h_oos": "stock-gap days in the window", "h_top_day": "largest day's share in the window",
        "h_new_style": "new colours of the style launched in the window", "h_new_type": "launches of the same type in the window", "h_dead": "no sales in the window's last month"}
def sig_table(S):
    T = []
    for c, lab in S.items():
        r = D.groupby("origin").apply(lambda g: spearmanr(g[c], g.res, nan_policy="omit")[0] if g[c].notna().sum() > 30 and g[c].nunique() > 1 else np.nan, include_groups=False)
        rr = r[r.index.isin(REC)]; T.append(dict(key=c, label=lab, rho=rnd(rr.mean(), 3), rho_all=rnd(r.mean(), 3), sign=rnd((np.sign(r.dropna()) == np.sign(r.mean())).mean(), 2), cover=rnd(D[c].notna().mean(), 2)))
    return sorted(T, key=lambda d: -abs(d["rho"] or 0))
OUT["sig_origin"] = sig_table(SIG); OUT["sig_hind"] = sig_table(HSIG)
print("\norigin-known signals (rho with residual, recent | all | sign stability):"); [print(f"  {d['label']:36s} {d['rho']:+.3f} {d['rho_all']:+.3f} {d['sign']}") for d in OUT["sig_origin"][:14]]
print("hindsight signals:"); [print(f"  {d['label']:48s} {d['rho']:+.3f} {d['rho_all']:+.3f} {d['sign']}") for d in OUT["sig_hind"]]
D["macro_c"] = D.macro.astype("category").cat.codes
BASEF = ["llv", "age_m", "lf", "spread", "bf_window", "macro_c", "disc_web", "size_run", "sh_tiktok", "sh_retail", "sh_partner", "n_stores", "trend91"]
BLK = {"promotion depth (discount change in the window)": ["h_disc_chg", "h_disc_chg_rel"], "distribution (stores, partners, channel shifts)": ["h_stores_chg", "h_partners_chg", "h_retail_chg", "h_partner_chg", "h_tiktok_chg"],
       "availability (size run, stock gaps, delisting)": ["h_size_run", "h_oos", "h_dead"], "launch plan (new colours, same-type launches)": ["h_new_style", "h_new_type"], "early read (month-1 sales)": ["h_m1r"]}
D["res_c"] = D.res - D.groupby("origin").res.transform("mean"); VF = {}
def voi(cols):
    """Cross-fitted on origins whose windows do not overlap the test origin's; target = residual minus its origin mean; origin total kept."""
    f = D.F.values.copy()
    for o in ORIG:
        oo = pd.Period(o).ordinal; tr = D[(D.oo - oo).abs() >= 4]; te = D[D.origin == o]
        m = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=6).fit(tr[cols], tr.res_c, sample_weight=np.log1p(tr.R3))
        g = (te.F + 1) * np.exp(m.predict(te[cols])) - 1; f[D.origin == o] = g * te.F.sum() / g.sum()
    Z = D.assign(f=f); r = Z[Z.rec]; VF[tuple(cols)] = f; return (r.f - r.y).abs().sum() / r.ar.sum(), r.f.sum() / r.y.sum() - 1, (Z.f - Z.y).abs().sum() / Z.ar.sum(), float(((r.f - r.y).abs() <= 0.2 * r.y).mean())
b0 = voi(BASEF); V = [dict(block="origin-known features only (reference)", wape=rnd(b0[0]), bias=rnd(b0[1]), wape_all=rnd(b0[2]), delta=0.0, hit=rnd(b0[3]))]
for k, c in BLK.items(): v = voi(BASEF + c); V.append(dict(block=k, wape=rnd(v[0]), bias=rnd(v[1]), wape_all=rnd(v[2]), delta=rnd(v[0] - b0[0]), hit=rnd(v[3])))
pl = sum([BLK[k] for k in list(BLK)[:4]], []); v = voi(BASEF + pl); V.append(dict(block="all four plannable blocks together", wape=rnd(v[0]), bias=rnd(v[1]), wape_all=rnd(v[2]), delta=rnd(v[0] - b0[0]), hit=rnd(v[3])))
v = voi(BASEF + pl + BLK["early read (month-1 sales)"]); V.append(dict(block="plannable blocks + early read", wape=rnd(v[0]), bias=rnd(v[1]), wape_all=rnd(v[2]), delta=rnd(v[0] - b0[0]), hit=rnd(v[3])))
pd.DataFrame({"origin": D.origin, "style_color": D.style_color, "f_ref": VF[tuple(BASEF)], "f_plan": VF[tuple(BASEF + pl)], "f_all": VF[tuple(BASEF + pl + BLK["early read (month-1 sales)"])]}).to_parquet("data/fk_err_voi_preds.parquet", index=False)
OUT["voi"] = V; OUT["voi_base_system"] = rnd(D[D.rec].ae.sum() / D[D.rec].ar.sum())
print("\nvalue of information (cross-fitted on non-overlapping origins; oracle features):"); [print(f"  {d['block']:52s} {d['wape']:.4f} ({d['delta']:+.4f}) bias {d['bias']:+.3f}") for d in V]

# ---------- 6 overfit audit
CH = [("R3 + rules -> first three-way (rules x quantile x NN)", "rules", "s_rq"), ("+ weather, events, Black Friday fix (D161)", "s_rq", "s_wx3bf"), ("+ time-aware co-purchase neighbours (D203)", "s_wx3bf", "s_dyn3bf"),
      ("+ TabPFN v2 member (D204, p2)", "s_dyn3bf", "s_p2"), ("arithmetic stack, conformal + 32-estimator TabPFN (D207)", "s_p2", "s_p2c32a"), ("new tree member: events, launch curves (D253)", "s_p2c32a", "s_ecns"),
      ("new NN inputs (D272)", "s_ecns", "F0"), ("segment + breakout corrections (D273)", "F0", "F")]
rng = np.random.default_rng(0); RR = D[D.rec]
def boot(a, b, key, B=1000):
    g = RR.groupby(key).apply(lambda z: pd.Series({"a": (z[a] - z.y).abs().sum(), "b": (z[b] - z.y).abs().sum(), "r": z.ar.sum()}), include_groups=False).values; n = len(g)
    ix = rng.integers(0, n, (B, n)); S = g[ix].sum(1); d = S[:, 1] / S[:, 2] - S[:, 0] / S[:, 2]; return [rnd(np.percentile(d, 2.5)), rnd(np.percentile(d, 97.5))], float(d.std())
def boot_origin(a, b, B=2000, L=3):
    g = RR.groupby("origin").apply(lambda z: pd.Series({"a": (z[a] - z.y).abs().sum(), "b": (z[b] - z.y).abs().sum(), "r": z.ar.sum()}), include_groups=False).reindex(REC).values; n = len(g); out = []
    for _ in range(B):
        st = rng.integers(0, n, n // L); ix = np.concatenate([(s + np.arange(L)) % n for s in st]); S = g[ix].sum(0); out.append(S[1] / S[2] - S[0] / S[2])
    return [rnd(np.percentile(out, 2.5)), rnd(np.percentile(out, 97.5))]
CHN = []; w = lambda Z, c: (Z[c] - Z.y).abs().sum() / Z.ar.sum()
for lab, a, b in CH:
    po = [w(D[D.origin == o], b) - w(D[D.origin == o], a) for o in ORIG]; ci_i, sd = boot(a, b, "style_color"); ci_s, _ = boot(a, b, "style")
    CHN.append(dict(step=lab, frm=rnd(w(RR, a)), to=rnd(w(RR, b)), d_recent=rnd(w(RR, b) - w(RR, a)), d_all=rnd(w(D, b) - w(D, a)), d_early=rnd(w(D[~D.rec], b) - w(D[~D.rec], a)),
                    wins13=int(sum(x < 0 for x in po)), wins9=int(sum(x < 0 for x, o in zip(po, ORIG) if o in REC)), per_origin=[rnd(x) for x in po], ci_item=ci_i, ci_style=ci_s, ci_origin=boot_origin(a, b), sd_item=rnd(sd, 5)))
    print(f"{lab[:58]:58s} {CHN[-1]['d_recent']:+.4f} all {CHN[-1]['d_all']:+.4f} early {CHN[-1]['d_early']:+.4f} wins {CHN[-1]['wins9']}/9 {CHN[-1]['wins13']}/13 item {ci_i} style {ci_s} origin {CHN[-1]['ci_origin']}")
OUT["chain"] = CHN
nq = len(glob.glob("data/q_backtest_*.parquet")); sdm = float(np.median([c["sd_item"] for c in CHN[4:]]))
OUT["multiplicity"] = dict(variant_files=nq, decisions=sum(1 for l in open("log.md", encoding="utf-8") if l.startswith("- **D")), scripts=len(glob.glob("eda/*.py")), sd_item=rnd(sdm, 5), null_max=rnd(sdm * np.sqrt(2 * np.log(nq)), 4),
                           null_max_20=rnd(sdm * np.sqrt(2 * np.log(20)), 4))
print("multiplicity", OUT["multiplicity"])

# ---------- 6b distribution: fleet growth and where the extra units came from
AXM = pd.read_parquet("data/fk_axes_month.parquet")
NONSTORE = {"u_cancel", "u_edikted_com", "u_tiktok", "u_flip", "u_pacsun", "u_nordstrom", "u_macys", "u_tillys", "u_bloomingdales", "u_urban", "u_hottopic", "u_intl"}
STC = [c for c in AXM.columns if c.startswith("u_") and c not in NONSTORE]
fleet = (AXM.groupby("m")[STC].sum() > 50).sum(1); OUT["fleet"] = [dict(m=m, stores=int(v)) for m, v in fleet.items()]
AI = AXM.set_index(["style_color", "m"]); AI["u_web"] = AI.u_edikted_com; AI["u_tt"] = AI.u_tiktok; AI["u_ret"] = AI.uc_retail; AI["u_par"] = AI.uc_drop
def chw(col, ks):
    out = np.zeros(len(D))
    for k in ks: out += AI[col].reindex(pd.MultiIndex.from_arrays([D.style_color, [str(pd.Period(o, "M") + k) for o in D.origin]])).fillna(0).values
    return out
for c in ["web", "tt", "ret", "par"]: D["y_" + c] = chw("u_" + c, [1, 2, 3, 4]); D["b_" + c] = chw("u_" + c, [-2, -1, 0]) * 4 / 3
D["fleet_chg"] = [int(fleet.get(str(pd.Period(o) + 4), np.nan) - fleet.get(o, np.nan)) for o in D.origin]
RR = D[D.rec]; bigu = RR[(RR.y > 1.65 * RR.F) & (RR.y - RR.F >= 50)]
def chan_mix(Z): t = {c: float((Z["y_" + c] - Z["b_" + c]).sum()) for c in ["web", "tt", "ret", "par"]}; tot = sum(t.values()); return {c: rnd(v / tot, 3) for c, v in t.items()}
OUT["channel_growth"] = dict(big_under=chan_mix(bigu), big_under_base={c: rnd(bigu["b_" + c].sum() / sum(bigu["b_" + k].sum() for k in ["web", "tt", "ret", "par"]), 3) for c in ["web", "tt", "ret", "par"]},
                             all_units={c: rnd(RR["y_" + c].sum() / sum(RR["y_" + k].sum() for k in ["web", "tt", "ret", "par"]), 3) for c in ["web", "tt", "ret", "par"]})
D["g_web"] = np.log((D.y_web + 1) / (D.b_web + 1)); D["res_web"] = np.log((D.y_web + 1) / (D.F * D.b_web / np.maximum(D.b_web + D.b_tt + D.b_ret + D.b_par, 1) + 1))
rs = lambda a, b: rnd(np.nanmean([spearmanr(g[a], g[b], nan_policy="omit")[0] for _, g in D.groupby("origin")]), 3)
st = D[D.n_stores >= 1]
OUT["store_check"] = dict(rho_resid=rs("h_stores_chg", "res"), rho_web_resid=rs("h_stores_chg", "res_web"), rho_web_growth=rs("h_stores_chg", "g_web"),
                          add2=dict(n=rnd((RR.h_stores_chg >= 2).mean(), 3), units=rnd(RR.y[RR.h_stores_chg >= 2].sum() / RR.y.sum(), 3), bias=rnd(RR.F[RR.h_stores_chg >= 2].sum() / RR.y[RR.h_stores_chg >= 2].sum() - 1, 3), err=rnd(RR.ae[RR.h_stores_chg >= 2].sum() / RR.ae.sum(), 3),
                                    web_bias=rnd((RR.F * RR.b_web / np.maximum(RR.b_web + RR.b_tt + RR.b_ret + RR.b_par, 1))[RR.h_stores_chg >= 2].sum() / RR.y_web[RR.h_stores_chg >= 2].sum() - 1, 3)),
                          flat=dict(n=rnd((RR.h_stores_chg.abs() < 2).mean(), 3), bias=rnd(RR.F[RR.h_stores_chg.abs() < 2].sum() / RR.y[RR.h_stores_chg.abs() < 2].sum() - 1, 3),
                                    web_bias=rnd((RR.F * RR.b_web / np.maximum(RR.b_web + RR.b_tt + RR.b_ret + RR.b_par, 1))[RR.h_stores_chg.abs() < 2].sum() / RR.y_web[RR.h_stores_chg.abs() < 2].sum() - 1, 3)),
                          lost=dict(n=rnd((RR.h_stores_chg <= -2).mean(), 3), bias=rnd(RR.F[RR.h_stores_chg <= -2].sum() / RR.y[RR.h_stores_chg <= -2].sum() - 1, 3),
                                    web_bias=rnd((RR.F * RR.b_web / np.maximum(RR.b_web + RR.b_tt + RR.b_ret + RR.b_par, 1))[RR.h_stores_chg <= -2].sum() / RR.y_web[RR.h_stores_chg <= -2].sum() - 1, 3)),
                          retail_share_by_origin=[dict(origin=o, share=rnd(D[D.origin == o].y_ret.sum() / D[D.origin == o].y.sum(), 3), fleet=int(fleet.get(o, 0)), fleet_chg=int(D[D.origin == o].fleet_chg.iloc[0])) for o in ORIG])
print("fleet", dict(list(fleet.items())[::3])); print("channel growth", OUT["channel_growth"]); print("store check", {k: OUT["store_check"][k] for k in ["rho_resid", "rho_web_resid", "rho_web_growth", "add2", "flat", "lost"]})

# ---------- 7 confounds
C = {}
ro = pd.DataFrame(OO); C["metric_corr"] = rnd(np.corrcoef(ro.r3abs, ro.ratio)[0, 1], 3); C["sys_abs_range"] = [rnd(ro["abs"].min(), 3), rnd(ro["abs"].max(), 3)]; C["r3_abs_range"] = [rnd(ro.r3abs.min(), 3), rnd(ro.r3abs.max(), 3)]
C["pooled_abs"] = dict(system=rnd(D.ae.sum() / D.y.sum(), 3), r3=rnd(D.ar.sum() / D.y.sum(), 3), system_recent=rnd(RR.ae.sum() / RR.y.sum(), 3), r3_recent=rnd(RR.ar.sum() / RR.y.sum(), 3))
P = D.pivot(index="style_color", columns="origin", values="res"); lag = []
for k in range(1, 7):
    cs_ = [np.corrcoef(P[ORIG[i]].dropna().reindex(P[ORIG[i + k]].dropna().index).dropna(), P[ORIG[i + k]].reindex(P[ORIG[i]].dropna().index).dropna())[0, 1] for i in range(len(ORIG) - k)]
    lag.append(dict(lag=k, corr=rnd(np.nanmean(cs_), 3)))
C["lag_corr"] = lag
ls = pd.to_datetime(it.last_sale).dt.to_period("M").astype(str); C["last_sale"] = {k: int(v) for k, v in ls.value_counts().sort_index().items()}; C["n_catalogue"] = int(len(it)); C["alive_at_end"] = int((ls == "2026-06").sum())
C["dead_by_origin"] = [dict(origin=o, dead=rnd(D[D.origin == o].h_dead.mean(), 4), collapse=rnd((D[D.origin == o].y < 0.25 * D[D.origin == o].R3).mean(), 4)) for o in ORIG]
ov = RR[RR.F > RR.y]; flag = (ov.h_oos > 0.10) | (ov.h_size_run < 0.6)
C["censor"] = dict(over_err_share=rnd(ov.ae.sum() / RR.ae.sum()), flagged_share_of_over_err=rnd(ov.ae[flag].sum() / ov.ae.sum()), flagged_n=rnd(flag.mean()), bias_flagged=rnd(RR[(RR.h_oos > 0.10) | (RR.h_size_run < 0.6)].F.sum() / RR[(RR.h_oos > 0.10) | (RR.h_size_run < 0.6)].y.sum() - 1),
                   bias_clean=rnd(RR[~((RR.h_oos > 0.10) | (RR.h_size_run < 0.6))].F.sum() / RR[~((RR.h_oos > 0.10) | (RR.h_size_run < 0.6))].y.sum() - 1), units_flagged=rnd(RR[(RR.h_oos > 0.10) | (RR.h_size_run < 0.6)].y.sum() / RR.y.sum()),
                   ratio_flagged=rnd(w(RR[(RR.h_oos > 0.10) | (RR.h_size_run < 0.6)], "F")), ratio_clean=rnd(w(RR[~((RR.h_oos > 0.10) | (RR.h_size_run < 0.6))], "F")))
C["mix"] = [dict(origin=o, n=int((D.origin == o).sum()), young=rnd((D[D.origin == o].age_m < 6).mean(), 3), ratio_mature=rnd(w(D[(D.origin == o) & (D.age_m >= 12)], "F"), 3) if ((D.origin == o) & (D.age_m >= 12)).sum() > 30 else None,
                 ratio_young=rnd(w(D[(D.origin == o) & (D.age_m < 6)], "F"), 3), stores=rnd(D[D.origin == o].n_stores.max(), 0), tiktok=rnd((D[D.origin == o].sh_tiktok * D[D.origin == o].R3).sum() / D[D.origin == o].R3.sum(), 3)) for o in ORIG]
brk = RR.assign(b=(RR.y >= 1.65 * RR.F) & (RR.y - RR.F >= 20)); C["breakout_by_launch_q"] = [dict(q=f"Q{q}", rate=rnd(brk[(brk.age_m < 6) & (((brk.launch_month - 1) // 3 + 1) == q)].b.mean(), 3), n=int(((brk.age_m < 6) & (((brk.launch_month - 1) // 3 + 1) == q)).sum())) for q in [1, 2, 3, 4]]
OUT["confounds"] = C
print("confounds:", {k: C[k] for k in ["metric_corr", "sys_abs_range", "r3_abs_range", "pooled_abs", "lag_corr", "censor", "alive_at_end"]})

TGs = set(pd.read_csv("data/targets.csv").style_color); D["is_t"] = D.style_color.isin(TGs)
C["target_bias"] = [dict(origin=o, target=rnd(D[(D.origin == o) & D.is_t].F.sum() / D[(D.origin == o) & D.is_t].y.sum() - 1, 3), other=rnd(D[(D.origin == o) & ~D.is_t].F.sum() / D[(D.origin == o) & ~D.is_t].y.sum() - 1, 3),
                         n_t=int(((D.origin == o) & D.is_t).sum()), r3_target=rnd(D[(D.origin == o) & D.is_t].R3.sum() / D[(D.origin == o) & D.is_t].y.sum() - 1, 3), r3_other=rnd(D[(D.origin == o) & ~D.is_t].R3.sum() / D[(D.origin == o) & ~D.is_t].y.sum() - 1, 3)) for o in ORIG]
u26 = pd.read_parquet("data/monthly_item.parquet"); u26.columns = [str(c) for c in u26.columns]; h1 = u26[[c for c in u26.columns if "2026-01" <= c <= "2026-06"]].sum(1).sort_values(ascending=False)
rk = pd.Series(np.arange(1, len(h1) + 1), index=h1.index); C["target_rank"] = dict(median_rank=int(rk.reindex(list(TGs)).median()), in_top100=int((rk.reindex(list(TGs)) <= 100).sum()), in_top200=int((rk.reindex(list(TGs)) <= 200).sum()), in_top400=int((rk.reindex(list(TGs)) <= 400).sum()),
                        units_share_2026h1=rnd(h1.reindex(list(TGs)).sum() / h1.sum(), 3))
grow = lambda S: rnd(np.log((u26.reindex(list(S))[[c for c in u26.columns if "2026-01" <= c <= "2026-06"]].sum().sum() + 1) / (u26.reindex(list(S))[[c for c in u26.columns if "2025-01" <= c <= "2025-06"]].sum().sum() + 1)), 3)
C["target_growth"] = dict(target=grow(TGs), other=grow(set(u26.index) - TGs))
print("target bias by origin:", [(d["origin"], d["target"], d["other"]) for d in C["target_bias"]]); print("target rank", C["target_rank"], C["target_growth"])
# ---------- 8 causes of the large misses and the worst misses
big = RR[(np.abs(np.log((RR.F + 1) / (RR.y + 1))) > np.log(1.65)) & (RR.ae >= 50)].copy(); un = big[big.y > big.F]; ov = big[big.F > big.y]
OUT["big"] = dict(n=int(len(big)), n_share=rnd(len(big) / len(RR)), err_share=rnd(big.ae.sum() / RR.ae.sum()), under_err=rnd(un.ae.sum() / RR.ae.sum()), over_err=rnd(ov.ae.sum() / RR.ae.sum()))
def tags(Z, T): return [dict(tag=k, err_share=rnd(Z.ae[m(Z)].sum() / Z.ae.sum()), n_share=rnd(m(Z).mean()), base=rnd(m(RR).mean())) for k, m in T.items()]
TU = {"young launch (under 6 months)": lambda z: z.age_m < 6, "month 1 already 1.5x the forecast pace": lambda z: z.h_m1 >= 1.5 * z.F / 4, "web discount deepened 3+ pts vs the site": lambda z: z.h_disc_chg_rel >= 0.03,
      "own stores added (2+)": lambda z: z.h_stores_chg >= 2, "partner added": lambda z: z.h_partners_chg >= 1, "TikTok share up 5+ pts": lambda z: z.h_tiktok_chg >= 0.05, "one day carried 15%+ of a month (spike)": lambda z: z.h_top_day >= 0.15,
      "window includes November": lambda z: z.bf_window == 1, "all five models under": lambda z: (z[mem].values < z.y.values[:, None]).all(1), "none of the tags above (except the last two)": lambda z: ~((z.age_m < 6) | (z.h_m1 >= 1.5 * z.F / 4) | (z.h_disc_chg_rel >= 0.03) | (z.h_stores_chg >= 2) | (z.h_partners_chg >= 1) | (z.h_tiktok_chg >= 0.05) | (z.h_top_day >= 0.15))}
TO = {"size run broken or stock gaps in the window": lambda z: (z.h_size_run < 0.6) | (z.h_oos > 0.10), "delisted: no sales in the last month": lambda z: z.h_dead == 1, "web discount pulled back 3+ pts vs the site": lambda z: z.h_disc_chg_rel <= -0.03,
      "own stores lost (2+)": lambda z: z.h_stores_chg <= -2, "partner lost": lambda z: z.h_partners_chg <= -1, "new colours of the style launched": lambda z: z.h_new_style >= 1, "spike in the origin month (top day 15%+)": lambda z: z.top_day_share >= 0.15,
      "falling fast at the origin": lambda z: z.trend91 < -0.5, "month 1 already under 0.67x the forecast pace": lambda z: z.h_m1 <= 0.67 * z.F / 4, "all five models over": lambda z: (z[mem].values > z.y.values[:, None]).all(1),
      "none of the first seven tags": lambda z: ~(((z.h_size_run < 0.6) | (z.h_oos > 0.10)) | (z.h_dead == 1) | (z.h_disc_chg_rel <= -0.03) | (z.h_stores_chg <= -2) | (z.h_partners_chg <= -1) | (z.h_new_style >= 1) | (z.top_day_share >= 0.15))}
OUT["causes"] = dict(under=tags(un, TU), over=tags(ov, TO))
print("\nlarge under-forecasts:"); [print(f"  {d['tag']:50s} error {d['err_share']:.2f} items {d['n_share']:.2f} base {d['base']:.2f}") for d in OUT["causes"]["under"]]
print("large over-forecasts:"); [print(f"  {d['tag']:50s} error {d['err_share']:.2f} items {d['n_share']:.2f} base {d['base']:.2f}") for d in OUT["causes"]["over"]]
wr = RR.sort_values("ae", ascending=False).drop_duplicates("style_color").head(16)
OUT["worst"] = [dict(item=r.style_color, name=str(it.product_name.get(r.style_color)), origin=r.origin, y=int(r.y), F=int(r.F), R3=int(r.R3), age=rnd(r.age_m, 1), lo=int(min(getattr(r, m) for m in mem)), hi=int(max(getattr(r, m) for m in mem)), m1=rnd(r.h_m1 / (r.F / 4), 2),
                     disc=rnd(r.h_disc_chg_rel, 3), stores=rnd(r.h_stores_chg, 0), partners=rnd(r.h_partners_chg, 0), topday=rnd(r.h_top_day, 2), tiktok=rnd(r.h_tiktok_chg, 3), newcol=rnd(r.h_new_style, 0), size=rnd(r.h_size_run, 2), macro=r.macro) for r in wr.itertuples()]

# ---------- 9 coverage
cv = RR[RR.p10.notna()]; OUT["coverage"] = dict(overall=rnd(((cv.y >= cv.p10) & (cv.y <= cv.p90)).mean(), 3), below=rnd((cv.y < cv.p10).mean(), 3), above=rnd((cv.y > cv.p90).mean(), 3))

# ---------- 10 the 100 targets at June 2026
TG = set(pd.read_csv("data/targets.csv").style_color); J = J26[J26.style_color.isin(TG)].copy(); J["lf"] = np.log((J.F + 1) / (J.R3 + 1))
age_g = {d["g"]: d for d in CO["age"]["groups"]}; ab = band(J.age_m, [-1, 3, 6, 12, 24, 999], AGE).astype(str)
cellrate = D[D.rec].groupby([AX["age"][1][D.rec].astype(str), AX["direction"][1][D.rec].astype(str)]).apply(lambda z: z.ae.sum() / z.y.sum(), include_groups=False)
dirb = band(J.lf, [-99, -0.35, -0.1, 0.1, 0.35, 99], ["far below R3 (under 0.7x)", "below (0.7-0.9x)", "about R3 (0.9-1.1x)", "above (1.1-1.4x)", "far above (over 1.4x)"]).astype(str)
J["exp_abs"] = [cellrate.get((a, d), np.nan) for a, d in zip(ab, dirb)]; J["age_band"] = ab.values; J["dir_band"] = dirb.values
flags = {"young (under 6 months)": J.age_m < 6, "models disagree (top fifth of spread)": J.spread > R.spread.quantile(0.8), "forecast far above R3": J.lf > 0.35, "forecast far below R3": J.lf < -0.35, "falling fast": J.trend91 < -0.5,
         "broken size run": J.size_run < 0.6, "TikTok 15% +": J.sh_tiktok >= 0.15, "spike in June (top day 15%+)": J.top_day_share >= 0.15, "dress": J.macro == "dress"}
OUT["targets26"] = dict(n=int(len(J)), total=int(J.F.sum()), r3=int(J.R3.sum()), flags=[dict(flag=k, n=int(v.sum()), units=rnd(J.F[v].sum() / J.F.sum(), 3)) for k, v in flags.items()],
                        exp_abs=rnd((J.exp_abs.fillna(J.exp_abs.mean()) * J.F).sum() / J.F.sum(), 3), by_age=[dict(g=g, n=int((J.age_band == g).sum()), units=rnd(J.F[J.age_band == g].sum() / J.F.sum(), 3)) for g in AGE],
                        by_dir=[dict(g=g, n=int((J.dir_band == g).sum()), units=rnd(J.F[J.dir_band == g].sum() / J.F.sum(), 3)) for g in pd.unique(dirb)])
J["nflag"] = sum(v.astype(int) for v in flags.values()); wl = J.sort_values(["nflag", "F"], ascending=False).head(18)
OUT["watch"] = [dict(item=r.style_color, name=str(it.product_name.get(r.style_color)), F=int(r.F), R3=int(r.R3), age=rnd(r.age_m, 1), spread=rnd(r.spread, 2), exp_abs=rnd(r.exp_abs, 2), flags=[k for k, v in flags.items() if bool(v.loc[r.Index])]) for r in wl.itertuples()]
print("\ntargets 2026:", OUT["targets26"]["flags"], "expected abs WAPE", OUT["targets26"]["exp_abs"])
json.dump(OUT, open("data/fk_err_analysis.json", "w"), indent=0); print("saved data/fk_err_analysis.json")
