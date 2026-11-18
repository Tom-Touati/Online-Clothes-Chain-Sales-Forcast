"""D278 (fork; user via Main: "the +/-20% hit rate is alarming; make it a central axis"): the share of item totals forecast within +/-20%
of the actual, taken apart. Hit = |forecast - actual| <= 0.2 x actual on the 4-month item total. Recent = the nine origins from June 2025.
  A tolerance curve (system, R3, seasonal naive) and the hit rate along the chain of adopted systems
  B miss bands (within 20, 20-30, 30-50, 50-100, beyond) by direction; what separates near misses from large ones
  C level or shape: hit rate with each origin's / group's / type x look cell's total known; the dispersion it implies
  D aggregation: hit rate of the same forecasts summed to style, look, type, group, origin
  E cohort extremes (from eda/198e) and a search for simple rules that hold in both halves of the backtest
  F a time-honest hit classifier (trained on earlier origins only): can a group with 60% be identified in advance?
  G what the plannable information and a month-1 read would do to the hit rate (from the eda/198e value-of-information fits)
  H the June 2026 targets under the classifier
Adds J["hit"] to data/fk_err_analysis.json; stdout -> eda/hit_rate_output.txt."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb, warnings, itertools
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
D = pd.read_parquet("data/fk_err_master.parquet"); J = json.load(open("data/fk_err_analysis.json")); J26 = pd.read_parquet("data/fk_err_master_2026.parquet")
ORIG = sorted(D.origin.unique()); REC = [o for o in ORIG if o >= "2025-06"]; D["rec"] = D.origin.isin(REC); D["oo"] = D.origin.map(lambda o: pd.Period(o, "M").ordinal)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
hit = lambda f, y, t=0.2: (np.abs(f - y) <= t * y)
D["hit"] = hit(D.F, D.y); R = D[D.rec]; H = {}
Fq = pd.read_parquet("data/features.parquet", columns=["origin", "style_color", "h", "m3", "lfl_type"]); Fq["st"] = Fq.m3 * np.exp(Fq.lfl_type.fillna(0))
D = D.merge(Fq.groupby(["origin", "style_color"]).st.sum().rename("sea").reset_index(), on=["origin", "style_color"], how="left"); D["sea"] = D.sea.fillna(D.R3); R = D[D.rec]

# ---- A tolerance curve and the chain
TOL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]
H["curve"] = [dict(tol=t, system=r(hit(R.F, R.y, t).mean()), r3=r(hit(R.R3, R.y, t).mean()), seasonal=r(hit(R.sea, R.y, t).mean()), units=r(R.y[hit(R.F, R.y, t)].sum() / R.y.sum())) for t in TOL]
CH = [("R3", "R3"), ("R3 x type seasonality", "sea"), ("R3 + rules", "rules"), ("first three-way", "s_rq"), ("+ weather, events (D161)", "s_wx3bf"), ("+ co-purchase (D203)", "s_dyn3bf"), ("+ TabPFN v2 (D204)", "s_p2"),
      ("arithmetic stack (D207)", "s_p2c32a"), ("new tree member (D253)", "s_ecns"), ("new NN inputs (D272)", "F0"), ("+ corrections = submitted", "F")]
H["chain"] = [dict(system=lab, hit=r(hit(R[c], R.y).mean()), hit_all=r(hit(D[c], D.y).mean()), wape=r((R[c] - R.y).abs().sum() / (R.R3 - R.y).abs().sum())) for lab, c in CH]
H["overall"] = dict(items=r(R.hit.mean()), units=r(R.y[R.hit].sum() / R.y.sum()), targets=r(R[R.target].hit.mean()), all13=r(D.hit.mean()), n=int(len(R)),
                    by_origin=[dict(origin=o, hit=r(D[D.origin == o].hit.mean(), 3), r3=r(hit(D[D.origin == o].R3, D[D.origin == o].y).mean(), 3)) for o in ORIG])
print("A overall", {k: v for k, v in H["overall"].items() if k != "by_origin"}); print("  chain", [(d["system"], d["hit"]) for d in H["chain"]])

# ---- B miss bands and what separates them
ape = (R.F - R.y).abs() / R.y; BANDS = [("within 20%", 0, .2), ("20-30% (near miss)", .2, .3), ("30-50%", .3, .5), ("50-100%", .5, 1.0), ("beyond 100%", 1.0, 1e9)]
mem = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; R = R.assign(ape=ape, iw=np.log((R.p90 + 1) / (R.p10 + 1)), lf=np.log((R.F + 1) / (R.R3 + 1)))
R["band"] = pd.cut(R.ape, [-1, .2, .3, .5, 1.0, 1e9], labels=[b[0] for b in BANDS]).astype(str)
bb = []
for lab, lo, hi_ in BANDS:
    Z = R[R.band == lab]
    bb.append(dict(band=lab, share=r(len(Z) / len(R)), units=r(Z.y.sum() / R.y.sum()), err=r((Z.F - Z.y).abs().sum() / (R.F - R.y).abs().sum()), under=r((Z.F < Z.y).mean()),
                   young=r((Z.age_m < 3).mean()), med_level=r(Z.level.median(), 0), spread=r(Z.spread.median(), 3), iw=r(Z.iw.median(), 3), far_above=r((Z.lf > 0.35).mean()), far_below=r((Z.lf < -0.35).mean()),
                   near_r3=r((Z.lf.abs() <= 0.1).mean()), bf=r(Z.bf_window.mean()), stores2=r((Z.h_stores_chg >= 2).mean()), m1_off=r(((Z.h_m1 >= 1.5 * Z.F / 4) | (Z.h_m1 <= 0.67 * Z.F / 4)).mean()),
                   all_same=r(((Z[mem].values < Z.y.values[:, None]).all(1) | (Z[mem].values > Z.y.values[:, None]).all(1)).mean()), inside=r(((Z.y >= Z.p10) & (Z.y <= Z.p90))[Z.p10.notna()].mean())))
H["bands"] = bb; miss = R[R.ape > .2]; H["miss_split"] = dict(near=r((miss.ape <= .3).mean()), mid=r(((miss.ape > .3) & (miss.ape <= .5)).mean()), large=r((miss.ape > .5).mean()), within30=r((R.ape <= .3).mean()), within25=r((R.ape <= .25).mean()))
print("B bands"); print(pd.DataFrame(bb).to_string(index=False)); print("  of the misses:", H["miss_split"])

# ---- C level or shape
def rescaled(keys):
    g = R.groupby(keys); f = R.F * g.y.transform("sum") / g.F.transform("sum"); return r(hit(f, R.y).mean()), r(R.y[hit(f, R.y)].sum() / R.y.sum())
le = np.log((R.F + 1) / (R.y + 1)); med = le.groupby(R.origin).transform("median"); sig = float(1.4826 * (le - med).abs().median()); sd = float((le - med).std())
ph = lambda s: float(norm.cdf(np.log(1.2) / s) - norm.cdf(np.log(0.8) / s)); need = lambda p: float(next(s for s in np.arange(0.60, 0.05, -0.001) if ph(s) >= p))
best = np.abs(R[mem].values - R.y.values[:, None]).min(1)
H["level_shape"] = dict(now=r(R.hit.mean()), origin_total=rescaled(["origin"])[0], group_total=rescaled(["origin", "macro"])[0], type_total=rescaled(["origin", "ptype"])[0], cell_total=rescaled(["origin", "ptype", "look"])[0],
                        style_total=rescaled(["origin", "style"])[0], best_member=r((best <= 0.2 * R.y).mean()), robust_sigma=r(sig, 3), sd=r(sd, 3), normal_hit_at_sigma=r(ph(sig), 3), sigma_for_50=r(need(0.5), 3), sigma_for_60=r(need(0.6), 3),
                        sigma_for_70=r(need(0.7), 3), median_ape=r(R.ape.median(), 3))
print("C level/shape", H["level_shape"])

# ---- D aggregation levels
def agg(keys):
    g = R.groupby(keys)[["F", "y", "R3"]].sum(); g = g[g.y > 0]; return dict(n=int(len(g)), hit=r(hit(g.F, g.y).mean(), 3), r3=r(hit(g.R3, g.y).mean(), 3), med_units=r(g.y.median(), 0), within10=r(hit(g.F, g.y, 0.1).mean(), 3))
H["levels"] = [dict(level=lab, **agg(k)) for lab, k in [("item (style-colour)", ["origin", "style_color"]), ("style (all colours)", ["origin", "style"]), ("look (cart + content cluster)", ["origin", "look"]),
                                                       ("product type", ["origin", "ptype"]), ("product group", ["origin", "macro"]), ("all items at an origin", ["origin"])]]
print("D levels"); print(pd.DataFrame(H["levels"]).to_string(index=False))

# ---- E cohort extremes and stable simple rules
G = [dict(axis=c["label"], key=k, g=g["g"], n=g["n"], hit=g["hit"], units=g["unit_share"], bias=g["bias"]) for k, c in J["cohorts"].items() for g in c["groups"] if g["n"] >= 250 and g["g"] != "n/a"]
G = sorted(G, key=lambda d: d["hit"]); H["cohort_low"] = G[:10]; H["cohort_high"] = G[::-1][:10]
D["vq"] = D.groupby("origin")["level"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)); D["lf"] = np.log((D.F + 1) / (D.R3 + 1)); D["iw"] = np.log((D.p90 + 1) / (D.p10 + 1))
D["iwq"] = D.groupby("origin")["iw"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)); D["spq"] = D.groupby("origin")["spread"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
sm = D.origin.str[-2:].astype(int)
COND = {"top two volume fifths": D.vq >= 3, "top volume fifth": D.vq == 4, "age 3 months or more": D.age_m >= 3, "age 12 months or more": D.age_m >= 12, "age 3 to 6 months": (D.age_m >= 3) & (D.age_m < 6),
        "narrowest two interval fifths": D.iwq <= 1, "narrowest interval fifth": D.iwq == 0, "calmest two model-spread fifths": D.spq <= 1, "calmest model-spread fifth": D.spq == 0,
        "forecast far below R3": D.lf < -0.35, "forecast within 10% of R3": D.lf.abs() <= 0.1, "forecast above R3": D.lf > 0.1, "flat 91-day trend": D.trend91.abs() <= 0.15, "rising trend": D.trend91 > 0.15,
        "in 8+ own stores": D.n_stores >= 8, "with 5+ partners": D.n_partners >= 5, "4+ live colours in the style": D.n_siblings >= 4, "no November in the window": D.bf_window == 0,
        "web discount 55% or more": D.disc_web >= 0.55, "warm items": D.macro == "warm", "sold in the same months last year": D.has_ly == 1, "largest day under 9% of the month": D.top_day_share < 0.09,
        "TikTok under 5%": D.sh_tiktok < 0.05}
h1 = D.origin.isin(ORIG[:7]).values; names = list(COND); rules = []
for kk in [1, 2, 3]:
    for cs in itertools.combinations(names, kk):
        m = np.logical_and.reduce([COND[c].values for c in cs]); a, b = m & h1, m & ~h1
        if a.sum() < 0.03 * h1.sum() or b.sum() < 0.03 * (~h1).sum(): continue
        ha, hb = D.hit.values[a].mean(), D.hit.values[b].mean(); mr = m & D.rec.values
        rules.append(dict(rule=" + ".join(cs), k=kk, first_half=r(ha, 3), second_half=r(hb, 3), recent=r(D.hit.values[mr].mean(), 3), cover=r(mr.sum() / D.rec.sum(), 3), units=r(D.y.values[mr].sum() / D.y.values[D.rec.values].sum(), 3), worst=min(ha, hb)))
rules = sorted(rules, key=lambda d: -d["worst"]); H["rules"] = [{k: v for k, v in d.items() if k != "worst"} for d in rules[:12]]; H["rules_n"] = len(rules)
H["rules_over55"] = int(sum(d["worst"] >= 0.55 for d in rules)); H["rules_over60"] = int(sum(d["worst"] >= 0.60 for d in rules)); H["rules_over50"] = int(sum(d["worst"] >= 0.50 for d in rules))
print("E lowest cohorts", [(d["g"], d["hit"]) for d in H["cohort_low"][:6]]); print("  highest", [(d["axis"][:20], d["g"], d["hit"]) for d in H["cohort_high"][:8]])
print("  rules tested", len(rules), "| worst-half >= 0.50:", H["rules_over50"], ">= 0.55:", H["rules_over55"], ">= 0.60:", H["rules_over60"]); print(pd.DataFrame(H["rules"]).to_string(index=False))

# ---- F time-honest hit classifier
D["llv"] = np.log1p(D.level); D["abs_lf"] = D.lf.abs(); D["macro_c"] = D.macro.astype("category").cat.codes; D["mcorr"] = np.log(D.F / D.F0); D["month"] = sm; D["mrange"] = (D[mem].max(1) - D[mem].min(1)) / D.F
FE = ["llv", "age_m", "lf", "abs_lf", "spread", "iw", "trend91", "f_r28", "disc_web", "disc_mom", "n_stores", "n_partners", "size_run", "sh_tiktok", "sh_retail", "sh_partner", "n_siblings", "share_of_style", "top_day_share",
      "top_hour_share", "oos_28", "hist_months", "has_ly", "bf_window", "macro_c", "mcorr", "vq", "month", "mrange"]
PC = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=6)
D["p_hit"] = np.nan
for o in REC:
    oo = pd.Period(o).ordinal; tr = D[D.oo + 4 <= oo]
    if tr.hit.nunique() < 2: continue
    D.loc[D.origin == o, "p_hit"] = lgb.LGBMClassifier(**PC).fit(tr[FE], tr.hit.astype(int)).predict_proba(D.loc[D.origin == o, FE])[:, 1]
Z = D[D.rec & D.p_hit.notna()].copy(); Z["dec"] = Z.groupby("origin").p_hit.transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
H["clf"] = dict(auc=r(roc_auc_score(Z.hit, Z.p_hit), 3), auc_by_origin=[dict(origin=o, auc=r(roc_auc_score(g.hit, g.p_hit), 3)) for o, g in Z.groupby("origin")],
                deciles=[dict(decile=int(d) + 1, hit=r(g.hit.mean(), 3), p=r(g.p_hit.mean(), 3), units=r(g.y.sum() / Z.y.sum(), 3)) for d, g in Z.groupby("dec")],
                bins=[dict(bin=lab, share=r(((Z.p_hit >= lo) & (Z.p_hit < hi_)).mean(), 3), hit=r(Z[(Z.p_hit >= lo) & (Z.p_hit < hi_)].hit.mean(), 3) if ((Z.p_hit >= lo) & (Z.p_hit < hi_)).sum() >= 30 else None,
                           p=r(Z[(Z.p_hit >= lo) & (Z.p_hit < hi_)].p_hit.mean(), 3) if ((Z.p_hit >= lo) & (Z.p_hit < hi_)).sum() >= 30 else None)
                      for lab, lo, hi_ in [("under 0.30", 0, .3), ("0.30-0.40", .3, .4), ("0.40-0.50", .4, .5), ("0.50-0.60", .5, .6), ("0.60 or more", .6, 1.01)]],
                top10=r(Z[Z.dec == 9].hit.mean(), 3), top20=r(Z[Z.dec >= 8].hit.mean(), 3), bottom20=r(Z[Z.dec <= 1].hit.mean(), 3),
                top10_by_origin=[dict(origin=o, hit=r(g[g.dec == 9].hit.mean(), 3), all=r(g.hit.mean(), 3)) for o, g in Z.groupby("origin")])
imp = lgb.LGBMClassifier(**PC).fit(D[FE], D.hit.astype(int)); gi = pd.Series(imp.booster_.feature_importance("gain"), index=FE).sort_values(ascending=False); H["clf"]["importance"] = [dict(f=k, share=r(v / gi.sum(), 3)) for k, v in gi.head(10).items()]
print("F classifier AUC", H["clf"]["auc"], "| top decile", H["clf"]["top10"], "top fifth", H["clf"]["top20"], "bottom fifth", H["clf"]["bottom20"]); print("  bins", H["clf"]["bins"]); print("  deciles", [(d["decile"], d["hit"]) for d in H["clf"]["deciles"]])
print("  top decile by origin", [(d["origin"], d["hit"], d["all"]) for d in H["clf"]["top10_by_origin"]]); print("  importance", H["clf"]["importance"][:8])

# ---- G information and the hit rate
VP = pd.read_parquet("data/fk_err_voi_preds.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([D.origin, D.style_color]))
for c in ["f_ref", "f_plan", "f_all"]: D[c] = VP[c].values
R = D[D.rec]; m1 = R.F * (R.h_m1 / (R.F / 4)).clip(0.25, 4) ** 0.5   # a naive early-read rescale, for comparison with the fitted one
H["info"] = dict(system=r(R.hit.mean()), ref=r(hit(R.f_ref, R.y).mean()), plan=r(hit(R.f_plan, R.y).mean()), plan_early=r(hit(R.f_all, R.y).mean()), blocks=[dict(block=d["block"], hit=d["hit"], wape=d["wape"]) for d in J["voi"]],
                 type_look_known=H["level_shape"]["cell_total"], best_member=H["level_shape"]["best_member"])
print("G info", {k: v for k, v in H["info"].items() if k != "blocks"})

# ---- H June 2026 targets
TG = set(pd.read_csv("data/targets.csv").style_color); q26 = pd.read_parquet("data/fk_2026_pf2c.parquet").set_index("style_color")
J26["lf"] = np.log((J26.F + 1) / (J26.R3 + 1)); J26["abs_lf"] = J26.lf.abs(); J26["llv"] = np.log1p(J26.level); J26["iw"] = np.log((q26.q90.reindex(J26.style_color).values + 1) / (q26.q10.reindex(J26.style_color).values + 1))
J26["macro_c"] = pd.Categorical(J26.macro, categories=D.macro.astype("category").cat.categories).codes; J26["mcorr"] = np.log(J26.F / J26.F0); J26["month"] = 6; J26["mrange"] = (J26[mem].max(1) - J26[mem].min(1)) / J26.F
J26["vq"] = pd.qcut(J26.level.rank(method="first"), 5, labels=False)
for c in FE:
    if c not in J26: J26[c] = np.nan
J26["p_hit"] = imp.predict_proba(J26[FE])[:, 1]; T26 = J26[J26.style_color.isin(TG)]
cz = pd.read_csv("data/final_2026_nx2_cascade.csv", index_col=0); T26 = T26.assign(conf=cz.confidence.reindex(T26.style_color).values)
H["targets26"] = dict(n=int(len(T26)), mean_p=r(T26.p_hit.mean(), 3), unit_p=r((T26.p_hit * T26.F).sum() / T26.F.sum(), 3), ge50=int((T26.p_hit >= 0.5).sum()), ge60=int((T26.p_hit >= 0.6).sum()), lt35=int((T26.p_hit < 0.35).sum()),
                      by_conf=[dict(conf=int(c), n=int(len(g)), p=r(g.p_hit.mean(), 3)) for c, g in T26.groupby("conf")], hist=[dict(lo=r(lo, 2), n=int(((T26.p_hit >= lo) & (T26.p_hit < lo + 0.05)).sum())) for lo in np.arange(0.2, 0.7, 0.05)])
print("H targets 2026", H["targets26"])
# ---- I two reference points: counting noise, and the horizon for a naive forecast
from scipy.stats import poisson
R = D[D.rec]; mu = R.y.values; pf = poisson.cdf(np.floor(mu / 0.8), mu) - poisson.cdf(np.ceil(mu / 1.2) - 1, mu)
H["floor"] = dict(overall=r(float(pf.mean()), 3), by_fifth=[dict(fifth=int(q) + 1, poisson=r(float(pf[(R.vq == q).values].mean()), 3), hit=r(R[R.vq == q].hit.mean(), 3), med=r(R[R.vq == q].y.median(), 0)) for q in range(5)])
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
nh = []
for k in [1, 2, 3, 4]:
    num = den = 0
    for o, g in R.groupby("origin"):
        act = M.reindex(g.style_color).reindex(columns=[str(pd.Period(o, "M") + j) for j in range(1, k + 1)]).fillna(0).values.sum(1); f = g.R3.values / 4 * k
        ok = act > 0; num += (np.abs(f - act)[ok] <= 0.2 * act[ok]).sum(); den += ok.sum()
    nh.append(dict(months=k, hit=r(num / den, 3)))
H["naive_horizon"] = nh; print("I counting-noise ceiling", H["floor"]); print("  naive recency by horizon", nh)
J["hit"] = H; json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
