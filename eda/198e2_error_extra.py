"""D276 (fork): additions to data/fk_err_analysis.json (run after eda/198e): members by season, a seasonal-naive baseline, level vs
allocation error, the deliverable's analogue window, effective sample size, young launches by distribution, learning-curve results."""
import json, numpy as np, pandas as pd
D = pd.read_parquet("data/fk_err_master.parquet"); J = json.load(open("data/fk_err_analysis.json")); ORIG = sorted(D.origin.unique()); REC = [o for o in ORIG if o >= "2025-06"]
D["ae"] = (D.F - D.y).abs(); D["ar"] = (D.R3 - D.y).abs(); R = D[D.origin.isin(REC)]; r = lambda x, k=4: None if not np.isfinite(x) else round(float(x), k)
w = lambda Z, c: (Z[c] - Z.y).abs().sum() / Z.ar.sum(); wa = lambda Z, c: (Z[c] - Z.y).abs().sum() / Z.y.sum()
sm = D.origin.str[-2:].astype(int); D["season"] = sm.map(lambda m: "spring" if m in (2, 3, 4) else "summer_autumn" if m in (5, 6) else "november" if m in (7, 8, 9, 10) else "holiday")
MEM = {"rules": "rule agents", "qdyn": "LightGBM (ec_ns)", "nn": "neural net", "pf2c": "TabPFN conformal", "pf2d32": "TabPFN dyn 32", "F0": "arithmetic stack", "F": "submitted (stack + corrections)"}
J["members"] = [dict(member=lab, recent=r(w(R, c)), all=r(w(D, c)), bias_recent=r(R[c].sum() / R.y.sum() - 1), **{s: r(w(D[D.season == s], c)) for s in ["spring", "summer_autumn", "november", "holiday"]},
                     **{"abs_" + s: r(wa(D[D.season == s], c)) for s in ["summer_autumn"]}, **{"bias_" + s: r(D[D.season == s][c].sum() / D[D.season == s].y.sum() - 1) for s in ["summer_autumn"]}) for c, lab in MEM.items()]
# seasonal naive: last 3 months' rate x last year's like-for-like type ratio for each target month (as eda/124 'sea_type')
F = pd.read_parquet("data/features.parquet", columns=["origin", "style_color", "h", "m3", "lfl_type"]); F["st"] = F.m3 * np.exp(F.lfl_type.fillna(0))
D = D.merge(F.groupby(["origin", "style_color"]).st.sum().rename("sea").reset_index(), on=["origin", "style_color"], how="left"); D["sea"] = D.sea.fillna(D.R3); R = D[D.origin.isin(REC)]
J["seasonal_naive"] = dict(recent_vs_r3=r(w(R, "sea")), all_vs_r3=r(w(D, "sea")), system_vs_seasonal_recent=r(R.ae.sum() / (R.sea - R.y).abs().sum()), system_vs_seasonal_all=r(D.ae.sum() / (D.sea - D.y).abs().sum()),
                           by_origin=[dict(origin=o, sea_abs=r(wa(D[D.origin == o], "sea"), 3), sys_vs_sea=r(D[D.origin == o].ae.sum() / (D[D.origin == o].sea - D[D.origin == o].y).abs().sum(), 3)) for o in ORIG])
# level vs allocation: the system rescaled so that each origin's total is right; and each origin x product-group total
D["F_tot"] = D.F * D.groupby("origin").y.transform("sum") / D.groupby("origin").F.transform("sum"); D["F_grp"] = D.F * D.groupby(["origin", "macro"]).y.transform("sum") / D.groupby(["origin", "macro"]).F.transform("sum")
R = D[D.origin.isin(REC)]; J["level_vs_alloc"] = dict(system=r(w(R, "F")), total_known=r(w(R, "F_tot")), group_known=r(w(R, "F_grp")), tot_err=[dict(origin=o, bias=r(D[D.origin == o].F.sum() / D[D.origin == o].y.sum() - 1, 3)) for o in ORIG])
# the deliverable's analogue: origins May-Jul 2025 (windows Jun-Sep, Jul-Oct, Aug-Nov)
A = D[D.origin.isin(["2025-05", "2025-06", "2025-07"])]; A6 = D[D.origin == "2025-06"]; cv = lambda Z: float(((Z.y >= Z.p10) & (Z.y <= Z.p90))[Z.p10.notna()].mean())
J["analogue"] = dict(o6=dict(ratio=r(w(A6, "F")), abs=r(wa(A6, "F")), r3abs=r(A6.ar.sum() / A6.y.sum()), bias=r(A6.F.sum() / A6.y.sum() - 1), cover=r(cv(A6), 3), n=int(len(A6)), tgt_bias=r(A6[A6.target].F.sum() / A6[A6.target].y.sum() - 1), within20=r(((A6.F - A6.y).abs() <= 0.2 * A6.y).mean(), 3)),
                     o567=dict(ratio=r(w(A, "F")), abs=r(wa(A, "F")), bias=r(A.F.sum() / A.y.sum() - 1), cover=r(cv(A), 3)))
lw = np.log((A6.p90 + 1) / (A6.p10 + 1)); mid = (np.log(A6.p90 + 1) + np.log(A6.p10 + 1)) / 2; ly = np.log(A6.y + 1)
for k in [1.0, 1.15, 1.3, 1.5, 1.75, 2.0]: J["analogue"].setdefault("widen", []).append(dict(k=k, cover=r(float((np.abs(ly - mid) <= k * lw / 2)[A6.p10.notna()].mean()), 3)))
lc = [d["corr"] for d in J["confounds"]["lag_corr"]]; J["neff"] = dict(all13=r(13 / (1 + 2 * sum(lc[:4])), 1), recent9=r(9 / (1 + 2 * sum(lc[:4])), 1))
Y = R[R.age_m < 3]; J["young_distribution"] = [dict(g=g, n=int(len(Z)), units=r(Z.y.sum() / Y.y.sum(), 3), bias=r(Z.F.sum() / Z.y.sum() - 1, 3), ratio=r(w(Z, "F"), 3), growth=r(Z.y.sum() / Z.R3.sum(), 2))
                           for g, Z in [("stores added: 5 or more", Y[Y.h_stores_chg >= 5]), ("stores added: 2 to 4", Y[(Y.h_stores_chg >= 2) & (Y.h_stores_chg < 5)]), ("stores about flat", Y[Y.h_stores_chg.abs() < 2]), ("stores lost: 2 or more", Y[Y.h_stores_chg <= -2]),
                                         ("partners added: 2 or more", Y[Y.h_partners_chg >= 2]), ("partners flat or fewer", Y[Y.h_partners_chg <= 0])]]
J["learning"] = json.load(open("data/fk_learning_curve.json")); J["learning"].pop("rows", None)
L = pd.DataFrame(json.load(open("data/fk_learning_curve.json"))["rows"]); J["learning"]["full"] = [dict(origin=o, ratio=r(v, 3)) for o, v in L[L.kind == "full"].set_index("origin").ratio.items()]
json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0)
print("members"); [print(m) for m in J["members"]]; print(J["seasonal_naive"]["recent_vs_r3"], J["seasonal_naive"]["system_vs_seasonal_recent"], J["seasonal_naive"]["system_vs_seasonal_all"], [(d["origin"], d["sys_vs_sea"]) for d in J["seasonal_naive"]["by_origin"]])
print(J["level_vs_alloc"]); print(J["analogue"]); print(J["neff"]); [print(y) for y in J["young_distribution"]]
