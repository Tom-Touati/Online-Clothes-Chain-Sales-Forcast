"""D313 (fork; user in the session: "we need a metric that reflects all items, so we need to have a wape : R3 that sums the ratio rather than
taking the ratio of sums, or something similar right?"). Item-equal versions of WAPE / R3, each item counted once:
  rel_i      = (|F - y| + 1) / (|R3 - y| + 1)      the item's error relative to R3's (+1 unit so an exact R3 does not divide by zero)
  MdRAE      median of rel_i                       (median relative absolute error)
  GMRAE      exp(mean log rel_i)                   (geometric mean relative absolute error: a 2x better and a 2x worse item cancel)
  mean rel   mean of min(rel_i, 5)                 (the plain "sum of ratios", each item capped at 5x so a few items where R3 was
                                                    almost exact cannot decide it)
  share better   share of items with |F - y| < |R3 - y|
  item-equal WAPE ratio   mean_i(|F - y| / y) / mean_i(|R3 - y| / y), each capped at 200%   (shown in D289 as "mean % error ÷ R3")
next to the unit-weighted WAPE / R3 (ratio of sums). Recent origins, all 13, by origin, by volume fifth; for R3 + rules, the five members, their
plain average, the submitted system and the D307 'sig' variant (with the submitted corrections). Also the sensitivity of GMRAE to the +1 floor.
Adds J["itemrel"] to data/fk_err_analysis.json; stdout -> eda/item_metrics_output.txt."""
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); ORIG = sorted(D.origin.unique()); rec = (D.origin >= "2025-06").values; allm = np.ones(len(D), bool); y = D.y.values; R3 = D.R3.values
K = pd.MultiIndex.from_arrays([D.origin, D.style_color]); S = pd.read_parquet("data/fk_member_refit_preds.parquet").set_index(["origin", "style_color"]).reindex(K)
MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; X = D[MEM].values.astype(float); corr = (D.F / D.F0).values; sig = (X.sum(1) - D.qdyn.values + S["+ origin-known signals"].values) / 5 * corr
SYS = [("R3 + rules", D.rules.values), ("LightGBM member", D.qdyn.values), ("Neural net member", D.nn.values), ("TabPFN member (conformal)", D.pf2c.values), ("TabPFN member (32 estimators)", D.pf2d32.values),
       ("Plain average of the five", X.mean(1)), ("Submitted system", D.F.values), ("Submitted with the 11 signals (D307)", sig)]
def met(f, m, floor=1.0):
    ef, er = np.abs(f - y)[m], np.abs(R3 - y)[m]; rel = (ef + floor) / (er + floor); yy = np.maximum(y[m], 1)
    return dict(wape_r3=r(ef.sum() / er.sum()), mdrae=r(float(np.median(rel))), gmrae=r(float(np.exp(np.mean(np.log(rel))))), mean_rel=r(float(np.minimum(rel, 5).mean())), better=r(float((ef < er).mean()), 3),
                item_wape=r(float(np.minimum(ef / yy, 2).mean() / np.minimum(er / yy, 2).mean())), n=int(m.sum()))
rows = [dict(system=lab, **met(f, rec), all13=met(f, allm)) for lab, f in SYS]
print("recent origins:"); print(pd.DataFrame([{k: v for k, v in d.items() if k != "all13"} for d in rows]).to_string(index=False))
print("all 13:"); print(pd.DataFrame([dict(system=d["system"], **d["all13"]) for d in rows]).to_string(index=False))
F = D.F.values; by_o = [dict(origin=o, **met(F, (D.origin == o).values)) for o in ORIG]; print(pd.DataFrame(by_o).to_string(index=False))
D["fifth"] = D.groupby("origin")["level"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)); by_f = [dict(fifth=k + 1, med_level=r(float(np.median(D.level.values[rec & (D.fifth.values == k)])), 0), **met(F, rec & (D.fifth.values == k))) for k in range(5)]
print(pd.DataFrame(by_f).to_string(index=False))
sens = [dict(floor=fl, gmrae=met(F, rec, fl)["gmrae"], mdrae=met(F, rec, fl)["mdrae"], mean_rel=met(F, rec, fl)["mean_rel"]) for fl in (0.5, 1, 5, 20)]; print("floor sensitivity:", sens)
J = json.load(open("data/fk_err_analysis.json")); J["itemrel"] = dict(systems=rows, by_origin=by_o, by_fifth=by_f, floor=sens); json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
