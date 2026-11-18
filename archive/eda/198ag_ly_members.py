"""D324 (fork; proposal #10a part 2, assigned in D314): the NN and TabPFN members refitted without last year's same window (o - 12 +/- 1) on the
GPU (eda/57_embed_nn.py via eda/197b_nn_ec.py, seeds 0-2; eda/196b_tabpfn.py v2 dyn with 32 and 8 estimators; LYREMOVE=1 in ~/edikted_fk_ly),
with the tree member of D315, scored alone and inside the system (the submitted correction multiplier on top):
  each member alone: full against without last year (the NN against its stored 3-seed run S13ec; TabPFN against the stored runs)
  the system with one member at a time without last year (the 6-seed NN and the conformal TabPFN moved by their run's ratio, without/with)
  the system with every learned member without last year at once (the rule agents unchanged)
Inputs copied back: data/fk_ly_nn_S13ecLY.parquet, data/fk_ly_pf_e32.parquet, data/fk_ly_pf_e8.parquet. Adds J["lymembers"]; stdout -> eda/ly_members_output.txt."""
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); K = pd.MultiIndex.from_arrays([D.origin, D.style_color]); rec = (D.origin >= "2025-06").values; allm = np.ones(len(D), bool)
y = D.y.values; ar = (D.R3 - D.y).abs().values; corr = (D.F / D.F0).values
q50 = lambda f: (lambda q: (q.groupby(["origin", "style_color"]).q50.sum() if "h" in q.columns and q.h.nunique() > 1 else q.set_index(["origin", "style_color"]).q50).reindex(K).values)(pd.read_parquet(f))
nn3, nn3ly = q50("data/nn_backtest_preds_S13ec.parquet"), q50("data/fk_ly_nn_S13ecLY.parquet")
e32, e32ly = D.pf2d32.values, q50("data/fk_ly_pf_e32.parquet"); e8, e8ly = q50("data/q_backtest_fktab_tabpfn_v2_dyn.parquet"), q50("data/fk_ly_pf_e8.parquet")
L = pd.read_parquet("data/fk_ly_removed_preds.parquet").set_index(["origin", "style_color"]).reindex(K); tree, treely = L["ec_ns | full"].values, L["ec_ns | no LY"].values
for nm, v in [("nn3ly", nn3ly), ("e32ly", e32ly), ("e8ly", e8ly), ("treely", treely)]: assert np.isfinite(v).all(), nm
ratio = lambda a, b: (a + 1) / (b + 1)
mem = dict(rules=D.rules.values, qdyn=D.qdyn.values, nn=D.nn.values, pf2c=D.pf2c.values, pf2d32=D.pf2d32.values)
ly = dict(qdyn=treely, nn=D.nn.values * ratio(nn3ly, nn3), pf2c=D.pf2c.values * ratio(e8ly, e8), pf2d32=e32ly)
wape = lambda f, m: float(np.abs(f - y)[m].sum() / ar[m].sum()); hit = lambda f, m: float((np.abs(f - y)[m] <= 0.2 * y[m]).mean()); gm = lambda f, m: float(np.exp(np.mean(np.log((np.abs(f - y)[m] + 1) / (ar[m] + 1)))))
sysf = lambda M: np.mean([M[k] for k in ["rules", "qdyn", "nn", "pf2c", "pf2d32"]], 0) * corr
alone = []
for lab, a, b in [("tree (ec_ns)", tree, treely), ("neural net (3 seeds)", nn3, nn3ly), ("TabPFN v2 dyn, 32 estimators", e32, e32ly), ("TabPFN v2 dyn, 8 estimators (base of the conformal member)", e8, e8ly)]:
    alone.append(dict(member=lab, full_rec=r(wape(a, rec)), ly_rec=r(wape(b, rec)), full_all=r(wape(a, allm)), ly_all=r(wape(b, allm)), deg_all=r(wape(b, allm) - wape(a, allm)), deg_rec=r(wape(b, rec) - wape(a, rec))))
A = pd.DataFrame(alone); print("member alone (WAPE / R3):"); print(A.to_string(index=False))
S0 = sysf(mem); rows = [dict(system="submitted system (all members with last year)", rec=r(wape(S0, rec)), all13=r(wape(S0, allm)), hit=r(hit(S0, rec), 3), gmrae=r(gm(S0, rec), 3))]
for k, lab in [("qdyn", "tree"), ("nn", "neural net"), ("pf2c", "conformal TabPFN"), ("pf2d32", "TabPFN 32")]:
    M = dict(mem); M[k] = ly[k]; f = sysf(M); rows.append(dict(system=f"{lab} without last year", rec=r(wape(f, rec)), all13=r(wape(f, allm)), hit=r(hit(f, rec), 3), gmrae=r(gm(f, rec), 3)))
M = dict(mem); M.update(ly); f = sysf(M); rows.append(dict(system="every learned member without last year", rec=r(wape(f, rec)), all13=r(wape(f, allm)), hit=r(hit(f, rec), 3), gmrae=r(gm(f, rec), 3)))
R = pd.DataFrame(rows); R["deg_all"] = (R.all13 - R.all13.iloc[0]).round(4); R["deg_rec"] = (R.rec - R.rec.iloc[0]).round(4); print("\nsystem (WAPE / R3, submitted correction multiplier):"); print(R.to_string(index=False))
print("check: the conformal TabPFN median equals its 8-estimator base:", bool(np.allclose(D.pf2c.values, e8, rtol=1e-3, atol=1)))
J = json.load(open("data/fk_err_analysis.json")); J["lymembers"] = dict(alone=alone, system=rows); json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
