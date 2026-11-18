"""Fable 5.2 fork (user, 18 Nov: "the trees should receive fashion attributes as well. it should be a very main signal. how can we maximize
the usage of these features?"). The tree member of the submitted system (Huber on the log ratio, eda/187 `huber`; for items under 6 months
the version with new items' rows x5) refitted with the fashion information put back in, in increasing strength:
  base     the current feature sets (ec_ns: no static descriptors)                                              [reproduces the member]
  attr     + the static descriptors in BOTH averaged models: product type, 7 tagger attributes, 14 language-model scores
  attr_x   attr + explicit attribute x season interactions: each seasonal score (summer, winter, halloween, back_to_school, party,
           festival) x the share of the four target months in that season (static scores become time-varying, target-aligned inputs)
  attr_nc  attr + photo / text neighbours (D205 NCSEL: the 5 nearest items in the content embedding: newest turn, momentum, last-year ramp)
  attr_fa  attr + attribute-cohort trends (D205: mean last-year ramp, momentum, year-on-year of the item's attribute values, LLM-profile
           alignment, share within type)
  emb      attr + 16 principal components of the photo (CLIP) + text embeddings
  all      everything above
Each is fitted twice (new items' rows x1 and x5) and scored inside the submitted system (other members as stored, the routed nx5's
correction multiplier held) three ways: `all` = the variant for every item (x5 version under 6 months); `new` = only for items under
6 months; `old` = only for the others. WAPE / R3 on recent 9 / held out / all 13 / Jul-Oct 25 / Mar-Jun 26 and on new items, with a
style bootstrap against the current system. Gains (LightGBM) of the fashion features are printed for the last origin.
Usage: python eda/227_fashion_tree.py [LYDROP=1 in env for the last-year-removed gate] -> eda/fashion_tree_output.txt (stdout),
data/fashion_tree_preds.parquet"""
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
src = open("eda/187_losses.py", encoding="utf-8").read().split("for v in VARS:")[0]
sys.argv = ["x"]; exec(compile(src, "187", "exec"))
STATIC = ns["STATIC"]; B0, F0 = list(B_), list(F_)
print("statics already in BASE / FULL:", len([c for c in STATIC if c in B0]), len([c for c in STATIC if c in F0]), "| static columns in the table:", len([c for c in STATIC if c in T.columns]), "of", len(STATIC), flush=True)
# extra blocks
T = T.merge(pd.read_parquet("data/fk_nnc_feats.parquet")[["origin", "style_color", "nc_acc1", "nc_mom3", "nc_sea4"]], on=["origin", "style_color"], how="left")
FA = ["fa_mean_sea4", "fa_mean_mom", "fa_mean_yoy", "fa_llm_yoy", "fa_in_type"]
T = T.merge(pd.read_parquet("data/fk_fash_feats.parquet")[["origin", "style_color"] + FA], on=["origin", "style_color"], how="left")
CE = np.load("data/content_emb.npz", allow_pickle=True); X = np.hstack([CE["img"], CE["txt"]]); X = X - X.mean(0)
U, S_, _ = np.linalg.svd(X, full_matrices=False); EMB = [f"ce_{k}" for k in range(16)]
T = T.merge(pd.DataFrame(U[:, :16] * S_[:16], columns=EMB).assign(style_color=CE["items"].astype(str)), on="style_color", how="left")
SEAS = {"summer": [6, 7, 8], "winter": [12, 1, 2], "halloween": [10], "back_to_school": [8, 9], "party": [11, 12], "festival": [4, 5, 6]}
mo = (T.o_ord.values % 12) + 1   # calendar month of the origin (Period ordinal: months since 1970-01)
XI = []
for k, months in SEAS.items():
    share = np.mean([np.isin(((mo - 1 + h) % 12) + 1, months) for h in (1, 2, 3, 4)], axis=0); T[f"x_{k}"] = pd.to_numeric(T[k], errors="coerce") * share; XI.append(f"x_{k}")
ST = [c for c in STATIC if c in T.columns]
VAR = {"base": [], "attr": ST, "attr_x": ST + XI, "attr_nc": ST + ["nc_acc1", "nc_mom3", "nc_sea4"], "attr_fa": ST + FA, "emb": ST + EMB, "all": ST + XI + ["nc_acc1", "nc_mom3", "nc_sea4"] + FA + EMB}
LYD = os.environ.get("LYDROP") == "1"; P["n_jobs"] = int(os.environ.get("NJOBS", 3))
def run(extra, wnew, want_gain=False):
    cols_sets = [list(dict.fromkeys(B0 + extra)), list(dict.fromkeys(F0 + extra))]; out, gains = [], None
    for o in ORIG:
        oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()]; te = T[T.origin == o]
        if LYD: tr = tr[~tr.o_ord.between(oo - 13, oo - 11)]
        w = np.where(tr.age_m.fillna(99).values < 6, wnew, 1.0); zs = []
        for cols in cols_sets:
            m = lgb.LGBMRegressor(objective="huber", alpha=0.5, **P).fit(tr[cols], tr["z"], sample_weight=w); zs.append(m.predict(te[cols]))
            if want_gain and o == ORIG[-1] and cols is cols_sets[1]:
                g = pd.Series(m.booster_.feature_importance("gain"), index=cols); gains = g / g.sum()
        R = te[["origin", "style_color"]].copy(); R["q50"] = np.clip((4 * te.level.values + 1) * np.exp(np.mean(zs, axis=0)) - 1, 0, None); out.append(R)
    return pd.concat(out), gains
Bz = pd.read_parquet("data/dec_base_nx5r26.parquet"); Bz = Bz[Bz.y.notna()].reset_index(drop=True)
Gz = pd.read_parquet("data/geo_nx5r_x_preds.parquet")[["origin", "style_color", "F", "geo"]].rename(columns={"F": "Fg"}); Bz = Bz.merge(Gz, on=["origin", "style_color"])
corr = (Bz.geo / Bz.Fg.clip(lower=1e-9)).fillna(1).values; K = pd.MultiIndex.from_frame(Bz[["origin", "style_color"]])
new = Bz.age_m.fillna(99).values < 6; O = Bz.origin.values; y, R3 = Bz.y.values, Bz.R3.values; sty = Bz["style"].values; rng = np.random.default_rng(0)
def system(tree): return ((Bz[["rules", "nn", "pf2c", "pf2d32"]].clip(lower=0).sum(axis=1).values + np.maximum(tree, 0) + 0.25 * Bz.rc.clip(lower=0).values) / 5.25) * corr
wr = lambda f, m: np.abs(f - y)[m].sum() / np.abs(R3 - y)[m].sum()
def ci(a, b, m):
    d = pd.DataFrame(dict(s=sty[m], a=np.abs(a - y)[m], b=np.abs(b - y)[m], r=np.abs(R3 - y)[m])).groupby("s").sum()
    i = rng.integers(0, len(d), size=(400, len(d))); v = (d.a.values[i].sum(1) - d.b.values[i].sum(1)) / d.r.values[i].sum(1); return f"[{np.percentile(v, 2.5):+.4f},{np.percentile(v, 97.5):+.4f}]"
SETS = [("recent", O >= "2025-06"), ("held out", O >= "2025-10"), ("all 13", O > "0"), ("JulOct25", O == "2025-06"), ("MarJun26", O == "2026-02"), ("new rec", (O >= "2025-06") & new), ("new held", (O >= "2025-10") & new), ("old rec", (O >= "2025-06") & ~new)]
keep = {}; cur = None
for nm, extra in VAR.items():
    t0 = time.time(); q1, g = run(extra, 1.0, want_gain=True); q5, _ = run(extra, 5.0)
    a1 = q1.set_index(["origin", "style_color"]).q50.reindex(K).values; a5 = q5.set_index(["origin", "style_color"]).q50.reindex(K).values; keep[nm + "_w1"], keep[nm + "_w5"] = a1, a5
    routed = np.where(new, a5, a1)
    if nm == "base": cur_tree = routed; cur = system(routed); print("base reproduces the stored tree column:", float(np.nanmax(np.abs(routed / Bz.qdyn.values - 1))), flush=True)
    for mode, tree in [("all", routed), ("new", np.where(new, a5, cur_tree)), ("old", np.where(new, cur_tree, a1))]:
        f = system(tree); print(f"{nm:8s} {mode:3s} " + " ".join(f"{lab} {wr(f, m):.4f}" for lab, m in SETS) + f" | vs current: recent {ci(f, cur, SETS[0][1])} held out {ci(f, cur, SETS[1][1])}" + (" LYDROP" if LYD else ""), flush=True)
    if g is not None and extra:
        fs = g[[c for c in extra if c in g.index]]; print(f"   share of the FULL model's gain from the added columns: {fs.sum():.1%}; top: " + ", ".join(f"{c} {v:.1%}" for c, v in fs.sort_values(ascending=False).head(6).items()) + f" ({round(time.time() - t0)} s)", flush=True)
pd.DataFrame(keep).assign(origin=O, style_color=Bz.style_color.values).to_parquet("data/fashion_tree_preds" + ("_lyd" if LYD else "") + ".parquet", index=False)
