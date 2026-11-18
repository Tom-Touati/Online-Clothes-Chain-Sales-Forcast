"""Fork "fashion" (D403; user, 18 Nov: "run a version with recent last month weekly momentum (for each momentum feature the 25th percentile
and 75th percentile of momentum across all regions)"). The tree member of the submitted system (Huber on the log ratio; for items under
6 months the version with new items' rows x5) refitted with location-level weekly momentum (eda/229a_loc_weekly_mom.py ->
data/loc_mom_feats.parquet: per item, the 25th and 75th percentile across its locations of last-4-weeks vs the 4 before, last-2 vs the
2 before, and the last week vs the 4-week pace):
  base   current features                         [reproduces the stored member]
  lm     + the six percentiles over all 31 locations (web regions, TikTok, own stores, partners)
  lr     + the six percentiles over the five web regions only
  lmr    both
Each used for all items, new items only (< 6 months) or established items only, scored inside the submitted system (other members as
stored, the correction multiplier of the routed nx5 held), as eda/227. Env SEED, LYDROP, NJOBS, SRC187 as eda/227.
stdout -> eda/loc_mom_tree_output.txt; data/loc_mom_tree_preds[_lyd][_sN].parquet"""

import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
src = open(os.environ.get("SRC187", "eda/187_losses.py"), encoding="utf-8").read().split("for v in VARS:")[0]
sys.argv = ["x"]; exec(compile(src, "187", "exec"))
STATIC = ns["STATIC"]; B0, F0 = list(B_), list(F_)
print("statics already in BASE / FULL:", len([c for c in STATIC if c in B0]), len([c for c in STATIC if c in F0]), "| static columns in the table:", len([c for c in STATIC if c in T.columns]), "of", len(STATIC), flush=True)
LM = [f"lm_{m}_p{q}" for m in ("mom4", "mom2", "mom1") for q in (25, 75)]; LR = [c.replace("lm_", "lr_") for c in LM]
T = T.merge(pd.read_parquet("data/loc_mom_feats.parquet")[["origin", "style_color"] + LM + LR], on=["origin", "style_color"], how="left")
assert not set(LM + LR) & set(B0 + F0), "an added column name is already a feature"
print("coverage in the training table:", round(T[LM[0]].notna().mean(), 3), round(T[LR[0]].notna().mean(), 3), flush=True)
VAR = {"base": [], "lm": LM, "lr": LR, "lmr": LM + LR}
LYD = os.environ.get("LYDROP") == "1"; P["n_jobs"] = int(os.environ.get("NJOBS", 3))
SEED = int(os.environ.get("SEED", 0)); P["random_state"] = SEED   # SEED: bagging / column-sampling seed, to separate a real gain from resampling noise
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
pd.DataFrame(keep).assign(origin=O, style_color=Bz.style_color.values).to_parquet("data/loc_mom_tree_preds" + ("_lyd" if LYD else "") + (f"_s{SEED}" if SEED else "") + ".parquet", index=False)
