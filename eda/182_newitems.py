"""D263 (user, 18 Nov: "a separate model for new items, that imputes missing fields in a learned way and predicts the probability of a
burst at every month. maybe poisson or something else"). New items = age < 6 months at the origin; models trained on items < 9 months.
Monthly rows (origin, item, h = 1..4) from data/features.parquet (+ LLM scores, cart / dynamic neighbours, temperature preference,
D252 launch-curve and age blocks, events of the target month itself (data/events_month + events_month2), 16 principal components of the
product photo + text embedding (CLIP; static)).
LEARNED IMPUTATION (at each origin o, from rows at origins <= o, where the field is observed; features = descriptors, content PCs,
cart / dynamic neighbours, type seasonality, target month): own_sea (the item's last-year ratio for the target month; missing for every
item under a year old) and trend91 (missing under 6 months). Imputed values + missingness flags enter the models.
MODELS at each test origin o (training rows: origins o' + 4 <= o, items < 9 months), base_h = level x exp(lfl_type_h) (the type's
seasonal step):
  tw     LightGBM Tweedie (variance power 1.3, a compound Poisson-gamma count model) of monthly units with offset log(base_h + 1)
  burst  P(burst_h), burst_h = units >= 2.5 x base_h and >= 30 (LightGBM binary); size = log((y + 1) / (base + 1)) on burst rows (L2);
         non-burst count = Tweedie on non-burst rows; mix = (1 - p) x nonburst + p x burst size (the mixture mean)
  q4     4-month total, quantile 0.5 + L2 on z = log((total + 1) / (4 level + 1)), young rows only, imputed fields (a separate
         'new items' version of the tree member)
Writes data/newitems_preds.parquet (origin, style_color, h, p_burst, tw, mix) and data/newitems_tot.parquet (4-month totals)."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb, warnings
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
F = pd.read_parquet("data/features.parquet")
X0 = pd.read_parquet("data/fk_explain_table.parquet"); META = json.load(open("data/fk_explain_meta.json"))
LLM = ["warmth", "coverage", "formality", "fit_tightness", "statement", "staple", "summer", "winter", "back_to_school", "party", "halloween", "festival", "layering", "versatility"]
st = X0.drop_duplicates("style_color").set_index("style_color")[LLM]
F = F.merge(st, left_on="style_color", right_index=True, how="left")
F = F.merge(X0[["origin", "style_color", "cp_nb_sea4", "cp_nb_lvl", "beta_T", "dT_norm", "occ_next", "occ_last"]], on=["origin", "style_color"], how="left")
DY = ["dy_nb_sea4", "dy_nb_lvl", "dy_nb_mom3", "dy_nb_yoy"]; F = F.merge(pd.read_parquet("data/fk_dyn_feats.parquet")[["origin", "style_color"] + DY], on=["origin", "style_color"], how="left")
NF = pd.read_parquet("data/nf2_feats.parquet"); CUR = ["curve_fwd", "curve_dev", "first30", "best_bucket", "anniv_s", "anniv_c", "launch_doy_s", "launch_doy_c", "pk_life", "pk_28", "wk_end_rel"]
F = F.merge(NF[["origin", "style_color"] + CUR], on=["origin", "style_color"], how="left")
E1 = pd.read_parquet("data/events_month.parquet").set_index("m"); E2 = pd.read_parquet("data/events_month2.parquet").set_index("m").drop(columns=["c_pay"])
EV = pd.concat([E1, E2], axis=1).add_prefix("m_"); F["tm"] = [str(pd.Period(o, "M") + h) for o, h in zip(F.origin, F.h)]
F = F.merge(EV, left_on="tm", right_index=True, how="left"); EVC = list(EV.columns)
CE = np.load("data/content_emb.npz", allow_pickle=True); Z = np.hstack([CE["img"], CE["txt"]]).astype(np.float32)
pcs = PCA(16, random_state=0).fit_transform((Z - Z.mean(0)) / (Z.std(0) + 1e-6)); PC = [f"pc{k}" for k in range(16)]
F = F.merge(pd.DataFrame(pcs, columns=PC).assign(style_color=CE["items"].astype(str)), on="style_color", how="left")
for c in ["ptype", "macro"] + [c for c in F.columns if c.startswith("a_")]: F[c] = F[c].astype("category")
F["base"] = F.level * np.exp(F.lfl_type.fillna(0)); F["burst"] = ((F.y >= 2.5 * F.base) & (F.y >= 30)).astype(int)
DESC = ["ptype", "macro", "color_num"] + [c for c in F.columns if c.startswith("a_")] + LLM + PC + ["cp_nb_sea4", "cp_nb_lvl"] + DY + ["lfl_type", "lfl_macro", "lfl_occ", "tmonth", "h"]
P = dict(n_estimators=300, learning_rate=0.04, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=8)
IMP = {"own_sea": DESC, "trend91": [c for c in DESC if c not in ("h", "tmonth")] + ["age_m", "lagr1", "lagr2", "f_r28", "curve_fwd", "curve_dev", "first30"]}
FEAT = (["lvl", "age_m", "days_live", "lagr1", "lagr2", "lagr3", "lagr4", "lagr5", "lagr6", "f_r7", "f_r28", "peak_r", "sh_ret", "sh_drop", "sh_tt", "disc_item", "disc_rel",
         "beta_T", "dT_norm", "occ_next", "occ_last"] + DESC + CUR + EVC + ["own_sea_i", "trend91_i", "own_sea_miss", "trend91_miss"])
ORIG = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
rows, tots = [], []
for o in ORIG:
    oo = pd.Period(o).ordinal; D = F[F.o_ord <= oo].copy()
    for c, cols in IMP.items():
        obs = D[c].notna(); D[c + "_miss"] = (~obs).astype(int)
        if obs.sum() < 200: D[c + "_i"] = D[c]; continue   # own_sea needs 14 months of history: none observed before 2025-03
        m = lgb.LGBMRegressor(**P).fit(D.loc[obs, cols], D.loc[obs, c]); D[c + "_i"] = np.where(obs, D[c], m.predict(D[cols]))
    tr = D[(D.o_ord + 4 <= oo) & D.y.notna() & (D.age_m < 9)]; te = D[(D.origin == o) & (D.age_m < 6)]
    assert (tr.o_ord + 4 <= oo).all() and len(te)
    off_tr, off_te = np.log(tr.base + 1), np.log(te.base + 1)
    tw = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, **P).fit(tr[FEAT], tr.y, init_score=off_tr)
    mu = np.exp(tw.predict(te[FEAT], raw_score=True) + off_te)
    pb = lgb.LGBMClassifier(**P).fit(tr[FEAT], tr.burst).predict_proba(te[FEAT])[:, 1]
    bt = tr[tr.burst == 1]; sz = lgb.LGBMRegressor(**{**P, "min_child_samples": 20}).fit(bt[FEAT], np.log((bt.y + 1) / (bt.base + 1))).predict(te[FEAT])
    nbt = tr[tr.burst == 0]; nb = np.exp(lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, **P).fit(nbt[FEAT], nbt.y, init_score=np.log(nbt.base + 1)).predict(te[FEAT], raw_score=True) + off_te)
    mix = (1 - pb) * nb + pb * ((te.base.values + 1) * np.exp(sz) - 1)
    rows.append(pd.DataFrame({"origin": o, "style_color": te.style_color.values, "h": te.h.values, "y": te.y.values, "base": te.base.values, "burst": te.burst.values, "p_burst": pb, "tw": mu, "mix": mix, "nonburst": nb}))
    # q4: separate 4-month total model for new items (one row per item: h = 1 row carries the window-averaged imputed own_sea)
    g = D.groupby(["origin", "style_color"]).own_sea_i.mean().rename("own_sea4_i").reset_index(); D1 = D[D.h == 1].merge(g, on=["origin", "style_color"])
    D1["z"] = np.log((D1.ytot + 1) / (4 * D1.level + 1)); F4 = [c for c in FEAT if c not in ("h", "tmonth", "own_sea_i") and not c.startswith("m_")] + ["own_sea4_i"]
    t1 = D1[(D1.o_ord + 4 <= oo) & D1.ytot.notna() & (D1.age_m < 9)]; e1 = D1[(D1.origin == o) & (D1.age_m < 6)]
    zq = np.mean([lgb.LGBMRegressor(objective=ob, **({"alpha": 0.5} if ob == "quantile" else {}), **P).fit(t1[F4], t1.z).predict(e1[F4]) for ob in ("quantile", "l2")], axis=0)
    tots.append(pd.DataFrame({"origin": o, "style_color": e1.style_color.values, "q4": np.clip((4 * e1.level.values + 1) * np.exp(zq) - 1, 0, None)}))
    print(o, "train rows", len(tr), "test young items", te.style_color.nunique(), "| burst rate train", round(tr.burst.mean(), 3), flush=True)
R = pd.concat(rows); R.to_parquet("data/newitems_preds.parquet", index=False)
TT = R.groupby(["origin", "style_color"])[["tw", "mix", "nonburst"]].sum().reset_index().merge(pd.concat(tots), on=["origin", "style_color"], how="outer")
TT["p_any"] = 1 - R.assign(q=1 - R.p_burst).groupby(["origin", "style_color"]).q.prod().reindex(pd.MultiIndex.from_frame(TT[["origin", "style_color"]])).values
TT.to_parquet("data/newitems_tot.parquet", index=False); print(TT.describe().round(2).T)
