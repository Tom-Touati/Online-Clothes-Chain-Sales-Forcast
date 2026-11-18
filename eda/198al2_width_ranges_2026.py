"""D370b (fork; Fork 2's range attempt 1): June 2026 P25 / P75 ratios from the D370 width model (median loss, one k per side), as an option
next to the adopted per-segment ratios (fork D329, data/fk_ranges_2026.csv). P10 / P90 stay on the adopted rule (the width model is not
better there on recent origins, D373b).
  model   LightGBM, L1 on |log((y + 1) / (P50 + 1))|, the D370 inputs and settings, fitted on all 13 backtest origins (candidate system P50)
  k       each side's factor fitted on purged out-of-fold widths (each origin's width from a model trained on origins 4+ months away), 25% of
          actuals below P25 and 25% above P75 over all 13 origins
  2026    inputs at the June 2026 origin: data/fk_err_master_2026.parquet, Fork 2's temporal features, the members' 2026 quantiles (pf2c / pf2d32
          data/fk_member_quantiles.parquet; tree data/q_backtest_ls_q25 / q75_2026.parquet; recommender data/rc_recsys_preds_q25 / q75.parquet,
          f_honest_deepfm), P50 of data/system_q2575_2026.parquet (nx5)
Apply as P = (P50 + 1) x r - 1. Writes data/fk_ranges_2026_wm.csv ('#' header lines); stdout -> eda/width_ranges_2026_output.txt."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "eda"); import fk_rangekit as K
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
key = ["origin", "style_color"]; TF = pd.read_parquet("data/rc_temporal_feats.parquet")
B = K.load().merge(TF, on=key, how="left", validate="1:1")
T = pd.read_parquet("data/fk_err_master_2026.parquet"); T = T[T.origin == "2026-06"].merge(TF, on=key, how="left", validate="1:1")
S26 = pd.read_parquet("data/system_q2575_2026.parquet"); T = T.merge(S26[["style_color", "P50"]], on="style_color", validate="1:1")
MQ = pd.read_parquet("data/fk_member_quantiles.parquet"); MQ = MQ[MQ.origin == "2026-06"]
for m in ["pf2c", "pf2d32"]:
    q = MQ[MQ.member == m].set_index("style_color").reindex(T.style_color)
    for c in ["q25", "q50", "q75"]: T[f"{m}_{c}"] = q[c].values
for c in ["q25", "q75"]:
    T[f"tree_{c}"] = pd.read_parquet(f"data/q_backtest_ls_{c}_2026.parquet").set_index("style_color").q50.reindex(T.style_color).values
    rc = pd.read_parquet(f"data/rc_recsys_preds_{c}.parquet"); T[f"rc_{c}"] = rc[rc.origin == "2026-06"].set_index("style_color").f_honest_deepfm.reindex(T.style_color).values
T["lP"] = np.log(T.P50 + 1); l50 = np.log(T.pf2c_q50 + 1); T["pw_q25"] = np.maximum(np.abs(l50 - np.log(T.pf2c_q25 + 1)), 0.02); T["pw_q75"] = np.maximum(np.abs(np.log(T.pf2c_q75 + 1) - l50), 0.02)
for Z in (B, T):
    lw = lambda a, b: np.log((Z[b].values + 1) / (Z[a].values + 1))
    Z["w_pf2c"] = Z.pw_q25 + Z.pw_q75; Z["w_pf2d32"] = lw("pf2d32_q25", "pf2d32_q75"); Z["w_tree"] = lw("tree_q25", "tree_q75"); Z["w_rc"] = lw("rc_q25", "rc_q75")
    Z["l_lev"] = np.log1p(Z.level); Z["dev_r3"] = Z.lP - np.log1p(Z.R3); Z["adev_r3"] = Z.dev_r3.abs()
X = ["lP", "l_lev", "dev_r3", "adev_r3", "spread", "w_pf2c", "w_pf2d32", "w_tree", "w_rc", "age_m", "hist_months", "has_ly", "days_live", "pre2024", "newest_colour",
     "disc_item", "disc_rel", "disc_mom", "sh_tiktok", "sh_retail", "sh_partner", "sh_web", "n_stores", "n_partners", "trend91", "peak_r", "lagr1", "lagr3", "own_sea",
     "bf_window", "top_day_share", "oos_28", "cancel_rate", "n_sizes_life", "multi_share",
     "te_ent_d", "te_ent_w", "te_spec", "tr_acf1_d", "tr_acf7_d", "tr_acf1_w", "tr_acf4_w", "tr_wkday_max", "tm_cv_d", "tm_cv_w", "tm_zero_d", "tm_maxmean_d",
     "td_slope4", "td_slope13", "td_wk_since_peak"]
print("2026 inputs missing share (top):", T[X].isna().mean().sort_values(ascending=False).head(6).round(3).to_dict(), "| backtest:", B[X].isna().mean().sort_values(ascending=False).head(3).round(3).to_dict())
XB, XT = B[X].astype(float).values, T[X].astype(float).values; A = np.abs(B.e.values); e = B.e.values
PRM = dict(objective="l1", num_leaves=15, min_data_in_leaf=80, learning_rate=0.03, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=3, verbose=-1, seed=0)
fit = lambda m: lgb.train(PRM, lgb.Dataset(XB[m], A[m]), 400)
oof = np.full(len(B), np.nan)
for o in sorted(B.origin.unique()):
    te = (B.origin == o).values; oof[te] = np.maximum(fit(K.train_mask(B, o, "purged")).predict(XB[te]), 0.03)
kl, ku = K.factors(e, oof, oof, 0.25); print("k (lower, upper):", round(kl, 3), round(ku, 3))
lo, hi = B.lP.values - kl * oof, B.lP.values + ku * oof; print("in-sample check over all 13: below", round(float((B.ly < lo).mean()), 3), "above", round(float((B.ly > hi).mean()), 3))
w26 = np.maximum(fit(np.ones(len(B), bool)).predict(XT), 0.03)
O = pd.DataFrame({"style_color": T.style_color.values, "w_hat": w26, "r25": np.exp(-kl * w26), "r75": np.exp(ku * w26)})
AD = pd.read_csv("data/fk_ranges_2026.csv", comment="#").set_index("style_color"); O["r25_adopted"] = AD.r25.reindex(O.style_color).values; O["r75_adopted"] = AD.r75.reindex(O.style_color).values
O["target"] = AD.target.reindex(O.style_color).values
print(O.groupby("target")[["r25", "r25_adopted", "r75", "r75_adopted"]].describe().T.round(3).to_string())
print("rank correlation with the adopted P25-P75 log width (targets):", round(float(pd.Series(np.log(O.r75 / O.r25)[O.target.values]).corr(pd.Series(np.log(O.r75_adopted / O.r25_adopted)[O.target.values]), method="spearman")), 3))
hdr = ["# D370b (fork): width-model P25 / P75 ratios at the June 2026 origin, an option next to the adopted per-segment ratios (fork D329, data/fk_ranges_2026.csv; P10 / P90 stay there).",
       "# width model: LightGBM (median loss) on each item's absolute log miss around the candidate system's P50, 50 origin-known inputs (members' spread and own widths, level, P50 vs R3, age, entropy / rhythm, discount, channels, stores, trend, out-of-stock), fitted on all 13 backtest origins.",
       f"# r25 = exp(-{kl:.4f} x w_hat), r75 = exp({ku:.4f} x w_hat); factors fitted on purged out-of-fold widths so that 25% of actuals fall below P25 and 25% above P75 over the 13 origins. Apply as P = (P50 + 1) x r - 1.",
       "# backtest (D370 / D373b, purged, recent 9 origins): P25-P75 50.7% inside, pinball / R3's 0.3244 vs the adopted 0.3316 (95% interval of the difference -0.0101 to -0.0046); honest 49.4% vs 48.0%."]
with open("data/fk_ranges_2026_wm.csv", "w", encoding="utf-8", newline="") as fh: fh.write("\n".join(hdr) + "\n"); O.round(4).to_csv(fh, index=False)
print("written data/fk_ranges_2026_wm.csv", len(O))
