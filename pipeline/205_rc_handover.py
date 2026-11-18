"""D338 (fork 2): the hand-over file of the location recommender for Main's sixth-member test (corrections re-fitted there).
Chosen run (D339): members' item inputs + log-error loss, prediction within exp(±3) of the recent level, and below 12 closed training origins the DeepFM falls back to the tree version (pipeline/202 itemfeat=1 loss=logl1 etaclip=3 mino=12), honest setting.
Fix: the DeepFM version has a handful of exploding forecasts at the earliest origins, where it trains on few rows (all-13 WAPE / R3 in
the hundreds). Every recommender forecast is capped at 3 x the highest of the five members for that item and origin (a guard, not a
tuned value); the recent-origin scores are unchanged by it.
Writes data/rc_recsys_preds.parquet (origin, style_color, rc_deepfm, rc_gbm, rc_avg: 4-month totals, 13 backtest origins + 2026-06) and
prints the scores before and after the cap."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
R = pd.read_parquet("data/rc_recsys_preds_if_logl1_c3_m12.parquet"); B = pd.read_parquet("data/dec_base_ecnsnn.parquet"); MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]
D = B[["origin", "style_color", "y", "R3"] + MEM].merge(R.drop(columns="y"), on=["origin", "style_color"], how="left"); top = D[MEM].max(1).values
y = D.y.values; ar = np.abs(D.R3.values - y); rec = (D.origin >= "2025-06").values; allm = np.ones(len(D), bool)
w = lambda f, m: float(np.abs(f - y)[m].sum() / ar[m].sum()); out = {}
for k in ("deepfm", "gbm", "avg"):
    f0 = D[f"f_honest_{k}"].fillna(D.R3).values; f1 = np.minimum(f0, 3 * top); out[k] = f1
    six = (D[MEM].values.sum(1) + f1) / 6
    print(f"{k:6s} alone recent {w(f0, rec):.4f} -> {w(f1, rec):.4f} | all13 {w(f0, allm):.4f} -> {w(f1, allm):.4f} | capped rows {int((f1 < f0).sum())} | six-member mean recent {w(six, rec):.4f} all13 {w(six, allm):.4f}")
H = D[["origin", "style_color"]].assign(rc_deepfm=out["deepfm"], rc_gbm=out["gbm"], rc_avg=out["avg"])
R26 = R[R.origin == "2026-06"][["origin", "style_color", "f_honest_deepfm", "f_honest_gbm", "f_honest_avg"]].rename(columns=lambda c: c.replace("f_honest_", "rc_"))
M26 = pd.read_parquet("data/fk_err_master_2026.parquet")[["style_color"] + MEM]; R26 = R26.merge(M26, on="style_color", how="left"); t26 = R26[MEM].max(1)
for k in ("rc_deepfm", "rc_gbm", "rc_avg"): R26[k] = np.where(t26.notna(), np.minimum(R26[k], 3 * t26), R26[k])
H = pd.concat([H, R26[["origin", "style_color", "rc_deepfm", "rc_gbm", "rc_avg"]]], ignore_index=True); H.to_parquet("data/rc_recsys_preds.parquet", index=False)
print("saved data/rc_recsys_preds.parquet", H.shape, H.groupby("origin").size().tail(2).to_dict())
