"""Fork 2 (user: "make the new system nx5 with the new calibration and routing for new vs old items"): base table for the new nx5.
nx5 (D361) = rules, Huber tree, NN ec, TabPFN conformal, TabPFN 32 and rc_deepfm at a quarter weight; its base table with the backtest and the
June 2026 rows is data/dec_base_ecnsnnhubrq26.parquet. The tree column (qdyn) is replaced by the routed tree (D389h): the Huber tree trained
with new items' rows x5 (eda/187 huber_new5) for items under 6 months old at the origin, the plain Huber tree for the rest.
F = (sum of the five + 0.25 rc) / 5.25 as in nx5; spread / e / ae recomputed. Writes data/dec_base_nx5r26.parquet (corrections then run with
DEC=_nx5r26; they train only on o' + 4 <= o, so the backtest rows are the time-honest backtest and the 2026-06 rows the forecast)."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
B = pd.read_parquet("data/dec_base_ecnsnnhubrq26.parquet"); K = pd.MultiIndex.from_frame(B[["origin", "style_color"]])
rd = lambda f: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).q50.reindex(K).values
h = np.where(B.origin == "2026-06", rd("data/q_backtest_ls_huber_2026.parquet"), rd("data/q_backtest_ls_huber.parquet"))
h5 = np.where(B.origin == "2026-06", rd("data/q_backtest_ls_huber_new5_2026.parquet"), rd("data/q_backtest_ls_huber_new5.parquet"))
print("Huber file reproduces the base's tree column:", float(np.nanmax(np.abs(h / B.qdyn.values - 1))), "| rows missing", int(np.isnan(h).sum()), int(np.isnan(h5).sum()))
new = B.age_m.fillna(99).values < 6
B["qdyn"] = np.where(new & np.isfinite(h5), h5, B.qdyn.values)
MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32", "rc"]; W = np.array([1, 1, 1, 1, 1, 0.25])
B["F"] = (B[MEM].clip(lower=0).values * W).sum(1) / W.sum()
B["spread"] = np.log(B[MEM].clip(lower=1).max(1) / B[MEM].clip(lower=1).min(1))
bt = B.y.notna(); B.loc[bt, "e"] = np.log((B.y[bt] + 1) / (B.F[bt] + 1)); B.loc[bt, "ae"] = (B.F[bt] - B.y[bt]).abs()
REC = (B.origin >= "2025-06") & bt; HO = (B.origin >= "2025-10") & bt
w = lambda m, f: (B.F[m] - B.y[m]).abs().sum() / (B.R3[m] - B.y[m]).abs().sum()
O0 = pd.read_parquet("data/dec_base_ecnsnnhubrq26.parquet")
print("before corrections, WAPE / R3: recent", round(w(REC, None), 4), "(nx5", round(((O0.F - O0.y).abs()[REC].sum() / (O0.R3 - O0.y).abs()[REC].sum()), 4), ") | held out", round(w(HO, None), 4),
      "(nx5", round(((O0.F - O0.y).abs()[HO].sum() / (O0.R3 - O0.y).abs()[HO].sum()), 4), ")")
T26 = (B.origin == "2026-06") & B.target.astype(bool); print("2026 targets: F", round(B.F[T26].sum()), "nx5", round(O0.F[T26].sum()), "| new items among targets", int((new & T26).sum()))
B.to_parquet("data/dec_base_nx5r26.parquet", index=False)
