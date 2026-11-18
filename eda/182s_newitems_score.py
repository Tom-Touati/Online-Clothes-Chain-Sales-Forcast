"""D263 scoring of the new-items models (data/newitems_tot.parquet, data/newitems_preds.parquet).
(a) Young items (< 6 months at the origin), recent origins: WAPE / R3 and bias of the current best system E:ec_ns_x_geo (D259), p2c32a,
    the new-items models alone (tw = Tweedie count, mix = burst mixture, q4 = separate 4-month tree) and 50/50 blends with the system.
(b) The whole system with young items replaced by each blend (other items unchanged): standard table on recent / all 13 / Black Friday
    origins and the 100 targets, bootstrap against E:ec_ns_x_geo.
(c) Burst probability: AUC and Brier (vs the base rate) per horizon, reliability by decile of p, and P(any burst in the window) against the
    4-month breakout of the system (actual > 1.65 x forecast, >= 20 units). Writes eda/newitems_score_output.txt (stdout)."""
import sys, contextlib, io, importlib.util, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color])
G = pd.read_parquet("data/geo_ec_ns_x_preds.parquet").set_index(["origin", "style_color"]).reindex(key)
G0 = pd.read_parquet("data/dec_base.parquet").set_index(["origin", "style_color"]).reindex(key)
N = pd.read_parquet("data/newitems_tot.parquet").set_index(["origin", "style_color"]).reindex(key)
T["best"] = G.geo.values; T["p2c32a"] = G0.F.values; T["young"] = N.tw.notna().values & (T.age_m < 6).values if "age_m" in T else N.tw.notna().values
for c in ["tw", "mix", "q4"]: T[c] = np.where(T.young, N[c].values, T.best)
T["b_tw"] = np.where(T.young, (T.best + T.tw) / 2, T.best); T["b_mix"] = np.where(T.young, (T.best + T.mix) / 2, T.best)
T["b_q4"] = np.where(T.young, (T.best + T.q4) / 2, T.best); T["b_3"] = np.where(T.young, (T.best + T.mix + T.q4) / 3, T.best)
REC_ = [o for o in ORIGINS if o >= "2025-06"]; R = T[T.origin.isin(REC_) & T.young]
print(f"== (a) young items (< 6 months), recent origins: {len(R)} item-origins, {R.y.sum() / T[T.origin.isin(REC_)].y.sum():.1%} of units")
for c in ["p2c32a", "best", "tw", "mix", "q4", "b_tw", "b_mix", "b_q4", "b_3"]:
    print(f"  {c:7s} WAPE / R3 {(R[c] - R.y).abs().sum() / (R.R3 - R.y).abs().sum():.4f} | bias {R[c].sum() / R.y.sum() - 1:+.3f} | relMAE vs R3 {np.exp(np.mean(np.log(((R[c] - R.y).abs() + 1) / ((R.R3 - R.y).abs() + 1)))):.4f}")
COLS = ["p2c32a", "best", "b_tw", "b_mix", "b_q4", "b_3", "mix", "q4"]
for nm, m in [("nine recent origins", T.origin.isin(REC_)), ("all 13 origins", np.ones(len(T), bool)), ("Black Friday origins", T.origin.isin(["2025-07", "2025-08", "2025-09", "2025-10"])), ("recent, 100 targets", T.origin.isin(REC_) & T.target)]:
    print(f"\n######## (b) {nm}"); print(sl.table(T[m], COLS, D).round(4).to_string())
F_ = T[T.origin.isin(REC_)]; print("\nbootstrap vs best (E:ec_ns_x_geo), recent:"); [print(f"  {c:7s}", np.round(sl.boot(F_, c, "best"), 4)) for c in COLS if c not in ("best",)]
P = pd.read_parquet("data/newitems_preds.parquet"); P = P[P.origin >= "2025-06"]
print("\n== (c) burst probability (recent origins, young items)")
for h, g in P.groupby("h"):
    br = np.mean((g.burst - g.burst.mean()) ** 2); bs = np.mean((g.burst - g.p_burst) ** 2)
    print(f"  h {h}: burst rate {g.burst.mean():.3f}, mean p {g.p_burst.mean():.3f}, AUC {roc_auc_score(g.burst, g.p_burst):.3f}, Brier {bs:.4f} vs base-rate {br:.4f} (skill {1 - bs / br:+.3f})")
P["dec"] = pd.qcut(P.p_burst, 10, labels=False, duplicates="drop")
print("  reliability by decile of p (mean p -> observed rate):", P.groupby("dec").apply(lambda d: f"{d.p_burst.mean():.2f}->{d.burst.mean():.2f}").tolist())
W = T[T.origin.isin(REC_) & T.young].copy(); W["p_any"] = N.p_any.reindex(pd.MultiIndex.from_frame(W[["origin", "style_color"]])).values
W["brk"] = ((np.log((W.y + 1) / (W.best + 1)) > 0.5) & (W.y >= 20)).astype(int); W = W[(W.y >= 20) | (W.best >= 20)]
print(f"  P(any burst in the window) vs the system's 4-month breakout (young, n {len(W)}, rate {W.brk.mean():.3f}): AUC {roc_auc_score(W.brk, W.p_any):.3f} (D247 breakout model AUC on young items 0.666)")
