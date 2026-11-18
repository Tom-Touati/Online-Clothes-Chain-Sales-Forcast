"""Untuned combination probes (D115b) on the same items: is the equal geometric three-way the best simple pooling?
  r_q_nn      geometric mean of rules, quantile, NN (current best)
  four        + LightGBM monthly ensemble as a fourth member (geometric)
  med3, med4  median of the three / four members
  trim4       geometric mean of the middle two of the four
  arith3      arithmetic mean of the three
  r_q_nn_R3   the three-way with R3 as a small fourth member (weight 0.1) - does recency add anything?"""
import numpy as np, pandas as pd, sys, importlib.util
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
T = pd.read_parquet("data/fixes_items.parquet")
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
q = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))
LQ = np.log1p(q[["q10", "q25", "q50", "q75", "q90"]].values); D = {c: LQ[:, j] - LQ[:, 2] for j, c in enumerate(["q10", "q25", "q50", "q75", "q90"])}
L3 = np.log(T[["rules", "qmodel", "nn"]].clip(lower=0.5).values); L4 = np.log(T[["rules", "qmodel", "nn", "lgb"]].clip(lower=0.5).values)
T["four"] = np.exp(L4.mean(1)); T["med3"] = np.exp(np.median(L3, 1)); T["med4"] = np.exp(np.median(L4, 1))
s4 = np.sort(L4, 1); T["trim4"] = np.exp(s4[:, 1:3].mean(1)); T["arith3"] = T[["rules", "qmodel", "nn"]].mean(1)
T["r_q_nn_R3"] = np.exp(0.9 * L3.mean(1) + 0.1 * np.log(T.R3.clip(lower=0.5)))
COLS = ["r_q_nn", "four", "med3", "med4", "trim4", "arith3", "r_q_nn_R3"]
REC = [o for o in sorted(T.origin.unique()) if o >= "2025-06"]
for nm, X in [("nine recent origins", T[T.origin.isin(REC)]), ("all 13 origins", T), ("recent, 100 targets", T[T.origin.isin(REC) & T.target])]:
    print(f"\n######## {nm}"); print(sl.table(X, COLS, D).round(4).to_string())
X = T[T.origin.isin(REC)]
print("\n== bootstrap WAPE(x) - WAPE(three-way), recent:"); [print(f"  {c:10s}", np.round(sl.boot(X, c, "r_q_nn"), 4)) for c in COLS[1:]]
