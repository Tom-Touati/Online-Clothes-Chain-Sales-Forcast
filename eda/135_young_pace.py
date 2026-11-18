"""Young-item momentum (D140): the members damp young items' recent pace both ways (falling-fast launches +10% over, rising -20% under
outside Black Friday windows). Add a pace-extrapolation member for items under 3 months only: P28 = 4 x level x exp(f_r28) (the last 28 days'
rate carried over the 4 months), geometrically with the three-way at fixed weights 1/4 and 1/3 (no tuning). Standard table on young items
and on all items (other items unchanged), recent origins and all 13."""
import numpy as np, pandas as pd, sys, importlib.util
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
T = pd.read_parquet("data/fixes_items.parquet")
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
q = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))
LQ = np.log1p(q[["q10", "q25", "q50", "q75", "q90"]].values); D = {c: LQ[:, j] - LQ[:, 2] for j, c in enumerate(["q10", "q25", "q50", "q75", "q90"])}
F = pd.read_parquet("data/features.parquet", columns=["origin", "style_color", "h", "f_r28", "level"]); F = F[F.h == 1].drop(columns="h").rename(columns={"level": "lvl_f"})
T = T.merge(F, on=["origin", "style_color"], how="left", suffixes=("", "_x"))
fr = T["f_r28_x"] if "f_r28_x" in T else T["f_r28"]
T["p28"] = 4 * ((T.lvl_f + 1) * np.exp(fr.fillna(0)) - 1).clip(lower=0)
young = (T.age_m < 3).values
for w, c in [(0.25, "y25"), (1 / 3, "y33")]:
    T[c] = np.where(young, np.exp((1 - w) * np.log(T.r_q_nn + 1) + w * np.log(T.p28 + 1)) - 1, T.r_q_nn)
COLS = ["r_q_nn", "p28", "y25", "y33"]; NM = {"r_q_nn": "three-way", "p28": "pace extrapolation (last 28 days x 4)", "y25": "three-way + pace 1/4 (young only)", "y33": "three-way + pace 1/3 (young only)"}
for nm, m in [("young items, recent", young & (T.origin >= "2025-06").values), ("young items, all 13", young), ("all items, recent", (T.origin >= "2025-06").values), ("all items, all 13", np.ones(len(T), bool))]:
    Z = T[m].reset_index(drop=True); print(f"\n######## {nm}: {len(Z)}"); print(sl.table(Z, COLS, {k: v[m] for k, v in D.items()}, NM).round(4).to_string())
Z = T[(T.origin >= "2025-06").values].reset_index(drop=True); print("\nbootstrap vs three-way, all items recent:", {c: np.round(sl.boot(Z, c, "r_q_nn"), 4).tolist() for c in ["y25", "y33"]})
Zy = T[young & (T.origin >= "2025-06").values].reset_index(drop=True); print("young recent:", {c: np.round(sl.boot(Zy, c, "r_q_nn"), 4).tolist() for c in ["y25", "y33"]})
