"""Score the bottom-up (per location, summed) forecast of eda/113 against the item-level systems in the standard table (D125).
Rows: R3 + rules, quantile model (item level), bottom-up sum, their geometric mean, the three-way, the three-way with bottom-up in place of
the quantile model, and the four-way (rules, quantile, NN, bottom-up). Also the rollout gap: share of actual window units sold at
locations where the item had no sales in the 3 months before the origin (invisible to a bottom-up model), by retail share."""
import numpy as np, pandas as pd, sys, importlib.util
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
T = pd.read_parquet("data/fixes_items.parquet")
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
q = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))
LQ = np.log1p(q[["q10", "q25", "q50", "q75", "q90"]].values); D = {c: LQ[:, j] - LQ[:, 2] for j, c in enumerate(["q10", "q25", "q50", "q75", "q90"])}
U = pd.read_parquet("data/bu_items.parquet"); T = T.merge(U, on=["origin", "style_color"], how="left")
print("items without a bottom-up forecast:", int(T.bu.isna().sum())); T["bu"] = T.bu.fillna(T.qmodel)
lg = lambda c: np.log(T[c].clip(lower=0.5))
T["bu_q"] = np.exp((lg("bu") + lg("qmodel")) / 2); T["r_bu_nn"] = np.exp((lg("rules") + lg("bu") + lg("nn")) / 3)
T["four_bu"] = np.exp((lg("rules") + lg("qmodel") + lg("nn") + lg("bu")) / 4)
COLS = ["qmodel", "bu", "bu_q", "r_q_nn", "r_bu_nn", "four_bu"]
NM = {"qmodel": "quantile model (item level)", "bu": "bottom-up (per location, summed)", "bu_q": "bottom-up x quantile", "r_q_nn": "rules + quantile + NN",
      "r_bu_nn": "rules + bottom-up + NN", "four_bu": "rules + quantile + NN + bottom-up"}
REC = [o for o in sorted(T.origin.unique()) if o >= "2025-06"]
for nm, Z in [("nine recent origins", T[T.origin.isin(REC)]), ("all 13 origins", T), ("recent, 100 targets", T[T.origin.isin(REC) & T.target])]:
    print(f"\n######## {nm}"); print(sl.table(Z, COLS, D, NM).round(4).to_string())
Z = T[T.origin.isin(REC)]
print("\nbootstrap WAPE(x) - WAPE(y), recent:")
for a, b in [("bu", "qmodel"), ("bu_q", "qmodel"), ("r_bu_nn", "r_q_nn"), ("four_bu", "r_q_nn")]: print(f"  {a} - {b}:", np.round(sl.boot(Z, a, b), 4))
gap = (Z.y - Z.cov_actual).clip(lower=0)
print(f"\nrollout gap (recent): {gap.sum() / Z.y.sum():.1%} of actual units were sold at locations with no sales by the item in the 3 months before the origin")
Z = Z.assign(gap=gap, rb=pd.cut(Z.sh_ret.fillna(0), [-1, 0.05, 0.2, 0.4, 1.1], labels=["<5%", "5-20%", "20-40%", ">40%"]))
print("by retail share:", Z.groupby("rb").apply(lambda x: round(x.gap.sum() / x.y.sum(), 3)).to_dict(), "| by age:", Z.groupby("age").apply(lambda x: round(x.gap.sum() / x.y.sum(), 3)).to_dict())
print("bias of bottom-up on covered locations only:", round(Z.bu.sum() / Z.cov_actual.sum() - 1, 3), "| on all units:", round(Z.bu.sum() / Z.y.sum() - 1, 3))
