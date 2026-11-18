"""D161 (from eda/157): the D159 quantile model (D153 + temperature + events, data/q_backtest_wx_cpllm_we.parquet) stacked like D153. Earlier text: the D150/D151 quantile model (lean + cart-neighbour seasonal ramp + LLM scores, data/q_backtest_nf_cpllm.parquet) in the
three-way and stacked with the Black Friday fix, against the D141 best system. Built from eda/137 (D141), standard table: three-way on the lean + occasion-timing quantile model (D135), plus the Black Friday
fix (D137: November level from last year + item-lift allocation; applied as each item's 4-month adjustment ratio from eda/132's monthly
split: sum_h base_h x r_h / sum_h base_h). Rows: the three-way, each fix alone, both together. Recent, all 13, targets, BF origins."""
import numpy as np, pandas as pd, sys, importlib.util
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
T = pd.read_parquet("data/fixes_items.parquet")
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
q = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))
LQ = np.log1p(q[["q10", "q25", "q50", "q75", "q90"]].values); D = {c: LQ[:, j] - LQ[:, 2] for j, c in enumerate(["q10", "q25", "q50", "q75", "q90"])}
lean = pd.read_parquet("data/q_backtest_abl_leanocct.parquet").set_index(["origin", "style_color"]).q50.reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color])).values
T["q_lean"] = lean; lg = lambda c: np.log(T[c].clip(lower=0.5))
T["r_qlean_nn"] = np.exp((lg("rules") + lg("q_lean") + lg("nn")) / 3)
X = pd.read_parquet("data/bf_fix_ratios.parquet")
b = pd.read_parquet("data/best_backtest_preds.parquet")[["origin", "style_color", "h", "pred"]].rename(columns={"pred": "lgb_m"})
n = pd.read_parquet("data/nn_backtest_preds_S13x6.parquet")[["origin", "style_color", "h", "q50"]].rename(columns={"q50": "nn_m"})
RR = []
for o in T.origin.unique():
    r = pd.read_csv(f"agent_eval/{o}/predictions_verified.csv", dtype={"month": str}); r["h"] = [(pd.Period(m, "M") - pd.Period(o, "M")).n for m in r.month]; RR.append(r.assign(origin=o))
Mo = b.merge(n, on=["origin", "style_color", "h"]).merge(pd.concat(RR)[["origin", "style_color", "h", "pred"]].rename(columns={"pred": "ru_m"}), on=["origin", "style_color", "h"])
Mo["geo"] = np.exp((np.log(Mo.lgb_m.clip(lower=.01)) + np.log(Mo.nn_m.clip(lower=.01)) + np.log(Mo.ru_m.clip(lower=.01))) / 3)
Mo = Mo.merge(X, on=["origin", "style_color", "h"], how="left").fillna({"r_level": 1.0, "r_bf": 1.0})
adj = Mo.assign(gb=Mo.geo * Mo.r_bf).groupby(["origin", "style_color"]).apply(lambda x: x.gb.sum() / x.geo.sum()).rename("adj").reset_index()
T = T.merge(adj, on=["origin", "style_color"], how="left"); T["adj"] = T.adj.fillna(1.0)
T["r_q_nn_bf"] = T.r_q_nn * T.adj; T["best"] = T.r_qlean_nn * T.adj
T["q_d153"] = pd.read_parquet("data/q_backtest_nf_cpllm.parquet").set_index(["origin", "style_color"]).q50.reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color])).values
T["r_q153_nn"] = np.exp((lg("rules") + lg("q_d153") + lg("nn")) / 3); T["best153"] = T.r_q153_nn * T.adj
T["q_new"] = pd.read_parquet("data/q_backtest_wx_cpllm_we.parquet").set_index(["origin", "style_color"]).q50.reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color])).values
T["r_qnew_nn"] = np.exp((lg("rules") + lg("q_new") + lg("nn")) / 3); T["best_new"] = T.r_qnew_nn * T.adj
COLS = ["r_q_nn", "best", "r_q153_nn", "best153", "r_qnew_nn", "best_new"]
NM = {"r_q_nn": "three-way (reference)", "r_qlean_nn": "+ lean quantile model with occasion timing", "r_q_nn_bf": "+ Black Friday fix", "best": "lean + Black Friday fix (D141 best)", "r_qnew_nn": "+ D159 quantile (D153 + temperature + events)", "best_new": "D159 three-way + Black Friday fix", "r_q153_nn": "D153 three-way", "best153": "D153 three-way + Black Friday fix"}
BF = T.origin.isin(["2025-07", "2025-08", "2025-09", "2025-10"]).values; REC = (T.origin >= "2025-06").values
for nm, m in [("nine recent origins", REC), ("all 13 origins", np.ones(len(T), bool)), ("recent, 100 targets", REC & T.target.values), ("Black Friday origins", BF)]:
    Z = T[m].reset_index(drop=True); print(f"\n######## {nm}"); print(sl.table(Z, COLS, {k: v[m] for k, v in D.items()}, NM).round(4).to_string())
Z = T[REC].reset_index(drop=True); print("\nbootstrap vs three-way, recent:", {c: np.round(sl.boot(Z, c, "r_q_nn"), 4).tolist() for c in COLS[1:]})
print("bootstrap vs the D153 + BF system, recent:", {c: np.round(sl.boot(Z, c, "best153"), 4).tolist() for c in ["r_qnew_nn", "best_new"]})
