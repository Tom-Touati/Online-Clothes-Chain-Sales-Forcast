"""The winning fixes stacked (D141), standard table: three-way on the lean + occasion-timing quantile model (D135), plus the Black Friday
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
COLS = ["r_q_nn", "r_qlean_nn", "r_q_nn_bf", "best"]
NM = {"r_q_nn": "three-way (reference)", "r_qlean_nn": "+ lean quantile model with occasion timing", "r_q_nn_bf": "+ Black Friday fix", "best": "+ both"}
BF = T.origin.isin(["2025-07", "2025-08", "2025-09", "2025-10"]).values; REC = (T.origin >= "2025-06").values
for nm, m in [("nine recent origins", REC), ("all 13 origins", np.ones(len(T), bool)), ("recent, 100 targets", REC & T.target.values), ("Black Friday origins", BF)]:
    Z = T[m].reset_index(drop=True); print(f"\n######## {nm}"); print(sl.table(Z, COLS, {k: v[m] for k, v in D.items()}, NM).round(4).to_string())
Z = T[REC].reset_index(drop=True); print("\nbootstrap vs three-way, recent:", {c: np.round(sl.boot(Z, c, "r_q_nn"), 4).tolist() for c in COLS[1:]})
# value ladder figure (D141): WAPE / R3 and MAE + |bias| / R3 per stage, recent origins -> eda/figs/value_ladder.png
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Fs = pd.read_parquet("data/features.parquet", columns=["origin", "style_color", "h", "m3", "lfl_type"]); Fs["st"] = Fs.m3 * np.exp(Fs.lfl_type.fillna(0))
T = T.merge(Fs.groupby(["origin", "style_color"]).st.sum().rename("sea_type").reset_index(), on=["origin", "style_color"], how="left")
Z = T[REC].reset_index(drop=True); Dz = {k: v[REC] for k, v in D.items()}
st = [("R3", "R3 (last 3 months x 4)"), ("sea_type", "x type seasonality"), ("qmodel", "quantile model"), ("q_nn", "+ NN"), ("r_q_nn", "+ rules (three-way)"),
      ("r_qlean_nn", "+ lean model, occasion timing"), ("best", "+ Black Friday fix")]
tb = sl.table(Z, [c for c, _ in st], Dz, dict(st))
vals = [tb.loc[n, "WAPE / R3"] for _, n in st]; vb = [tb.loc[n, "(MAE+|bias|) / R3"] for _, n in st]
fig, ax = plt.subplots(figsize=(11, 4.8)); x = np.arange(len(st))
ax.bar(x - 0.2, vals, 0.4, label="WAPE / R3", color="#2a78d6"); ax.bar(x + 0.2, vb, 0.4, label="(MAE + |bias|) / R3", color="#eb6834")
for i, (a, b2) in enumerate(zip(vals, vb)): ax.text(i - 0.2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8); ax.text(i + 0.2, b2 + 0.01, f"{b2:.3f}", ha="center", fontsize=8)
ax.axhline(tb.loc["R3 + rules (rule agents)", "WAPE / R3"], color="green", lw=1, ls="--", label="R3 + rules (rule agents), WAPE / R3")
ax.set_xticks(x); ax.set_xticklabels([n for _, n in st], rotation=20, ha="right", fontsize=9); ax.set_ylabel("ratio to R3 (lower is better)")
ax.set_title("Value ladder, nine recent origins: each stage's error as a ratio to recency"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("eda/figs/value_ladder.png", dpi=80)
print("ladder:", dict(zip([n for _, n in st], np.round(vals, 4))))
