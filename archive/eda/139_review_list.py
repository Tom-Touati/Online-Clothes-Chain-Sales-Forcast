"""Review list for the Jul-Oct 2026 targets (D143): member disagreement is the strongest error signal found (D114: the most-disagreeing
fifth has twice the error), so the targets where the members disagree most are the ones worth a human look. Per target: the June 2026 rule
agent, the quantile P50, the NN, the LightGBM monthly model (4-month totals), their disagreement (sd of logs), the D121 candidate's final
number and its conformal P10-P90 (eda/125), and the two largest drivers from the 2026 waterfall (eda/101, TreeSHAP groups).
Writes data/review_list_2026.csv and prints the top 15."""
import numpy as np, pandas as pd, sys
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
S = pd.read_csv("data/final_forecast_q4.csv", index_col=0); C = pd.read_csv("data/cand_rnns_final_forecast_q3.csv", index_col=0)
R6 = pd.read_csv("agent_eval/2026-06/predictions_verified.csv").groupby("style_color").pred.sum()
N6 = sum(pd.read_parquet(f"data/{f}.parquet").groupby("style_color").q50.sum() for f in ["nn_backtest_preds_finalS", "nn_backtest_preds_finalS_s10"]) / 2
L6 = pd.read_csv("data/final_forecast.csv", index_col=0)[["jul", "aug", "sep", "oct"]].sum(axis=1)
Q = pd.read_csv("data/cand_rnn_conformal_2026.csv", index_col=0); SH = pd.read_csv("data/waterfall_2026_shap_log.csv", index_col=0)
it = pd.read_parquet("data/items.parquet").set_index("style_color")
idx = C.index; X = pd.DataFrame({"name": it.product_name.reindex(idx).str[:30], "type": it.product_type.reindex(idx), "age_m": C.age_m.round(1),
                                 "rules": R6.reindex(idx).round(0), "quantile": S.q50.reindex(idx).round(0), "nn": N6.reindex(idx).round(0), "lightgbm": L6.reindex(idx).round(0)})
X["disagreement"] = np.log(X[["rules", "quantile", "nn", "lightgbm"]].clip(lower=0.5)).std(axis=1).round(3)
X["max/min member"] = (X[["rules", "quantile", "nn", "lightgbm"]].max(axis=1) / X[["rules", "quantile", "nn", "lightgbm"]].min(axis=1).clip(lower=1)).round(2)
X["candidate"] = C.final_total; X["submitted"] = S.final_total.reindex(idx)
X["P10"] = Q.P10.reindex(idx); X["P90"] = Q.P90.reindex(idx)
drv = SH.drop(columns="base").reindex(idx)
X["main drivers (x)"] = [", ".join(f"{g} x{np.exp(v):.2f}" for g, v in r.dropna().abs().sort_values(ascending=False).head(2).items() for v in [r[g]]) if r.notna().any() else "not in model set" for _, r in drv.iterrows()]
X = X.sort_values("disagreement", ascending=False); X.to_csv("data/review_list_2026.csv")
print("targets with members more than 2x apart:", int((X["max/min member"] > 2).sum()), "| units in them (candidate):", int(X.loc[X["max/min member"] > 2, "candidate"].sum()), "of", int(X.candidate.sum()))
print(X.head(15).to_string())
