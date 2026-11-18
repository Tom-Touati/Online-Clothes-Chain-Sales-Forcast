"""Predicted quantiles for the Jul-Oct 2026 targets (D133): binned conformal ranges (D117 'D2') learned from all 13 backtest origins
(every 4-month outcome is complete by June 2026), centred on the rules + quantile + NN candidate's final number (data/cand_rnn_final_forecast_q4.csv).
Bins: member disagreement (sd of log forecasts of rules, quantile P50, NN, LightGBM) quintiles x the quantile model's log IQR terciles,
from the backtest; 2026 members: the June 2026 rule agent, the stored quantile P50, the NN (finalS + finalS_s10), the LightGBM monthly
point model (data/final_forecast.csv). Bins under 150 rows fall back to the disagreement bin. Writes data/cand_rnn_conformal_2026.csv and
prints coverage of the same method on the backtest (leave-one-origin-out) as a check."""
import numpy as np, pandas as pd, sys
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
T = pd.read_parquet("data/fixes_items.parquet")
q = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))
QC = ["q10", "q25", "q50", "q75", "q90"]; TAU = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
LQ = np.log1p(q[QC].values); qiqr = LQ[:, 3] - LQ[:, 1]
dis = np.log(T[["rules", "qmodel", "nn", "lgb"]].clip(lower=0.5).values).std(axis=1); r = np.log1p(T.y.values) - np.log1p(T.r_q_nn.values)
db = np.quantile(dis, [0.2, 0.4, 0.6, 0.8]); qb = np.quantile(qiqr, [1 / 3, 2 / 3])
def offsets(d_new, q_new, mask):
    bd, bq = np.digitize(dis[mask], db), np.digitize(qiqr[mask], qb); out = np.zeros((len(d_new), 5))
    for i, (a, b) in enumerate(zip(np.digitize(d_new, db), np.digitize(q_new, qb))):
        sel = (bd == a) & (bq == b)
        if sel.sum() < 150: sel = bd == a
        qq = np.quantile(r[mask][sel], TAU); out[i] = qq - qq[2]
    return out
# leave-one-origin-out check on the backtest
cov = []
for o in sorted(T.origin.unique()):
    te = T.origin.values == o; off = offsets(dis[te], qiqr[te], ~te); Q = np.expm1(np.log1p(T.r_q_nn.values[te])[:, None] + off); y = T.y.values[te]
    cov.append((o, np.mean((y >= Q[:, 1]) & (y <= Q[:, 3])), np.mean((y >= Q[:, 0]) & (y <= Q[:, 4]))))
C = pd.DataFrame(cov, columns=["origin", "P25-75", "P10-90"]); print("leave-one-origin-out coverage:"); print(C.round(3).to_string(index=False)); print("mean", C[["P25-75", "P10-90"]].mean().round(3).to_dict())
# 2026
Cd = pd.read_csv("data/cand_rnn_final_forecast_q4.csv", index_col=0); S = pd.read_csv("data/final_forecast_q4.csv", index_col=0)
R6 = pd.read_csv("agent_eval/2026-06/predictions_verified.csv").groupby("style_color").pred.sum()
N6 = sum(pd.read_parquet(f"data/{f}.parquet").groupby("style_color").q50.sum() for f in ["nn_backtest_preds_finalS", "nn_backtest_preds_finalS_s10"]) / 2
L6 = pd.read_csv("data/final_forecast.csv", index_col=0)[["jul", "aug", "sep", "oct"]].sum(axis=1)
idx = Cd.index; mem = pd.DataFrame({"rules": R6.reindex(idx), "q": S.q50.reindex(idx), "nn": N6.reindex(idx), "lgb": L6.reindex(idx)})
d6 = np.log(mem.clip(lower=0.5)).std(axis=1).fillna(np.median(dis)).values; q6 = (np.log1p(S.q75.reindex(idx)) - np.log1p(S.q25.reindex(idx))).fillna(np.median(qiqr)).values
off = offsets(d6, q6, np.ones(len(T), bool)); c = np.log1p(Cd.final_total.values)[:, None]
Q = pd.DataFrame(np.expm1(c + off).round(0), index=idx, columns=["P10", "P25", "P50", "P75", "P90"]); Q["final_total"] = Cd.final_total; Q["disagreement"] = d6.round(3)
Q["range_ratio_P90_P10"] = (Q.P90 / Q.P10.clip(lower=1)).round(2); Q.to_csv("data/cand_rnn_conformal_2026.csv")
print("\n2026 targets: sum of P10 / P25 / P50 / P75 / P90 (not a portfolio interval):", Q[["P10", "P25", "P50", "P75", "P90"]].sum().astype(int).to_dict())
print("median P90/P10 ratio:", round(float(Q.range_ratio_P90_P10.median()), 2), "| production calibrated P90/P10 (p90c/p10c):", round(float((Cd.p90c / Cd.p10c).median()), 2))
print(Q.sort_values("range_ratio_P90_P10", ascending=False).head(8).to_string())
