"""D431 (user, 18 Nov: "also add a table that shows the improvement of each metric over R3, seasonal naive, and agent rule based"): every
metric of the submitted system (scoreboard row nx9_geo = nx12's numbers: the six-member system with corrections and the campaign uplift) next
to R3, the seasonal naive and the agents' planning rules, on the nine recent backtest month-ends, all scored exactly as data/scoreboard.json
(pipeline/159_scoreboard.py's construction up to its scoring loop is reused: the same item-origins, the same seasonal naive, pipeline/score_lib.py and
pipeline/159's per-item views; pinball gives each forecast the stored quantile model's log spread x 1.25). Buying loss from pipeline/231_sim_nx11.py
(data/sim_nx11_windows.csv; the seasonal naive was simulated by pipeline/231b_sim_naive.py when its windows are there).
Writes data/metrics_table_nx12.csv (one row per metric: the four values and the system's improvement over each baseline) and prints it."""
import os, sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
src = open("pipeline/159_scoreboard.py", encoding="utf-8").read(); exec(src.split("cols = [s[4] for s in SYSTEMS]")[0])   # T, D, REC, sl, sea_naive
m = REC; Z = T[m].reset_index(drop=True); DZ = {k: v[m] for k, v in D.items()}
SYS = {"System": "nx9_geo", "R3": "R3", "Seasonal naive": "sea_naive", "Agents' rules": "rules"}
assert Z[list(SYS.values())].notna().all().all(), "a forecast is missing on the recent item-origins"
y = Z.y.values
def per(c):
    F = Z[c].values.astype(float); e = np.abs(F - y); e3 = np.abs(Z.R3.values - y)
    ape = e / np.maximum(y, 1e-9); hit = (e <= 0.2 * y) & (y > 0); ly, lf = np.log1p(y), np.log1p(np.clip(F, 0, None))
    return {"WAPE (error / units)": e.sum() / y.sum(),
            "Median error per item": float(np.median(ape[y > 0])),
            "Items within +/-20%": float(hit.mean()),
            "Units within +/-20%": float((hit * y).sum() / y.sum()),
            "Bias (forecast / actual - 1)": F.sum() / y.sum() - 1,
            "(MAE + |bias|) / actual units": float(sl.mb(Z, c)),
            "R2 on units": 1 - ((F - y) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
            "R2 on log units": 1 - ((lf - ly) ** 2).sum() / ((ly - ly.mean()) ** 2).sum(),
            "Pinball loss, P10-P90 (/ units)": float(sl.pin(Z, c, DZ))}
V = pd.DataFrame({k: per(c) for k, c in SYS.items()})
sim = pd.read_csv("data/sim_nx11_windows.csv"); L = sim.groupby("system").loss_pct.mean()
V.loc["Buying loss (% of perfect-foresight profit)"] = [L.get("nx11", np.nan), L.get("R3", np.nan), L.get("sea_naive", np.nan), L.get("rules", np.nan)]
LOWER = {"WAPE (error / units)", "Median error per item", "(MAE + |bias|) / actual units", "Pinball loss, P10-P90 (/ units)", "Buying loss (% of perfect-foresight profit)"}
def better(metric, s, b):
    if not np.isfinite(b) or not np.isfinite(s): return np.nan
    if metric in LOWER: return 1 - s / b                                             # share of the baseline's error removed
    if metric.startswith("Bias"): return abs(b) - abs(s)                             # points of bias removed
    return s - b                                                                       # points gained (shares, R2)
out = V.copy()
for b in ["R3", "Seasonal naive", "Agents' rules"]: out[f"vs {b}"] = [better(i, V.at[i, "System"], V.at[i, b]) for i in V.index]
# win rate: share of item-origins where the system is closer than each baseline (ties half), as the scoreboard
for b, c in [("R3", "R3"), ("Seasonal naive", "sea_naive"), ("Agents' rules", "rules")]:
    es, eb = np.abs(Z.nx9_geo.values - y), np.abs(Z[c].values - y); out.loc["System closer than the baseline (items)", f"vs {b}"] = float(((es < eb) + 0.5 * (es == eb)).mean())
out.to_csv("data/metrics_table_nx12.csv")
chk = {"WAPE / R3": V.at["WAPE (error / units)", "System"] / V.at["WAPE (error / units)", "R3"], "WAPE / seasonal": V.at["WAPE (error / units)", "System"] / V.at["WAPE (error / units)", "Seasonal naive"],
       "WAPE / rules": V.at["WAPE (error / units)", "System"] / V.at["WAPE (error / units)", "Agents' rules"]}
print(f"{len(Z)} item-origins, nine recent month-ends | ratios (scoreboard: 0.3795 / 0.6077 / 0.778):", {k: round(v, 4) for k, v in chk.items()})
print(out.round(4).to_string())
