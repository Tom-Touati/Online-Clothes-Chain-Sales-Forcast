"""D371b (fork; attempts 2-3 of Fork 2's delegation): June 2026 ranges for totals by the studentised conformal rule of D371 (T4), for the
candidate system (nx5 P50, data/system_q2575_2026.parquet) and its per-segment item ranges (fork D329, data/fk_ranges_2026.csv).
  spread  s = sqrt(s_items^2 + c^2): s_items from the item half-ranges added in quadrature (normal approximation, relative to the total),
          c = the root mean square of the all-item total's log miss over the 13 backtest origins (the shared origin term)
  bounds  log(F + 1) + q(r) x s, with q the 25th / 75th / 10th / 90th percentiles of r = log miss / s over all backtest groups (all levels,
          all 13 origins; data/fk_total_ranges_backtest.parquet)
Levels: product group, product type, look (3+ items), all items. Also the backtest share within +/-20% by level. Writes
data/fk_total_ranges_2026.csv; adds J["total_ranges_2026"]; stdout -> eda/total_ranges_2026_output.txt."""
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
S = pd.read_parquet("data/system_q2575_2026.parquet"); T = pd.read_parquet("data/fk_err_master_2026.parquet")[["style_color", "macro", "ptype", "look"]]
RG = pd.read_csv("data/fk_ranges_2026.csv", comment="#"); A = S.merge(T, on="style_color", validate="1:1").merge(RG[["style_color", "r10", "r25", "r75", "r90"]], on="style_color", validate="1:1")
assert len(A) == 1423 and A.P50.notna().all()
for q, r_ in [("L50", "r25"), ("H50", "r75"), ("L80", "r10"), ("H80", "r90")]: A[q] = np.maximum((A.P50 + 1) * A[r_] - 1, 0)
Z50 = 0.6745; A["v50l"] = ((A.P50 - A.L50) / Z50) ** 2; A["v50h"] = ((A.H50 - A.P50) / Z50) ** 2; A["all"] = "all items"
G = pd.read_parquet("data/fk_total_ranges_backtest.parquet"); J = json.load(open("data/fk_err_analysis.json")); TOT = pd.Series(J["total_ranges"]["total_miss"])
c = float(np.sqrt(np.mean(TOT.values ** 2)))
G["s"] = np.sqrt(((np.sqrt(G.v50l) + np.sqrt(G.v50h)) / 2 / (G.F + 1)) ** 2 + c ** 2); r = (G.e / G.s).values
qs = {t: float(np.quantile(r, t)) for t in [0.10, 0.25, 0.75, 0.90]}; print("shared origin term c:", round(c, 4), "| quantiles of the studentised miss:", {k: round(v, 3) for k, v in qs.items()})
hit = G.assign(h=(G.F - G.y).abs() <= 0.2 * G.y).groupby("level").h.mean()
LV = {"product group": "macro", "product type": "ptype", "look": "look", "total": "all"}; out = []
for lab, key in LV.items():
    g = A.groupby(key).agg(F=("P50", "sum"), n=("P50", "size"), v50l=("v50l", "sum"), v50h=("v50h", "sum")).reset_index().rename(columns={key: "group"}); g = g[g.n >= 3]
    s = np.sqrt(((np.sqrt(g.v50l) + np.sqrt(g.v50h)) / 2 / (g.F + 1)) ** 2 + c ** 2); lF = np.log(g.F + 1)
    for t, nm in [(0.10, "P10"), (0.25, "P25"), (0.75, "P75"), (0.90, "P90")]: g[nm] = np.exp(lF + qs[t] * s) - 1
    g.insert(0, "level", lab); g["backtest_within_20pct"] = round(float(hit[lab]), 3); out.append(g.drop(columns=["v50l", "v50h"]))
O = pd.concat(out, ignore_index=True); O = O[["level", "group", "n", "P10", "P25", "F", "P75", "P90", "backtest_within_20pct"]].rename(columns={"F": "P50", "n": "items"})
assert (O.P10 <= O.P25).all() and (O.P25 <= O.P50).all() and (O.P50 <= O.P75).all() and (O.P75 <= O.P90).all()
U = ["P10", "P25", "P50", "P75", "P90"]; O[U] = O[U].round(1); print(O[O.level.isin(["product group", "total"])].to_string(index=False)); print("rows by level:", O.level.value_counts().to_dict())
print("median range as a share of P50 by level:", O.assign(w50=(O.P75 - O.P25) / O.P50, w80=(O.P90 - O.P10) / O.P50).groupby("level")[["w50", "w80"]].median().round(3).to_dict())
hdr = ["# D371b (fork): July-October 2026 ranges for totals of the candidate system (nx5 P50, data/system_q2575_2026.parquet), studentised conformal rule of D371.",
       f"# spread s = sqrt(s_items^2 + c^2): item half-ranges (fork D329 ratios) added in quadrature relative to the total, plus the shared origin term c = {c:.4f} (RMS of the all-item total's log miss, 13 backtest origins).",
       f"# bounds = exp(log(P50 + 1) + q x s) - 1 with q = {qs[0.10]:.3f} / {qs[0.25]:.3f} / {qs[0.75]:.3f} / {qs[0.90]:.3f} (P10 / P25 / P75 / P90), the percentiles of log miss / s over all backtest groups.",
       "# backtest (D371, factors left one origin out, recent 9 origins): P25-P75 inside 57% product group, 57% product type, 50% look; P10-P90 91% / 85% / 78%. Workbook hand rules (World Cup cap, out of stock) are not applied here."]
with open("data/fk_total_ranges_2026.csv", "w", encoding="utf-8", newline="") as fh: fh.write("\n".join(hdr) + "\n"); O.to_csv(fh, index=False)
J["total_ranges_2026"] = dict(c=c, q=qs, rows=len(O)); json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0, default=float); print("written data/fk_total_ranges_2026.csv", len(O))
