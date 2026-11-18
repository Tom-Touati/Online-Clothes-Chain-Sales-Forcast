"""Fork 2 (user: "can we just show a colored table that shows the sys ratio of each metric with each one of the other systems? (write and color by
the ratio, add the system metric values to the columns)" and "i want to put it in the model explanation").
Input: data/metrics_table_nx12.csv (the builder session's D431 table, scored exactly as data/scoreboard.json; nx12 = nx13 numbers): SYS, R3, SN, AR
on the nine recent backtest month-ends. Output: data/metric_ratio_table.csv, one row per metric:
  Metric | Better when | SYS | R3 | SYS / R3 | SN | SYS / SN | AR | SYS / AR | colour of each ratio cell
Ratio = SYS's value / the baseline's value (bias: of the absolute bias). Colour by how much better SYS is: the ratio for lower-is-better
metrics, its inverse for higher-is-better ones; green when SYS is better, red when worse, deeper with the size (five steps, as the hex fills).
Also writes archive/eda/drafts/metric_ratio_table_preview.html for a look before the builder puts it in the workbook."""
import numpy as np, pandas as pd
T = pd.read_csv("data/metrics_table_nx12.csv", index_col=0)
ROWS = [("Volume-weighted error (WAPE, share of units)", "WAPE (error / units)", "lower", False),
        ("Per-item error (median, every item counts once)", "Median error per item", "lower", False),
        ("Items within +/-20%", "Items within +/-20%", "higher", False),
        ("Units within +/-20%", "Units within +/-20%", "higher", False),
        ("Bias (absolute, forecast / actual - 1)", "Bias (forecast / actual - 1)", "lower", True),
        ("Error plus bias ((MAE + |bias|) / units)", "(MAE + |bias|) / actual units", "lower", False),
        ("Variance explained (R2, units)", "R2 on units", "higher", False),
        ("Variance explained (R2, log units)", "R2 on log units", "higher", False),
        ("Range accuracy (pinball loss P10-P90, / units)", "Pinball loss, P10-P90 (/ units)", "lower", False),
        ("Buying simulation (profit lost vs perfect foresight)", "Buying loss (% of perfect-foresight profit)", "lower", False)]
BASE = [("R3", "R3"), ("SN", "Seasonal naive"), ("AR", "Agents' rules")]
def fill(g):   # g = how many times better SYS is (>1 better); five steps each way
    if g >= 2.0: return "1E8449"
    if g >= 1.25: return "52BE80"
    if g >= 1.05: return "A9DFBF"
    if g > 1 / 1.05: return "F2F3F4"
    if g > 1 / 1.25: return "F5B7B1"
    return "E74C3C"
rows = []
for name, key, better, absval in ROWS:
    v = T.loc[key]; sys_ = abs(v["System"]) if absval else v["System"]
    r = {"Metric": name, "Better when": better, "SYS": round(float(v["System"]), 4)}
    for ab, col in BASE:
        b = abs(v[col]) if absval else v[col]; ratio = sys_ / b
        g = 1 / ratio if better == "lower" else ratio
        r[ab] = round(float(v[col]), 4); r[f"SYS / {ab}"] = round(float(ratio), 3); r[f"fill SYS / {ab}"] = fill(g)
    rows.append(r)
D = pd.DataFrame(rows); D.to_csv("data/metric_ratio_table.csv", index=False)
print(D.drop(columns=[c for c in D.columns if c.startswith("fill")]).to_string(index=False))
pct = lambda x, k: f"{x:.1%}" if k not in ("Variance explained (R2, units)", "Variance explained (R2, log units)") else f"{x:.2f}"
h = ["<table style='border-collapse:collapse;font:13px sans-serif'><tr>" + "".join(f"<th style='padding:4px 8px;border:1px solid #ccc'>{c}</th>" for c in ["Metric", "Better when", "SYS", "R3", "SYS / R3", "SN", "SYS / SN", "AR", "SYS / AR"]) + "</tr>"]
for r in rows:
    cells = [r["Metric"], r["Better when"], pct(r["SYS"], r["Metric"])]
    tds = "".join(f"<td style='padding:4px 8px;border:1px solid #ccc'>{c}</td>" for c in cells)
    for ab, _ in BASE:
        tds += f"<td style='padding:4px 8px;border:1px solid #ccc;text-align:right'>{pct(r[ab], r['Metric'])}</td><td style='padding:4px 8px;border:1px solid #ccc;text-align:right;background:#{r[f'fill SYS / {ab}']}'><b>{r[f'SYS / {ab}']:.2f}</b></td>"
    h.append(f"<tr>{tds}</tr>")
open("archive/eda/drafts/metric_ratio_table_preview.html", "w", encoding="utf-8").write("<!doctype html><meta charset='utf-8'>" + "".join(h) + "</table>")
