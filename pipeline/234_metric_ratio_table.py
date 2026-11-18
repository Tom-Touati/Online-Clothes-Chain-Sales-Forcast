"""Fork 2 (user: "can we just show a colored table that shows the sys ratio of each metric with each one of the other systems? (write and color by
the ratio, add the system metric values to the columns)" and "i want to put it in the model explanation").
Input: data/metrics_table_nx12.csv (the builder session's D431 table, scored exactly as data/scoreboard.json; nx12 = nx13 numbers): SYS, R3, SN, AR
on the nine recent backtest month-ends. Output: data/metric_ratio_table.csv, one row per metric:
  Metric | SYS | R3 | SYS / R3 | SN | SYS / SN | AR | SYS / AR | fill of each ratio cell
Every metric is lower-is-better (hit rates and R2 as their complements: share outside +/-20%, unexplained variance 1 - R2; bias absolute),
so ratio = SYS / baseline < 1 means SYS is better. Fill: cool (blue) below 1, warm (red) above, deeper with distance from 1, grey within 5%.
Also writes archive/eda/drafts/metric_ratio_table_preview.html for a look before the builder puts it in the workbook."""
import numpy as np, pandas as pd
T = pd.read_csv("data/metrics_table_nx12.csv", index_col=0)
# user (second message): "all metrics just need to be aligned to the same direction (negative sign where needed), and use a cool warm".
# A sign flip leaves a ratio unchanged, so the higher-is-better metrics are turned into their complements: every row is now lower-is-better
# and SYS / baseline < 1 means SYS is better. kind: "raw" value, "abs" absolute value, "comp" 1 - value.
ROWS = [("Volume-weighted error (WAPE, share of units)", "WAPE (error / units)", "raw"),
        ("Per-item error (median, every item counts once)", "Median error per item", "raw"),
        ("Items outside +/-20%", "Items within +/-20%", "comp"),
        ("Units outside +/-20%", "Units within +/-20%", "comp"),
        ("Absolute bias (|forecast / actual - 1|)", "Bias (forecast / actual - 1)", "abs"),
        ("Error plus bias ((MAE + |bias|) / units)", "(MAE + |bias|) / actual units", "raw"),
        ("Unexplained variance (1 - R2, units)", "R2 on units", "comp"),
        ("Unexplained variance (1 - R2, log units)", "R2 on log units", "comp"),
        ("Range error (pinball loss P10-P90, / units)", "Pinball loss, P10-P90 (/ units)", "raw"),
        ("Buying loss (profit lost vs perfect foresight)", "Buying loss (% of perfect-foresight profit)", "raw")]
BASE = [("R3", "R3"), ("SN", "Seasonal naive"), ("AR", "Agents' rules")]
def fill(ratio):   # cool below 1 (SYS better), warm above (SYS worse), deeper with distance from 1; grey within 5%
    if ratio <= 0.5: return "2166AC"
    if ratio <= 0.8: return "67A9CF"
    if ratio < 0.95: return "D1E5F0"
    if ratio <= 1.05: return "F2F3F4"
    if ratio < 1.25: return "FDDBC7"
    if ratio < 2.0: return "EF8A62"
    return "B2182B"
val = lambda x, kind: (1 - x) if kind == "comp" else (abs(x) if kind == "abs" else x)
rows = []
for name, key, kind in ROWS:
    v = T.loc[key]; sys_ = val(v["System"], kind)
    r = {"Metric": name, "SYS": round(float(sys_), 4)}
    for ab, col in BASE:
        b = val(v[col], kind); ratio = sys_ / b
        r[ab] = round(float(b), 4); r[f"SYS / {ab}"] = round(float(ratio), 3); r[f"fill SYS / {ab}"] = fill(ratio)
    rows.append(r)
D = pd.DataFrame(rows); D.to_csv("data/metric_ratio_table.csv", index=False)
print(D.drop(columns=[c for c in D.columns if c.startswith("fill")]).to_string(index=False))
pct = lambda x, k: f"{x:.2f}" if k.startswith("Unexplained") else f"{x:.1%}"
h = ["<table style='border-collapse:collapse;font:13px sans-serif'><tr>" + "".join(f"<th style='padding:4px 8px;border:1px solid #ccc'>{c}</th>" for c in ["Metric (lower is better)", "SYS", "R3", "SYS / R3", "SN", "SYS / SN", "AR", "SYS / AR"]) + "</tr>"]
for r in rows:
    cells = [r["Metric"], pct(r["SYS"], r["Metric"])]
    tds = "".join(f"<td style='padding:4px 8px;border:1px solid #ccc'>{c}</td>" for c in cells)
    for ab, _ in BASE:
        tds += f"<td style='padding:4px 8px;border:1px solid #ccc;text-align:right'>{pct(r[ab], r['Metric'])}</td><td style='padding:4px 8px;border:1px solid #ccc;text-align:right;background:#{r[f'fill SYS / {ab}']};color:{'#fff' if r[f'fill SYS / {ab}'] in ('2166AC','B2182B') else '#000'}'><b>{r[f'SYS / {ab}']:.2f}</b></td>"
    h.append(f"<tr>{tds}</tr>")
open("archive/eda/drafts/metric_ratio_table_preview.html", "w", encoding="utf-8").write("<!doctype html><meta charset='utf-8'>" + "".join(h) + "</table>")
