"""Fork 2 (user, 18 Nov: "put it to the model explanation. but it should be a column for each of the model types (R3, seasonal, each model separately,
and the submission system) the color should pertain to absolute values where all metrics are aligned to a 'good' direction. in the value, put the
value and the ratio to system in paranthases like <abs_value>(ratio to sys)").
Scored exactly as the builder's pipeline/232_metrics_table.py (pipeline/159's construction and item-origins, pipeline/score_lib.py, the nine recent month-ends):
columns SYS (scoreboard nx9_geo = the submitted numbers), R3, SN (seasonal naive), and the six models of the system each alone: AR (the agents'
rules), LightGBM (the routed tree: the launch-weighted version for items under six months), the neural network, TabPFN calibrated, TabPFN, the
recommender (members from data/dec_base_nx5r26.parquet, the backtest the system's corrections were fitted on).
Every metric is lower-is-better (hit rates and R2 as complements: share outside +/-20%, 1 - R2; bias absolute). Cell = "value (value / SYS)".
Colour per row by the value: the row's best = deepest blue, worst = deepest red, on a log scale between them, interpolated continuously through seven RdBu anchors; text white on dark fills; SYS's cell carries only
its value. Buying loss exists for the simulated systems only (SYS, R3, SN, AR). Writes data/model_metrics_table.csv and a preview HTML."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
src = open("pipeline/232_metrics_table.py", encoding="utf-8").read().split("V = pd.DataFrame(")[0]
exec(src)   # Z (recent item-origins, scoreboard columns), y, per(), sl, DZ
B = pd.read_parquet("data/dec_base_nx5r26.parquet"); B = B[B.y.notna()].set_index(["origin", "style_color"])
K = pd.MultiIndex.from_arrays([Z.origin, Z.style_color])
MEM = {"AR": "rules", "LightGBM": "qdyn", "Neural network": "nn", "TabPFN calibrated": "pf2c", "TabPFN": "pf2d32", "Recommender": "rc"}
for lab, c in MEM.items(): Z[f"m_{c}"] = B[c].reindex(K).values
print("members missing on the recent rows:", {lab: int(np.isnan(Z[f'm_{c}'].values.astype(float)).sum()) for lab, c in MEM.items()})
print("AR from the base table vs the scoreboard's rules column, max rel diff:", float(np.nanmax(np.abs(Z["m_rules"] / Z["rules"] - 1))))
COLS = {"SYS": "nx9_geo", "R3": "R3", "SN": "sea_naive", **{lab: f"m_{c}" for lab, c in MEM.items()}}
# the members can be missing on a few rows (an item a member did not cover): score every column on the rows all of them have
ok = np.ones(len(Z), bool)
for c in COLS.values(): ok &= np.isfinite(Z[c].values.astype(float))
print("rows used:", int(ok.sum()), "of", len(Z))
Z = Z[ok].reset_index(drop=True); DZ = {k: v[ok] for k, v in DZ.items()}; y = Z.y.values
V = pd.DataFrame({k: per(c) for k, c in COLS.items()})
sim = pd.read_csv("data/sim_nx11_windows.csv"); L = sim.groupby("system").loss_pct.mean()
V.loc["Buying loss (% of perfect-foresight profit)"] = [L.get("nx11"), L.get("R3"), L.get("sea_naive"), L.get("rules")] + [np.nan] * 5
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
val = lambda x, kind: (1 - x) if kind == "comp" else (abs(x) if kind == "abs" else x)
SCALE = ["2166AC", "67A9CF", "D1E5F0", "F7F7F7", "FDDBC7", "EF8A62", "B2182B"]   # best -> worst
def shade(t):   # continuous: interpolate between the seven anchors (best 0 -> worst 1), so every value gets its own shade
    t = min(max(t, 0.0), 1.0) * (len(SCALE) - 1); i = min(int(t), len(SCALE) - 2); f = t - i
    a, b = [tuple(int(h[j:j + 2], 16) for j in (0, 2, 4)) for h in (SCALE[i], SCALE[i + 1])]
    return "".join(f"{round(a[j] + (b[j] - a[j]) * f):02X}" for j in range(3))
fmt = lambda x, name: "–" if not np.isfinite(x) else (f"{x:.2f}" if name.startswith("Unexplained") else f"{x:.1%}")
# user, after the first preview: "you can just use the ratio as the value for the table" / "whatever you think is better visually".
# SYS's column shows its own value; every other column shows its ratio to SYS (value / SYS, > 1 = worse than SYS), coloured on ONE scale shared by
# all rows: log ratio, white at 1, blue below (better than SYS), red above, full colour at 3x (or 1/3x).
DIV = ["2166AC", "67A9CF", "D1E5F0", "F7F7F7", "FDDBC7", "EF8A62", "B2182B"]
def shade_ratio(q):
    t = 0.5 + 0.5 * max(-1.0, min(1.0, np.log(q) / np.log(3.0)))     # 1/3x -> 0, 1x -> 0.5, 3x -> 1
    t *= (len(DIV) - 1); i = min(int(t), len(DIV) - 2); f = t - i
    a, b = [tuple(int(h[j:j + 2], 16) for j in (0, 2, 4)) for h in (DIV[i], DIV[i + 1])]
    return "".join(f"{round(a[j] + (b[j] - a[j]) * f):02X}" for j in range(3))
out = []
for name, key, kind in ROWS:
    v = {k: val(float(V.loc[key, k]), kind) for k in COLS}; s_ = v["SYS"]
    r = {"Metric (lower is better)": name, "SYS": fmt(s_, name), "fill SYS": "FFFFFF", "value SYS": round(s_, 5)}
    for k in list(COLS)[1:]:
        x = v[k]
        if not np.isfinite(x): r[k] = "–"; r[f"fill {k}"] = "FFFFFF"; continue
        q = x / s_; r[k] = f"{q:.2f}"; r[f"fill {k}"] = shade_ratio(q); r[f"value {k}"] = round(x, 5); r[f"ratio {k}"] = round(q, 4)
    out.append(r)
D = pd.DataFrame(out); D.to_csv("data/model_metrics_table.csv", index=False)
print(D[["Metric (lower is better)"] + list(COLS)].to_string(index=False))
H = ["<table style='border-collapse:collapse;font:12px sans-serif'><tr>" + "".join(f"<th style='padding:4px 6px;border:1px solid #ccc'>{c}</th>" for c in ["Metric (lower is better)", "SYS (value)"] + [f"{c} / SYS" for c in list(COLS)[1:]]) + "</tr>"]
for r in out:
    tds = f"<td style='padding:4px 6px;border:1px solid #ccc'>{r['Metric (lower is better)']}</td>"
    for k in COLS:
        f = r[f"fill {k}"]; lum = 0.299 * int(f[0:2], 16) + 0.587 * int(f[2:4], 16) + 0.114 * int(f[4:6], 16); col = "#fff" if lum < 140 else "#000"
        tds += f"<td style='padding:4px 6px;border:1px solid #ccc;text-align:right;background:#{f};color:{col}'>{r[k]}</td>"
    H.append(f"<tr>{tds}</tr>")
open("archive/eda/drafts/model_metrics_table_preview.html", "w", encoding="utf-8").write("<!doctype html><meta charset='utf-8'>" + "".join(H) + "</table>")
