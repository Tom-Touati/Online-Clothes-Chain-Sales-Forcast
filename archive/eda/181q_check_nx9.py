"""D422: checks the nx9 workbook (eda/181q_final_nx9_2026.py).
 1 Regression: the same builder, pointed at nx8's correction and range files, must reproduce the nx8 workbook's Forecast numbers and cascade
   exactly (so the only differences between nx9 and nx8 come from the two inputs that changed).
 2 The change: each modelled item's model forecast moved by exactly the look-alike multiplier (geo / geo_current in
   data/geo_nx5r_fash_2026.parquet); the four hand-ruled rows did not move.
 3 Integrity: the sheets Edikted provided untouched, pictures identical, every number quoted in an explanation is the number in its row.
Run from the lab root: python eda/181q_check_nx9.py"""
import os, re, subprocess, sys, zipfile, hashlib, numpy as np, pandas as pd, openpyxl
PY = sys.executable; NUM = ["Jul-2026 units", "Aug-2026 units", "Sep-2026 units", "Oct-2026 units", "Total Jul-Oct", "Confidence (1-3)", "Model raw total Jul-Oct (optional, before your adjustments)"]
def forecast(path):
    wb = openpyxl.load_workbook(path); ws = wb["Forecast"]; H = {c.value: c.column for c in ws[1]}
    return pd.DataFrame([{**{"style_color": r[1].value, "explanation": ws.cell(row=r[0].row, column=H["Explanation (at least 15 items of your choice)"]).value}, **{h: ws.cell(row=r[0].row, column=H[h]).value for h in NUM}} for r in ws.iter_rows(min_row=2) if r[1].value]).set_index("style_color"), wb
ok = True
# 1 regression against nx8
src = open("eda/181q_final_nx9_2026.py", encoding="utf-8").read()
for a, b in [('DATA / "geo_nx5r_fash_2026.parquet"', 'DATA / "geo_nx5r_x_2026.parquet"'), ('DATA / "ranges_nx5r_fash_2026.csv"', 'DATA / "ranges_nx5r_2026.csv"'), ("candidates/Edikted_Forecasting_Exercise_candidate_nx9.xlsx", "_tmp_nx9_regression.xlsx"),
             ("final_2026_nx9_cascade.csv", "_tmp_nx9_regression_cascade.csv"), ("final_2026_nx9_cascade_hierarchy.csv", "_tmp_nx9_regression_hierarchy.csv"), ("final_2026_nx9_total_ranges.csv", "_tmp_nx9_regression_total_ranges.csv")]:
    assert src.count(a) == 1, a; src = src.replace(a, b)
open("eda/_tmp_nx9_regression.py", "w", encoding="utf-8").write(src)
r = subprocess.run([PY, "eda/_tmp_nx9_regression.py"], capture_output=True, text=True, encoding="utf-8", errors="replace"); assert r.returncode == 0, r.stderr[-1500:]
f8, _ = forecast("candidates/Edikted_Forecasting_Exercise_candidate_nx8.xlsx"); fr, _ = forecast("_tmp_nx9_regression.xlsx")
d = int((f8[NUM].values != fr[NUM].values).sum()); print("1 regression, Forecast number cells differing from nx8 with nx8's inputs:", d); ok &= d == 0
c8, cr = pd.read_csv("data/final_2026_nx8_cascade.csv", index_col=0), pd.read_csv("data/_tmp_nx9_regression_cascade.csv", index_col=0)
shared = [c for c in c8.columns if c in cr.columns and pd.api.types.is_numeric_dtype(c8[c])]
mx = max(float(np.nanmax(np.abs(c8[c].values.astype(float) - cr[c].values.astype(float)))) for c in shared); print("  cascade:", len(shared), "shared numeric columns, max difference", mx); ok &= mx < 1e-9
t8, tr = pd.read_csv("data/final_2026_nx8_total_ranges.csv"), pd.read_csv("data/_tmp_nx9_regression_total_ranges.csv"); print("  totals ranges identical:", t8.equals(tr)); ok &= t8.equals(tr)
same_expl = int((f8.explanation.fillna("") == fr.explanation.fillna("")).sum()); print("  explanations identical to nx8:", same_expl, "of 100 (the three that quote the launch-weighted tree's lift are computed differently now)")
for f in ["eda/_tmp_nx9_regression.py", "_tmp_nx9_regression.xlsx", "data/_tmp_nx9_regression_cascade.csv", "data/_tmp_nx9_regression_hierarchy.csv", "data/_tmp_nx9_regression_total_ranges.csv"]:
    if os.path.exists(f): os.remove(f)
# 2 the change
f9, wb9 = forecast("candidates/Edikted_Forecasting_Exercise_candidate_nx9.xlsx"); c9 = pd.read_csv("data/final_2026_nx9_cascade.csv", index_col=0)
g = pd.read_parquet("data/geo_nx5r_fash_2026.parquet").set_index("style_color").reindex(c9.index); mult = g.geo / g.geo_current
HAND = ["S24840_LIGHT-GRAY-MELANGE", "S25039_YELLOW", "S24847_NAVY", "S16079_DARK-GRAY"]; modelled = ~c9.index.isin(["S16079_DARK-GRAY"])
err = float(np.nanmax(np.abs(c9.forecast[modelled] / c8.forecast[modelled] - mult[modelled]))); print("2 model forecast moved by the look-alike multiplier: max deviation", err, "| multiplier on the targets min / median / max:", round(float(mult[modelled].min()), 4), round(float(mult[modelled].median()), 4), round(float(mult[modelled].max()), 4)); ok &= err < 1e-9
hand_same = bool((f9.loc[HAND, NUM[:5]].values == f8.loc[HAND, NUM[:5]].values).all()); print("  hand-ruled rows unchanged (months and totals):", hand_same); ok &= hand_same
chg = f9["Total Jul-Oct"] / f8["Total Jul-Oct"] - 1; print("  totals: nx8", int(f8["Total Jul-Oct"].sum()), "-> nx9", int(f9["Total Jul-Oct"].sum()), f"({f9['Total Jul-Oct'].sum() / f8['Total Jul-Oct'].sum() - 1:+.2%}); items {chg.min():+.1%} to {chg.max():+.1%}; confidence", f9["Confidence (1-3)"].value_counts().sort_index().to_dict(), "changed for", int((f9["Confidence (1-3)"] != f8["Confidence (1-3)"]).sum()), "items")
ok &= all(isinstance(v, int) and v >= 0 for h in NUM[:4] for v in f9[h]) and bool((f9[NUM[:4]].sum(axis=1) == f9["Total Jul-Oct"]).all()) and set(f9["Confidence (1-3)"]) <= {1, 2, 3}
# 3 integrity
wt = openpyxl.load_workbook("Edikted_Forecasting_Exercise.xlsx")
untouched = {n: sum(1 for r in wt[n].iter_rows() for c in r if c.value != wb9[n][c.coordinate].value) for n in ["Read me first", "Item gallery", "Data dictionary", "Planned campaigns"]}
med = lambda p: sorted(hashlib.md5(zipfile.ZipFile(p).read(n)).hexdigest() for n in zipfile.ZipFile(p).namelist() if n.startswith("xl/media/"))
pics = med("Edikted_Forecasting_Exercise.xlsx") == med("candidates/Edikted_Forecasting_Exercise_candidate_nx9.xlsx"); print("3 sheets from Edikted, cells changed:", untouched, "| pictures identical:", pics, "| sheets:", wb9.sheetnames); ok &= not any(untouched.values()) and pics
issues = []
for s, r in f9[f9.explanation.notna()].iterrows():
    e, raw, tot = r.explanation, r[NUM[6]], r["Total Jul-Oct"]
    m = re.search(r"Adjusted .*?\), to ([\d,]+)", e) if "Adjusted" in e else None
    if "Adjusted" in e and (not m or int(m.group(1).replace(",", "")) != tot): issues.append((s, "adjusted total"))
    if "Not adjusted" in e and abs(tot - raw) > 0.005 * raw + 2: issues.append((s, "says not adjusted"))
    if re.search(r"\{|\}|nan|D\d{3}", e): issues.append((s, "placeholder / nan / internal id"))
    if "confidence 1" in e and r["Confidence (1-3)"] != 1: issues.append((s, "confidence 1 claimed"))
    if s not in HAND and f"{raw:,}" not in e: issues.append((s, "raw total not quoted"))
n_expl = int(f9.explanation.notna().sum()); print("  explanations:", n_expl, "| issues:", issues); ok &= not issues and n_expl >= 15
me = wb9["Model explanation"]; boxes = {r[0].value: (len(r[1].value), me.row_dimensions[r[0].row].height) for r in me.iter_rows(min_row=2, max_row=4)}
braces = any("{" in str(c.value) for n in ["Model explanation", "Method and decisions"] for r in wb9[n].iter_rows() for c in r if c.value); print("  boxes (characters, row height):", boxes, "| unfilled placeholders:", braces, "| decision rows:", wb9["Method and decisions"].max_row - 2); ok &= not braces and all(h < 409 for _, h in boxes.values())
print("  lift quoted for the launches:", {s: re.search(r"including a (\d+%) lift", e).group(1) for s, e in f9.explanation.dropna().items() if "lift" in e}, "|", re.search(r"For the \d+ targets under six months, the launch-weighted tree raises the six-model average by about \d+%", me["B4"].value).group(0))
print("CHECK", "PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)
