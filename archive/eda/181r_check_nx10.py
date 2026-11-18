"""D424: checks the nx10 workbook (eda/181r_final_nx10_2026.py) against nx9 (D422). nx10 is a text change only, so:
 1 Numbers: every cell of the Forecast sheet (numbers and explanations) equals nx9's; the cascade, hierarchy and totals-range tables equal nx9's.
 2 Integrity: the sheets Edikted provided untouched, pictures identical.
 3 Text: no unfilled placeholder, the three boxes under Excel's 409 pt row limit, no "hand rules" left for the agents' rules, the old month-1
   figure (0.395 to 0.369) gone, Fork 2's repairs A1-A7 and roadmap B1-B2 (D413) present.
Run from the lab root: python eda/181r_check_nx10.py"""
import re, sys, zipfile, hashlib, pandas as pd, openpyxl
N9, N10 = "candidates/Edikted_Forecasting_Exercise_candidate_nx9.xlsx", "candidates/Edikted_Forecasting_Exercise_candidate_nx10.xlsx"
w9, w10 = openpyxl.load_workbook(N9), openpyxl.load_workbook(N10); ok = True
# 1 numbers
diff = [(c.coordinate, c.value, w10["Forecast"][c.coordinate].value) for r in w9["Forecast"].iter_rows() for c in r if c.value != w10["Forecast"][c.coordinate].value]
same_dims = (w9["Forecast"].max_row, w9["Forecast"].max_column) == (w10["Forecast"].max_row, w10["Forecast"].max_column)
print("1 Forecast sheet: cells differing from nx9:", len(diff), diff[:3], "| same size:", same_dims); ok &= not diff and same_dims
for t in ["cascade", "cascade_hierarchy", "total_ranges"]:
    a, b = pd.read_csv(f"data/final_2026_nx9_{t}.csv"), pd.read_csv(f"data/final_2026_nx10_{t}.csv"); e = a.equals(b); print(f"  {t} identical to nx9:", e); ok &= e
# 2 integrity
wt = openpyxl.load_workbook("Edikted_Forecasting_Exercise.xlsx")
untouched = {n: sum(1 for r in wt[n].iter_rows() for c in r if c.value != w10[n][c.coordinate].value) for n in ["Read me first", "Item gallery", "Data dictionary", "Planned campaigns"]}
med = lambda p: sorted(hashlib.md5(zipfile.ZipFile(p).read(n)).hexdigest() for n in zipfile.ZipFile(p).namelist() if n.startswith("xl/media/"))
pics = med("Edikted_Forecasting_Exercise.xlsx") == med(N10); print("2 sheets from Edikted, cells changed:", untouched, "| pictures identical:", pics, "| sheets:", w10.sheetnames == w9.sheetnames); ok &= not any(untouched.values()) and pics and w10.sheetnames == w9.sheetnames
# 3 text
me = w10["Model explanation"]; boxes = {r[0].value: (len(r[1].value), me.row_dimensions[r[0].row].height) for r in me.iter_rows(min_row=2, max_row=4)}
md = w10["Method and decisions"]; rows = [(r[0].value, r[1].value, r[2].value) for r in md.iter_rows(min_row=3) if r[1].value]
txt = "\n".join(str(c.value) for n in ["Model explanation", "Method and decisions"] for r in w10[n].iter_rows() for c in r if c.value)
braces = "{" in txt or "}" in txt; print("3 boxes (characters, row height):", boxes, "| unfilled placeholders:", braces, "| decision rows: nx9", sum(1 for r in w9["Method and decisions"].iter_rows(min_row=3) if r[1].value), "-> nx10", len(rows))
ok &= not braces and all(h < 409 for _, h in boxes.values())
bad = [p for p in ["hand rules", "Hand-written planning rules", "hand-written planning rules", "0.395 to 0.369", "What would improve it most"] if p in txt]; print("  old wording left:", bad); ok &= not bad
need = {"A1": "planning rules written by AI agents", "A2": "Planning rules written by AI agents", "A3": "the agents' rules", "A4": "Besides WAPE (volume-weighted)", "A5": "arriving 21 days later",
        "A6": "Explanations checked against the model", "A7": "14 fixed 0-10 fashion fields", "B1": "What would improve it, by measured value", "B2": "Compute last"}
miss = [k for k, p in need.items() if p not in txt]; print("  repairs present:", [k for k in need if k not in miss], "| missing:", miss); ok &= not miss
road = [r[1] for r in rows if r[0] == "Roadmap"]; print("  roadmap rows:", road); ok &= len(road) == 6
print("CHECK", "PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)
