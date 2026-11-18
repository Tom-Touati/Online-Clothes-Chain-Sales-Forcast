"""D421: checks the consolidated builder (eda/181p_final_nx8_2026.py) against the workbook the user approved for its numbers, nx7c
(eda/181o_final_nx7c_2026.py): every numeric cell of the Forecast sheet, every shared column of the cascade table and the totals ranges must
be identical; the sheets Edikted provided must be untouched; every number quoted in an explanation must be the number in its row."""
import re, sys, zipfile, hashlib, numpy as np, pandas as pd, openpyxl
A, B, T = "Edikted_Forecasting_Exercise_candidate_nx7c.xlsx", "Edikted_Forecasting_Exercise_candidate_nx8.xlsx", "Edikted_Forecasting_Exercise.xlsx"
wa, wb, wt = openpyxl.load_workbook(A), openpyxl.load_workbook(B), openpyxl.load_workbook(T)
a, b = wa["Forecast"], wb["Forecast"]; H = {c.value: c.column for c in b[1]}; EC = H["Explanation (at least 15 items of your choice)"]
bad = [(r, k) for r in range(1, 102) for k in range(1, b.max_column + 1) if k != EC and a.cell(row=r, column=k).value != b.cell(row=r, column=k).value]
print("Forecast sheet, cells differing from nx7c outside the explanation column:", len(bad), bad[:5])
print("sheets from Edikted, cells differing from the template:", {n: sum(1 for r in wt[n].iter_rows() for c in r if c.value != wb[n][c.coordinate].value) for n in ["Read me first", "Item gallery", "Data dictionary", "Planned campaigns"]}, "| gallery images", len(wb["Item gallery"]._images))
za, zb = zipfile.ZipFile(T), zipfile.ZipFile(B); med = lambda z: sorted(hashlib.md5(z.read(n)).hexdigest() for n in z.namelist() if n.startswith("xl/media/"))
print("image files identical to the template:", med(za) == med(zb), len(med(zb)), "| zip ok:", zb.testzip() is None, "| sheets:", wb.sheetnames)
ca, cb = pd.read_csv("data/final_2026_nx7c_cascade.csv", index_col=0), pd.read_csv("data/final_2026_nx8_cascade.csv", index_col=0)
assert list(ca.index) == list(cb.index)
shared = [c for c in cb.columns if c in ca.columns and pd.api.types.is_numeric_dtype(cb[c]) and c != "buying"]
diff = {c: float(np.nanmax(np.abs(ca[c].values.astype(float) - cb[c].values.astype(float)))) for c in shared}; nan_mismatch = [c for c in shared if (ca[c].isna() != cb[c].isna()).any()]
print("cascade: shared numeric columns", len(shared), "| max abs difference", max(diff.values()), "| columns differing:", {c: v for c, v in diff.items() if v > 1e-9}, "| NaN pattern differs:", nan_mismatch)
ta, tb = pd.read_csv("data/final_2026_nx7c_total_ranges.csv"), pd.read_csv("data/final_2026_nx8_total_ranges.csv"); print("totals ranges identical:", ta.equals(tb), ta.shape)
rows = [r for r in b.iter_rows(min_row=2) if r[1].value]; g = lambda r, h: b.cell(row=r[0].row, column=H[h]).value
ex = [(r[1].value, g(r, "Explanation (at least 15 items of your choice)"), g(r, "Model raw total Jul-Oct (optional, before your adjustments)"), g(r, "Total Jul-Oct"), g(r, "Jul-2026 units"), g(r, "Oct-2026 units"), g(r, "Confidence (1-3)")) for r in rows]
ex = [e for e in ex if e[1]]; issues = []
for s, e, raw, tot, jul, oct_, conf in ex:
    m = re.search(r"Adjusted .*?\), to ([\d,]+)", e) if "Adjusted" in e else None
    if "Adjusted" in e and (not m or int(m.group(1).replace(",", "")) != tot): issues.append((s, "adjusted total", m and m.group(1), tot))
    if "Not adjusted" in e and abs(tot - raw) > 0.005 * raw + 2: issues.append((s, "says not adjusted", raw, tot))
    if re.search(r"\{|\}|nan|D\d{3}", e): issues.append((s, "placeholder / nan / internal id"))
    if "confidence 1" in e and conf != 1: issues.append((s, "confidence 1 claimed", conf))
    if "Confidence 3" in e and conf != 3: issues.append((s, "confidence 3 claimed", conf))
    if s not in ("S24840_LIGHT-GRAY-MELANGE", "S25039_YELLOW", "S24847_NAVY", "S16079_DARK-GRAY") and f"{raw:,}" not in e: issues.append((s, "raw total not quoted", raw))
print("explanations:", len(ex), "| issues:", issues, "| words min/median/max:", sorted(len(e[1].split()) for e in ex)[::max(1, len(ex) // 2)][:3])
me = wb["Model explanation"]; print("boxes (chars, row height):", {r[0].value: (len(r[1].value), me.row_dimensions[r[0].row].height) for r in me.iter_rows(min_row=2, max_row=4)}, "| braces left:", any("{" in (r[1].value or "") for r in me.iter_rows(min_row=2, max_row=4)))
md = wb["Method and decisions"]; print("decision rows:", md.max_row - 2, "| braces left:", any("{" in str(c.value) for r in md.iter_rows(min_row=3) for c in r))
print("totals nx8:", sum(e for e in [g(r, "Total Jul-Oct") for r in rows]), "| confidence:", pd.Series([g(r, "Confidence (1-3)") for r in rows]).value_counts().sort_index().to_dict())
ok = not bad and max(diff.values()) < 1e-9 and not nan_mismatch and ta.equals(tb) and not issues
print("CHECK", "PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)
