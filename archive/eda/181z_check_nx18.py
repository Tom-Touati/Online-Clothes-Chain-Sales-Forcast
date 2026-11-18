"""D431: checks the nx18 workbook (eda/181z_final_nx18_2026.py) against nx11 (D429), with nx11's checks plus nx18's: every box line opens
with a bold summary, the box text is black and upright, the decisions sheet is uniformly aligned. nx11 header follows.
D429: checks the nx11 workbook (eda/181s_final_nx11_2026.py) against nx10 (D424). nx11 is a text change only, so:
 1 Numbers: every cell of the Forecast sheet (numbers and item explanations) equals nx10's; the cascade, hierarchy and totals-range tables
   equal nx10's, and so do the three cascade sheets of the workbook.
 2 Integrity: the sheets Edikted provided untouched, pictures identical, the same sheets in the same order.
 3 Text: no unfilled placeholder; the three boxes under Excel's 409 pt row limit at the widened column (120); the "Method and decisions"
   areas in the order Framing, Models, Information, Judgement, Metrics, Evaluation, Explainability, Roadmap; the nine Metrics rows, the
   cascade row first under Explainability, the two renamed Information rows, the two covered Evaluation rows gone; no "hand rules" left for
   the agents' rules.
Run from the lab root: python eda/181z_check_nx18.py"""
import sys, zipfile, hashlib, pandas as pd, openpyxl
N10, N11 = "candidates/Edikted_Forecasting_Exercise_candidate_nx11.xlsx", "candidates/Edikted_Forecasting_Exercise_candidate_nx18.xlsx"
w10, w11 = openpyxl.load_workbook(N10), openpyxl.load_workbook(N11); ok = True
# 1 numbers
for sh in ["Forecast", "Cascade (items)", "Cascade (hierarchy)", "Ranges (totals)"]:
    a, b = w10[sh], w11[sh]
    diff = [c.coordinate for r in a.iter_rows() for c in r if c.value != b[c.coordinate].value]
    same = (a.max_row, a.max_column) == (b.max_row, b.max_column)
    print(f"1 {sh}: cells differing from nx10:", len(diff), diff[:3], "| same size:", same); ok &= not diff and same
for t in ["cascade", "cascade_hierarchy", "total_ranges"]:
    e = pd.read_csv(f"data/final_2026_nx11_{t}.csv").equals(pd.read_csv(f"data/final_2026_nx18_{t}.csv")); print(f"  data/final_2026_nx18_{t}.csv identical to nx11's:", e); ok &= e
# 2 integrity
wt = openpyxl.load_workbook("Edikted_Forecasting_Exercise.xlsx")
untouched = {n: sum(1 for r in wt[n].iter_rows() for c in r if c.value != w11[n][c.coordinate].value) for n in ["Read me first", "Item gallery", "Data dictionary", "Planned campaigns"]}
med = lambda p: sorted(hashlib.md5(zipfile.ZipFile(p).read(n)).hexdigest() for n in zipfile.ZipFile(p).namelist() if n.startswith("xl/media/"))
pics = med("Edikted_Forecasting_Exercise.xlsx") == med(N11)
print("2 sheets from Edikted, cells changed:", untouched, "| pictures identical:", pics, "| same sheets:", ([s for s in w11.sheetnames if s != "Metrics vs baselines"] == w10.sheetnames and "Metrics vs baselines" in w11.sheetnames))
ok &= not any(untouched.values()) and pics and ([s for s in w11.sheetnames if s != "Metrics vs baselines"] == w10.sheetnames and "Metrics vs baselines" in w11.sheetnames)
# 3 text
me = w11["Model explanation"]; boxes = {r[0].value: (len(r[1].value), me.row_dimensions[r[0].row].height) for r in me.iter_rows(min_row=2) if r[0].value in ("Your approach", "How you tested it", "Where it is weak")}
width = sum(me.column_dimensions[k].width for k in "BCDEFGHIJ")
md = w11["Method and decisions"]; rows = [(r[0].value, r[1].value, r[2].value) for r in md.iter_rows(min_row=3) if r[1].value]
txt = "\n".join(str(c.value) for n in ["Model explanation", "Method and decisions"] for r in w11[n].iter_rows() for c in r if c.value)
braces = "{" in txt or "}" in txt
print("3 boxes (characters, row height):", boxes, "| column B width:", width, "| unfilled placeholders:", braces)
ok &= not braces and all(h < 409 for _, h in boxes.values()) and width == 169
ORDER = ["Framing", "Models", "Information", "Judgement", "Metrics", "Evaluation", "Explainability", "Roadmap"]
areas = [r[0] for r in rows]; in_order = areas == sorted(areas, key=ORDER.index) and set(areas) == set(ORDER)
metrics = [r[1] for r in rows if r[0] == "Metrics"]; expl = [r[1] for r in rows if r[0] == "Explainability"]; info = [r[1] for r in rows if r[0] == "Information"]
gone = [d for d in ["Error measured against baselines", "A buying simulation", "Momentum", "Looks and items bought together"] if d in [r[1] for r in rows]]
print("  decision rows: nx10", sum(1 for r in w10["Method and decisions"].iter_rows(min_row=3) if r[1].value), "-> nx11", len(rows), "| areas in order:", in_order)
print("  metrics rows:", len(metrics), metrics); print("  explainability:", expl); print("  information:", info); print("  rows that should be gone:", gone)
ok &= in_order and len(metrics) == 9 and expl[:1] == ["A cascade for every item and group"] and "Explanations checked against the model" in expl and not gone
ok &= "Recent history" in info and "Fashion trends and relations between items" in info
bad = [p for p in ["hand rules'", "the hand rules,", "Hand-written planning rules", "hand-written planning rules"] if p in txt]; print("  old wording left:", bad); ok &= not bad
need = ["The profit lost: SYS 10.6%", "NOAA", "simulated on the submitted system itself", "0.455 to 0.439", "Real-world evaluation", "Ranges: they are the weakest"]
miss = [p for p in need if p not in txt]; print("  nx18 additions missing:", miss); ok &= not miss and "latest simulated version" not in txt
wr = openpyxl.load_workbook(N11, rich_text=True); bad_lines, grey = [], []
BOXROWS = [r for r in range(2, me.max_row + 1) if me.cell(row=r, column=1).value in ("Your approach", "How you tested it", "Where it is weak")]
for r in BOXROWS:
    v = wr["Model explanation"].cell(row=r, column=2); grey += [v.coordinate] if (v.font.color is None or v.font.color.rgb != "FF000000" or v.font.i) else []
    blocks = list(v.value); i = 0
    while i < len(blocks):                                   # pairs: bold summary, then the rest of the line
        bb = blocks[i]; ok_line = hasattr(bb, "font") and bb.font.b and str(bb.text).endswith(":") and bb.font.color is not None and bb.font.color.rgb == "FF000000"
        if not ok_line: bad_lines.append((v.coordinate, str(getattr(bb, "text", bb))[:40]))
        i += 2
print("  box lines without a bold black summary:", bad_lines, "| grey or italic box cells:", grey); ok &= not bad_lines and not grey
dd = w11["Method and decisions"]; mis = [c.coordinate for r in dd.iter_rows(min_row=3) for c in r[:3] if not (c.alignment.wrap_text and c.alignment.vertical == "top" and c.alignment.horizontal == "left" and c.font.name == "Calibri" and c.font.sz == 11)]
nb = [r[1].coordinate for r in dd.iter_rows(min_row=3) if not (r[0].font.b and r[1].font.b and not r[2].font.b)]
print("  decisions cells not aligned / not Calibri 11:", mis[:5], len(mis), "| rows without bold area + decision:", nb[:5]); ok &= not mis and not nb
# D436: the all-models table twice (under "How you tested it" and below the boxes), each equal to data/model_metrics_table.csv; the layout
MT = pd.read_csv("data/model_metrics_table.csv"); bad_t = []; COLS = ["SYS", "R3", "SN", "AR", "LightGBM", "Neural network", "TabPFN calibrated", "TabPFN", "Recommender"]
lay = {me.cell(row=r, column=1).value: r for r in range(1, me.max_row + 1) if me.cell(row=r, column=1).value in ("Your approach", "How you tested it", "Where it is weak")}
leads = [r for r in range(1, me.max_row + 1) if str(me.cell(row=r, column=1).value).startswith("Every metric of the submitted system")]
note = [r for r in range(1, me.max_row + 1) if str(me.cell(row=r, column=1).value).startswith("Three boxes, free text")]
print("  layout:", lay, "| table leads at rows", leads, "| template note at", note)
ok &= len(leads) == 2 and leads[0] == lay["How you tested it"] + 1 and lay["Where it is weak"] == leads[0] + 2 + len(MT) and len(note) == 1 and note[0] < leads[1]
for T0 in leads:
    for i, r in MT.iterrows():
        n = T0 + 2 + i
        if me.cell(row=n, column=1).value != r["Metric (lower is better)"]: bad_t.append((n, "metric"))
        for j, col in enumerate(COLS, start=2):
            cell = me.cell(row=n, column=j); want = r[f"value {col}"] if col == "SYS" else r[f"ratio {col}"]
            if pd.isna(want):
                if cell.value != "\u2013": bad_t.append((n, col, "dash"))
            elif abs(float(cell.value) - float(want)) > 1e-9: bad_t.append((n, col, "value"))
            if col != "SYS" and pd.notna(want) and abs(float(r[f"value {col}"]) / float(r["value SYS"]) - float(want)) > 0.001: bad_t.append((n, col, "ratio"))
            if cell.fill.fgColor.rgb[-6:].upper() != str(r[f"fill {col}"]).upper(): bad_t.append((n, col, "fill"))
        if me.row_dimensions[n].height != 36: bad_t.append((n, "height"))
merged = {str(m) for m in me.merged_cells.ranges}; need_m = {"B1:J1"} | {f"B{r}:J{r}" for r in lay.values()} | {f"A{r}:J{r}" for r in leads}
print("  all-models table mismatches (both copies):", bad_t, "| header, boxes and leads merged:", need_m <= merged); ok &= not bad_t and need_m <= merged
print("CHECK", "PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)
