import sys, zipfile, openpyxl, pandas as pd
orig, new = sys.argv[1], sys.argv[2]
zo, zn = zipfile.ZipFile(orig), zipfile.ZipFile(new)
img = lambda z: sorted(i.file_size for i in z.infolist() if "media/" in i.filename)
print("images: orig", len(img(zo)), "new", len(img(zn)), "| identical byte sizes:", img(zo) == img(zn))
print("parts only in orig:", sorted(set(zo.namelist()) - set(zn.namelist())))
a, b = openpyxl.load_workbook(orig), openpyxl.load_workbook(new)
print("sheets:", a.sheetnames == b.sheetnames, b.sheetnames)
W = pd.read_csv("data/final_forecast_q4.csv", index_col=0)
ws, wo = b["Forecast"], a["Forecast"]
changed, bad = set(), []
for row_o, row_n in zip(wo.iter_rows(), ws.iter_rows()):
    for co, cn in zip(row_o, row_n):
        if co.value != cn.value: changed.add(cn.column_letter)
        if (co.hyperlink is None) != (cn.hyperlink is None) or (co.hyperlink and co.hyperlink.target != cn.hyperlink.target): bad.append(cn.coordinate)
        if co.fill.fgColor.rgb != cn.fill.fgColor.rgb: bad.append("fill " + cn.coordinate)
print("columns whose values changed:", sorted(changed), "| hyperlink/fill differences:", bad[:5], len(bad))
print("freeze panes kept:", ws.freeze_panes == wo.freeze_panes, "| gallery images:", len(b["Item gallery"]._images))
tot_err, conf_ok, n_expl, blank = 0, 0, 0, 0
for r in range(2, 102):
    sc = ws.cell(r, 2).value; f = W.loc[sc]
    v = [ws[f"{c}{r}"].value for c in "GHIJ"]
    tot_err += abs(sum(v) - f.final_total) > 2
    conf_ok += ws[f"L{r}"].value in (1, 2, 3)
    n_expl += bool(ws[f"M{r}"].value); blank += any(x is None or x < 0 for x in v)
    assert ws[f"K{r}"].value == f"=SUM(G{r}:J{r})"
print("rows with sum(G:J) != final total:", tot_err, "| valid confidence:", conf_ok, "| explanations:", n_expl, "| blank/negative month cells:", blank)
print("totals: final", sum(sum(ws[f"{c}{r}"].value for c in "GHIJ") for r in range(2, 102)), "| raw (N)", sum(ws[f"N{r}"].value for r in range(2, 102)))
print("confidence counts:", pd.Series([ws[f"L{r}"].value for r in range(2, 102)]).value_counts().sort_index().to_dict())
me = b["Model explanation"]
for c in ("B2", "B3", "B4"): print(c, len(me[c].value), "chars | italic", me[c].font.i, "| color", me[c].font.color.rgb, "| starts:", me[c].value[:70])
print("A-column labels unchanged:", [me[f"A{r}"].value for r in (2, 3, 4)] == [a["Model explanation"][f"A{r}"].value for r in (2, 3, 4)])
for s in ("Read me first", "Item gallery", "Data dictionary", "Planned campaigns"):
    same = all(x.value == y.value for ro, rn in zip(a[s].iter_rows(), b[s].iter_rows()) for x, y in zip(ro, rn)); print(f"sheet '{s}' unchanged:", same)
