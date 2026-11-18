import sys, openpyxl
wb = openpyxl.load_workbook(sys.argv[1])
ws = wb["Forecast"]
print("dims", ws.dimensions, "merged", ws.merged_cells.ranges, "freeze", ws.freeze_panes, "images", len(ws._images))
for r in (1, 2, 3, 101, 102, 103):
    for c in "FGHIJKLMN":
        cell = ws[f"{c}{r}"]
        print(cell.coordinate, repr(cell.value)[:60], "| font", cell.font.name, cell.font.sz, cell.font.color.rgb if cell.font.color else None, "| fill", cell.fill.fgColor.rgb if cell.fill.fill_type else None, "| fmt", cell.number_format, "| wrap", cell.alignment.wrap_text, "| link", bool(cell.hyperlink))
print("col widths", {k: v.width for k, v in ws.column_dimensions.items()})
ws = wb["Model explanation"]
for row in ws.iter_rows():
    for cell in row:
        if cell.value: print(cell.coordinate, repr(cell.value)[:70], "| font", cell.font.name, cell.font.sz, cell.font.color.rgb if cell.font.color else None, cell.font.i, "| wrap", cell.alignment.wrap_text)
print("col widths", {k: v.width for k, v in ws.column_dimensions.items()}, "row heights", {k: v.height for k, v in ws.row_dimensions.items()})
print("gallery images", len(wb["Item gallery"]._images))
