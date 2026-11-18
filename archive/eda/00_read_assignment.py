import sys, openpyxl
wb = openpyxl.load_workbook(sys.argv[1], read_only=True, data_only=True)
for ws in wb.worksheets:
    print("=== SHEET", repr(ws.title), ws.max_row, ws.max_column)
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i >= 60: print("..."); break
        vals = [v for v in row]
        if any(v is not None for v in vals):
            # trim trailing None
            while vals and vals[-1] is None: vals.pop()
            print(i+1, vals[:30])
