import sys, openpyxl, csv
wb = openpyxl.load_workbook(sys.argv[1], read_only=True, data_only=True)
ws = wb["Forecast"]
rows = [r[:5] for r in ws.iter_rows(min_row=2, values_only=True) if r[1]]
with open("data/targets.csv","w",newline="",encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["n","style_color","product_name","product_type","published_date"]); w.writerows(rows)
print(len(rows))
ws = wb["Planned campaigns"]
for r in ws.iter_rows(values_only=True): print(r)
