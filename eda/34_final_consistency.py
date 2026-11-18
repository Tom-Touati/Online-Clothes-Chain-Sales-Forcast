"""End-to-end consistency checks on the final forecast (data/final_forecast_q4.csv) and the filled workbook.
A July/June step by item group vs what the same kind of items did in 2025 (and 2024)
B mature items: forecast vs last year, reconciled with spring trend x measured ageing
C Homecoming/Halloween floors vs the 2025 item-level distribution of Jul-Oct/June for comparable (younger) items
D every explanation in the workbook: stated numbers vs the cells and the forecast file"""
import re, sys, openpyxl, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 230)
O = pd.read_csv("data/final_forecast_q4.csv", index_col=0); T = list(O.index)
M = pd.read_parquet("data/monthly_item.parquet"); it = pd.read_parquet("data/items.parquet").set_index("style_color")
start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
occ = it.occasion.fillna(""); mo = lambda y, ms: [f"{y}-{m:02d}" for m in ms]
WC = ["S24840_LIGHT-GRAY-MELANGE", "S24847_NAVY", "S25039_YELLOW"]
def group(ids, y):
    age = (pd.Timestamp(f"{y}-06-30") - start.reindex(ids)).dt.days.values / 30.44
    g = np.where(age < 3, "launch <3m", np.where(age < 12, "3-12 months", "12m+"))
    g = np.where(it.product_type.reindex(ids).isin(["Swimwear Top", "Swimwear Bottom"]).values, "swimwear", g)
    g = np.where(occ.reindex(ids).str.contains("World Cup").values, "World Cup", g)
    return pd.Series(g, index=ids)

print("== A  July / June by group")
rows = []
for y in (2024, 2025):
    ids = M.index[(M[f"{y}-06"] >= 50) & (start.reindex(M.index) < pd.Timestamp(f"{y}-06-15")).values]
    g = group(ids, y)
    for k, s in g.groupby(g.values):
        rows.append(dict(year=y, group=k, n=len(s), jun=M.loc[s.index, f"{y}-06"].sum(), jul=M.loc[s.index, f"{y}-07"].sum()))
gT = group(pd.Index(T), 2026)
for k, s in gT.groupby(gT.values):
    rows.append(dict(year="2026 fc", group=k, n=len(s), jun=M.loc[s.index, "2026-06"].sum(), jul=O.loc[s.index, "f_jul"].sum()))
A = pd.DataFrame(rows); A["jul_jun"] = A.jul / A.jun
P = A.pivot(index="group", columns="year", values="jul_jun").round(2); P["2026 June share"] = (A[A.year == "2026 fc"].set_index("group").jun / A[A.year == "2026 fc"].jun.sum()).round(3)
P["n 2026"] = A[A.year == "2026 fc"].set_index("group").n; print(P.to_string())
w = A[A.year == "2026 fc"].set_index("group").jun
for y in (2024, 2025):
    r = A[A.year == y].set_index("group").jul_jun.reindex(w.index)
    if "World Cup" in w.index and pd.isna(r.get("World Cup")): r["World Cup"] = A[(A.year == "2026 fc") & (A.group == "World Cup")].jul_jun.iloc[0]
    print(f"July/June expected with {y} group ratios on the 2026 June mix (World Cup at its forecast): {(r * w).sum() / w.sum():.2f}")
print(f"forecast July/June: {O.f_jul.sum() / M.loc[T, '2026-06'].sum():.2f} | without World Cup items: {O.drop(WC).f_jul.sum() / M.loc[[s for s in T if s not in WC], '2026-06'].sum():.2f}")
print(f"Jul-Oct total / June: forecast {O.final_total.sum() / M.loc[T, '2026-06'].sum():.2f}")

print("\n== B  mature items (effective start before April 2025)")
m = O[O.mature].index
jo25 = M.loc[m, mo(2025, [7, 8, 9, 10])].sum(axis=1); yoy = (M.loc[m, mo(2026, [4, 5, 6])].sum(axis=1)) / M.loc[m, mo(2025, [4, 5, 6])].sum(axis=1)
yoyMJ = M.loc[m, mo(2026, [3, 4, 5, 6])].sum(axis=1) / M.loc[m, mo(2025, [3, 4, 5, 6])].sum(axis=1)
r = O.loc[m, "final_total"] / jo25
print(f"n={len(m)} | forecast / Jul-Oct 2025: item median {r.median():.2f}, aggregate {O.loc[m, 'final_total'].sum() / jo25.sum():.2f}")
print(f"spring trend: Mar-Jun yoy item median {yoyMJ.median():.2f} (aggregate {M.loc[m, mo(2026, [3,4,5,6])].sum().sum() / M.loc[m, mo(2025, [3,4,5,6])].sum().sum():.2f}); Apr-Jun yoy item median {yoy.median():.2f}")
print(f"expected = trend x ageing 0.843: item median {(yoy * 0.843).median():.2f}; forecast / (Jul-Oct 2025 x Apr-Jun yoy x 0.843): item median {(r / (yoy * 0.843)).median():.2f}, p10 {(r / (yoy * 0.843)).quantile(.1):.2f}, p90 {(r / (yoy * 0.843)).quantile(.9):.2f}")
# the ageing factor itself: items live before Mar 2024 vs their 2024 -> 2025 continuation
old = start[start < pd.Timestamp("2024-03-01")].index.intersection(M.index)
x = (M.loc[old, mo(2025, [7, 8, 9, 10])].sum(axis=1) + 5) / ((M.loc[old, mo(2024, [7, 8, 9, 10])].sum(axis=1) + 5) * (M.loc[old, mo(2025, [4, 5, 6])].sum(axis=1) + 5) / (M.loc[old, mo(2024, [4, 5, 6])].sum(axis=1) + 5))
print(f"check of the ageing factor on 2024->2025 (items live before Mar 2024, n={len(old)}): median actual / (Jul-Oct 2024 x Apr-Jun yoy) = {x.median():.2f}")

print("\n== C  floors vs the 2025 distribution for comparable items (live by mid-May 2025, under 15 months old at June 2025, June >= 50)")
for tag, fl in [("Homecoming", 6.31), ("Halloween", 6.86)]:
    ids = M.index[(M["2025-06"] >= 50) & (start.reindex(M.index) < pd.Timestamp("2025-05-15")).values & (start.reindex(M.index) >= pd.Timestamp("2024-03-31")).values & occ.reindex(M.index).str.contains(tag).values]
    rr = M.loc[ids, mo(2025, [7, 8, 9, 10])].sum(axis=1) / M.loc[ids, "2025-06"]
    sel = O[(O.seg == tag) & (~O.mature)]
    print(f"{tag}: 2025 comparable items n={len(ids)}, Jul-Oct/June p10 {rr.quantile(.1):.2f} p25 {rr.quantile(.25):.2f} median {rr.median():.2f} p75 {rr.quantile(.75):.2f} p90 {rr.quantile(.9):.2f}; share of items below the {fl} floor: {(rr < fl).mean():.0%}")
    print("   targets:", {s: dict(model=round(O.loc[s, 'model_answer'] / O.loc[s, 'jun'], 2), final=round(O.loc[s, 'after_segment_rule'] / O.loc[s, 'jun'], 2), pct_in_2025=f"{(rr <= O.loc[s, 'after_segment_rule'] / O.loc[s, 'jun']).mean():.0%}") for s in sel.index})

print("\n== D  workbook explanations vs cells")
ws = openpyxl.load_workbook(sys.argv[1], read_only=True, data_only=False)["Forecast"]
bad, n = [], 0
for r in ws.iter_rows(min_row=2, max_row=101, values_only=True):
    sc, G, H, I, J, L, Mx, N = r[1], r[6], r[7], r[8], r[9], r[11], r[12], r[13]
    o = O.loc[sc]; tot = G + H + I + J
    if tot != int(o.final_total) or N != int(round(o.raw_total)) or L != int(o.conf): bad.append((sc, "cells vs file"))
    if not Mx: continue
    n += 1
    num = lambda pat: (int(re.search(pat, Mx).group(1).replace(",", "")) if re.search(pat, Mx) else None)
    if num(r"Final ([\d,]+)\.") != tot: bad.append((sc, f"Final {num(r'Final ([0-9,]+)[.]')} vs cells {tot}"))
    if num(r"Confidence (\d)") != L: bad.append((sc, f"confidence text vs L={L}"))
    ma = num(r"Model answer ([\d,]+)")
    if ma is not None and ma != int(round(o.model_answer)): bad.append((sc, f"model answer {ma} vs {o.model_answer:.0f}"))
    if ma is not None and N != ma: bad.append((sc, f"model answer {ma} vs column N {N}"))
    if "Months follow the Halloween pattern, peaking in October" in Mx and not (J >= max(G, H, I)): bad.append((sc, "says October peak but months disagree"))
    if "peaking in September" in Mx and not (I >= max(G, H, J)): bad.append((sc, "says September peak but months disagree"))
    if "falling from July" in Mx and not (G >= H >= I >= J * 0.999): bad.append((sc, "says falling from July but months disagree"))
    if "capped at 2" in Mx and L != 2: bad.append((sc, "says capped at 2"))
print(f"explanations checked: {n} | inconsistencies: {len(bad)}"); [print("  ", b) for b in bad]
