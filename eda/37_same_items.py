"""Same targets, their own 2025: July/June, July share of Jul-Oct, and Jul-Oct/June, against the 2026 forecast. Grouped by age at June 2026."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
src = open("eda/34_final_consistency.py", encoding="utf-8").read(); src = src[:src.index('print("== A')]; ns = {}; exec(src, ns)
O, M, group, T, WC, mo, start = ns["O"], ns["M"], ns["group"], ns["T"], ns["WC"], ns["mo"], ns["start"]
g = group(pd.Index(T), 2026)
live25 = [s for s in T if start[s] < pd.Timestamp("2025-05-15") and M.loc[s, "2025-06"] >= 50]
for k in ["12m+", "3-12 months"]:
    ids = [s for s in g[g == k].index if s in live25]
    if not ids: print(k, "no items live in June 2025"); continue
    h = M.loc[ids]; f = O.loc[ids]
    print(f"{k}: {len(ids)} targets live in June 2025 | own 2025 July/June {h['2025-07'].sum() / h['2025-06'].sum():.2f} vs forecast {f.f_jul.sum() / h['2026-06'].sum():.2f} | own 2025 July share {h['2025-07'].sum() / h[mo(2025,[7,8,9,10])].sum().sum():.3f} vs forecast {f.f_jul.sum() / f.final_total.sum():.3f} | own 2025 Jul-Oct/June {h[mo(2025,[7,8,9,10])].sum().sum() / h['2025-06'].sum():.2f} vs forecast {f.final_total.sum() / h['2026-06'].sum():.2f}")
ids = [s for s in g[g == "3-12 months"].index if s not in live25]
print(f"3-12 months not live in June 2025: {len(ids)} items, forecast July/June {O.loc[ids].f_jul.sum() / M.loc[ids, '2026-06'].sum():.2f}, Jul-Oct/June {O.loc[ids].final_total.sum() / M.loc[ids, '2026-06'].sum():.2f}, segments {O.loc[ids].seg.value_counts().to_dict()}")
# history for items of the same age (3-12 months at June 2025) split by whether they had passed their peak by June
ids25 = M.index[(M["2025-06"] >= 50) & (start.reindex(M.index) < pd.Timestamp("2025-03-31")).values & (start.reindex(M.index) >= pd.Timestamp("2024-06-30")).values]
pk = M.loc[ids25, "2024-01":"2025-06"].max(axis=1); off = M.loc[ids25, "2025-06"] / pk
for lab, sel in [("June at >= 70% of peak", off >= .7), ("June at 35-70% of peak", (off >= .35) & (off < .7)), ("June below 35% of peak", off < .35)]:
    x = M.loc[ids25[sel.values]]; print(f"2025, 3-12 month items, {lab}: n={sel.sum()}, July/June {x['2025-07'].sum() / x['2025-06'].sum():.2f}, Jul-Oct/June {x[mo(2025,[7,8,9,10])].sum().sum() / x['2025-06'].sum():.2f}")
ids26 = g[g == "3-12 months"].index; pk26 = M.loc[ids26, "2025-01":"2026-06"].max(axis=1); off26 = M.loc[ids26, "2026-06"] / pk26
for lab, sel in [("June at >= 70% of peak", off26 >= .7), ("June at 35-70% of peak", (off26 >= .35) & (off26 < .7)), ("June below 35% of peak", off26 < .35)]:
    s = ids26[sel.values]; print(f"2026 forecast, 3-12 month targets, {lab}: n={len(s)}, July/June {O.loc[s].f_jul.sum() / M.loc[s, '2026-06'].sum():.2f}, Jul-Oct/June {O.loc[s].final_total.sum() / M.loc[s, '2026-06'].sum():.2f}")
