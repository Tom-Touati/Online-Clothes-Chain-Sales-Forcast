"""Occasion-tag / tier residuals vs a SEGMENT x month oracle from a January base.
LY: Jan-25 base -> Feb..May-25.  TY check: Dec-25 base -> Jan-26.  Also current tag multipliers in rules."""
import sys
import pandas as pd, numpy as np
sys.path.insert(0, ".")
import rules
it, di, st = rules.load()
mi = pd.read_parquet("data/monthly_item.parquet")
A = mi.pivot_table(index="style_color", columns="month", values="units", aggfunc="sum").fillna(0)
TAGS = list(rules.OCCASION) + ["TIER:" + t for t in ["Party", "Evening", "Resort-Festival", "Loungewear", "Basics", "Social Daywear"]]

def resid(base, targets):
    o = pd.Period(base, "M").end_time.normalize()
    last28 = di[(di.date > o - pd.Timedelta(days=28)) & (di.date <= o)].groupby("style_color").units.sum()
    items = last28[last28 > 0].index
    items = items[it.launch.reindex(items) < pd.Period(base, "M").start_time]      # >= 1 month old
    b = A.loc[items, base]
    seg = it.seg.reindex(items)
    occ = it.occasion.reindex(items).fillna(""); tier = it.tier.reindex(items).fillna("")
    out = {}
    for t in targets:
        a = A.loc[items, t]
        pred = b * seg.map(a.groupby(seg).sum() / b.groupby(seg).sum()).fillna(1)
        row = {}
        for tg in TAGS:
            if tg.startswith("TIER:"):
                m = tier.str.contains(tg[5:], regex=False) & ~occ.str.contains("Halloween|New Year's Eve|Valentine's")
            else:
                m = occ.str.contains(tg, regex=False)
            row[tg] = (a[m].sum() / pred[m].sum(), int(m.sum()))
        out[t] = {k: round(v[0], 2) for k, v in row.items()}
        out["n"] = {k: v[1] for k, v in row.items()}
    return pd.DataFrame(out)
print("LY Jan-25 base"); print(resid("2025-01", ["2025-02", "2025-03", "2025-04", "2025-05"]).to_string())
print("\nTY Dec-25 base -> Jan-26"); print(resid("2025-12", ["2026-01"]).to_string())
print("\nLY Dec-24 base -> Jan-25"); print(resid("2024-12", ["2025-01"]).to_string())
# current multipliers from a January base
for tag, mult in rules.OCCASION.items():
    print(tag, [round(mult.get(m, 1.0) / mult.get(1, 1.0), 3) for m in [2, 3, 4, 5]])
for tr, mult in rules.TIER_IDX.items():
    print("TIER", tr, [round(mult.get(m, 1.0) / mult.get(1, 1.0), 3) for m in [2, 3, 4, 5]])
