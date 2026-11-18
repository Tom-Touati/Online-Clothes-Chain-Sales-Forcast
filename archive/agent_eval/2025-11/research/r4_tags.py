"""Tag / tier profiles from a NOV-24 base, controlled for segment: tagged items' actual / (base x same-segment
untagged items' actual/base ratio), per month Dec-24..Mar-25.  Plus ramp variants on the Nov-24 allocation test."""
import sys
import pandas as pd, numpy as np
sys.path.insert(0, ".")
sys.argv = ["x"]
exec(open("research/r3_bt.py").read().split("base, pb = run")[0])     # reuse setup (items, act, neutral, run)

base = R.forecast(O, items, it, di, st, factors=neutral)               # b x days x ramp (no occasion)
R_OCC = dict(R.OCCASION); R.OCCASION.clear()
b0 = R.forecast(O, items, it, di, st, factors=neutral)                  # without occasion multipliers
R.OCCASION.update(R_OCC)
seg = it.seg.reindex(items).fillna("NA")
occ = it.occasion.reindex(items).fillna("")
tier = it.tier.reindex(items).fillna("")

def ctrl(mask, label):
    out = {}
    for c in act.columns:
        ref = ~mask
        r = act.loc[ref.values, c].groupby(seg[ref.values]).sum() / b0.loc[ref.values, c].groupby(seg[ref.values]).sum()
        allr = act.loc[ref.values, c].sum() / b0.loc[ref.values, c].sum()
        exp = b0.loc[mask.values, c] * seg[mask.values].map(r).fillna(allr)
        out[c] = round(act.loc[mask.values, c].sum() / exp.sum(), 2)
    print(f"{label:30s} n={mask.sum():3d}  same-seg rel {out}")

notag = ~occ.str.contains("Halloween|New Year|Homecoming|Valentine|Graduation|4th of July|St. Patrick")
for tag in ["Halloween", "New Year's Eve", "Valentine's", "Homecoming", "Graduation", "4th of July", "St. Patrick"]:
    ctrl(occ.str.contains(tag, regex=False), tag)
ctrl(occ.str.contains("New Year's Eve", regex=False) & ~occ.str.contains("Homecoming"), "NYE w/o Homecoming")
ctrl(occ.str.contains("Halloween") & tier.str.contains("Party|Evening"), "Halloween party/evening")
ctrl(occ.str.contains("Halloween") & ~tier.str.contains("Party|Evening"), "Halloween other tier")
for tr in ["Party", "Evening", "Social Daywear", "Casual", "Basics", "Loungewear", "Resort-Festival"]:
    ctrl((tier == tr) & notag, f"tier {tr} (no tag)")
ctrl(tier.str.contains("Party") & ~occ.str.contains("Halloween"), "Party tier w/o Halloween")
# all tags in the item master (any with n >= 8)
tags = occ.str.split("; ").explode().value_counts()
print("tags present:", tags[tags >= 8].to_dict())

for kw in [dict(RAMP0=1.45, RAMP1=1.3), dict(RAMP0=1.3, RAMP1=1.2, RAMP2=1.1), dict(RAMP_H2=1.0),
           dict(RAMP0=1.45, RAMP1=1.3, RAMP_H2=1.0), dict(SPIKE_CAP=1.0, RAMP0=1.45, RAMP1=1.3),
           dict(SPIKE_CAP=1.0, RAMP_H2=1.0), dict(SPIKE_CAP=1.0, W14=0.3)]:
    run(str(kw), **kw)
