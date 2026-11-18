"""Occasion-tag profiles last year from a June base: tagged items vs same-segment untagged items (LFL)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import rules
it, di, st = rules.load()
M = di.pivot_table(index="style_color", columns="m", values="units", aggfunc="sum").fillna(0)
occ = it.occasion.fillna("")
fc = pd.read_csv("data/to_forecast.csv").style_color
tags = ["4th of July", "Graduation", "Homecoming", "Halloween", "Valentine", "New Year", "Christmas", "Thanksgiving",
        "Back to School", "Spring Break", "Festival"]
base = pd.Period("2024-06"); tg = [pd.Period(x) for x in ["2024-07", "2024-08", "2024-09", "2024-10", "2024-11"]]
ok = (it.launch.reindex(M.index) < base.start_time - pd.Timedelta(days=30)) & (M[base] > 0)
seg = it.seg.reindex(M.index)
print("all tags present in panel:", pd.Series([t for s in occ.reindex(fc) for t in s.split("; ") if t]).value_counts().head(25).to_dict())
for t in tags:
    m = occ.reindex(M.index).fillna("").str.contains(t, regex=False)
    x = M[ok & m]
    if len(x) < 3:
        continue
    rows = []
    for s, g in x.groupby(seg[ok & m]):
        ctrl = M[ok & ~m & (seg == s)]
        if len(ctrl) < 3:
            continue
        rows.append([g[base].sum()] + [g[k].sum() / g[base].sum() / (ctrl[k].sum() / ctrl[base].sum()) for k in tg])
    if not rows:
        continue
    r = np.array(rows); w = r[:, 0] / r[:, 0].sum()
    prof = (r[:, 1:] * w[:, None]).sum(axis=0)
    n_now = occ.reindex(fc).str.contains(t, regex=False).sum()
    u_now = M.reindex(fc)[pd.Period("2025-06")][occ.reindex(fc).str.contains(t, regex=False).values].sum()
    print(f"{t:14s} n_LY {len(x):3d} Jul..Nov vs Jun-24 rel to seg: " + " ".join(f"{v:.2f}" for v in prof)
          + f"   | panel now n {n_now}, Jun-25 units {u_now:.0f}")
# this year's May->Jun tag move vs segment, and LY's
for t in ["4th of July", "Graduation", "Homecoming", "Halloween"]:
    out = []
    for a, b in [("2024-05", "2024-06"), ("2025-05", "2025-06")]:
        a, b = pd.Period(a), pd.Period(b)
        okk = (it.launch.reindex(M.index) < a.start_time - pd.Timedelta(days=30)) & (M[a] > 0)
        m = occ.reindex(M.index).fillna("").str.contains(t, regex=False)
        x = M[okk & m]; c = M[okk & ~m & seg.isin(seg[okk & m].unique())]
        out.append(f"{a}->{b} {x[b].sum() / x[a].sum() / (c[b].sum() / c[a].sum()):.2f} (n {len(x)})")
    print(t, " | ".join(out))
