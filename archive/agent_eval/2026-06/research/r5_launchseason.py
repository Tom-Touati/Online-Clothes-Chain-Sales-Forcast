import pandas as pd, numpy as np
from bt import R, it, di, st, ACT, sb, ORIG
at = pd.read_csv("data/item_attributes.csv").set_index("style_color")
rows = []
for o in ORIG:
    B = pd.Period(o, "M"); items = sb[sb.origin == o].style_color.unique().tolist()
    f = R.forecast(B.end_time.normalize(), items, it, di, st)
    ms = [str(B + h) for h in range(1, 5)]
    a = ACT.reindex(index=items, columns=ms).fillna(0)
    L = it.launch.reindex(items)
    age = ((B.start_time - L).dt.days / 30.44)
    lm = L.dt.month
    half = np.where(lm.isin([8, 9, 10, 11, 12, 1]), "FW", "SS")
    df = pd.DataFrame({"p": f[ms].sum(axis=1), "a": a.sum(axis=1), "age": age.values, "half": half,
                       "tier": it.tier.reindex(items).replace("", "none").values}, index=items)
    lv = df.a.sum() / df.p.sum()
    for cond, lab in [((df.age >= 3) & (df.age < 15), "3-15")]:
        x = df[cond]
        r = x.groupby("half").apply(lambda g: g.a.sum() / g.p.sum() / lv)
        n = x.groupby("half").size()
        rows.append(dict(o=o, FW=round(r.get("FW", np.nan), 3), SS=round(r.get("SS", np.nan), 3), nFW=n.get("FW", 0), nSS=n.get("SS", 0)))
    # tiers monthly
    if o in ("2025-05", "2025-06", "2025-07"):
        for tr in ["Basics", "Resort-Festival", "Evening", "Party", "none", "Social Daywear"]:
            m = df.tier == tr
            mon = [(a.loc[m, ms[k]].sum() / f.loc[m, ms[k]].sum()) / (a[ms[k]].sum() / f[ms[k]].sum()) for k in range(4)]
            print(o, tr, m.sum(), " ".join(f"{v:.2f}" for v in mon))
print(pd.DataFrame(rows).to_string())
