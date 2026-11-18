import pandas as pd, numpy as np
import bt
from bt import R, it, di, st, ACT, sb
at = pd.read_csv("data/item_attributes.csv").set_index("style_color")
res = {}
for o in ["2025-05", "2025-06", "2025-07"]:
    B = pd.Period(o, "M"); items = sb[sb.origin == o].style_color.unique().tolist()
    f = R.forecast(B.end_time.normalize(), items, it, di, st)
    ms = [str(B + h) for h in range(1, 5)]
    a = ACT.reindex(index=items, columns=ms).fillna(0)
    df = pd.DataFrame({"p": f[ms].sum(axis=1), "a": a.sum(axis=1)})
    for k, m in enumerate(ms): df[f"p{k+1}"] = f[m]; df[f"a{k+1}"] = a[m]
    L = it.launch.reindex(items)
    df["age"] = pd.cut(((B.start_time - L).dt.days / 30.44).values, [-99, 0, 1, 2, 4, 8, 99], labels=["<0", "0-1", "1-2", "2-4", "4-8", "8+"])
    df["grp"] = it.grp.reindex(items).values; df["type"] = it.product_type.reindex(items).values
    df["tier"] = it.tier.reindex(items).replace("", "none").values
    df["occ"] = it.occasion.reindex(items).fillna("").values
    df["size"] = pd.qcut(df.p.rank(method="first"), 4, labels=["q1", "q2", "q3", "q4"])
    res[o] = df
def tab(key, minn=12):
    out = {}
    for o, df in res.items():
        lv = df.a.sum() / df.p.sum()
        g = df.groupby(key, observed=True)
        r = (g.a.sum() / g.p.sum() / lv)
        out[o] = r.round(2); out[o + "_n"] = g.size()
    t = pd.DataFrame(out)
    return t[t[[c for c in t if c.endswith("_n")]].max(axis=1) >= minn]
for k in ["grp", "age", "tier", "size"]: print(tab(k).to_string())
print(tab("type", 15).to_string())
tags = ["4th of July", "Graduation", "Halloween", "Homecoming", "New Year", "Valentine", "St. Patrick"]
for tg in tags:
    line = []
    for o, df in res.items():
        lv = df.a.sum() / df.p.sum(); m = df.occ.str.contains(tg, regex=False)
        x = df[m]
        mon = [(x[f"a{k}"].sum() / x[f"p{k}"].sum()) / (df[f"a{k}"].sum() / df[f"p{k}"].sum()) for k in range(1, 5)]
        line.append(f"{o}: n{m.sum()} tot {x.a.sum()/x.p.sum()/lv:.2f} mon {' '.join(f'{v:.2f}' for v in mon)}")
    print(tg, " | ".join(line))
# group by month for June origin
df = res["2025-06"]
for g in ["WINTER", "SUMMER", "CORE"]:
    x = df[df.grp == g]
    print(g, [round((x[f"a{k}"].sum() / x[f"p{k}"].sum()) / (df[f"a{k}"].sum() / df[f"p{k}"].sum()), 2) for k in range(1, 5)])
