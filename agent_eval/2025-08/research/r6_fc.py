import os, sys, json, pandas as pd
sys.path.insert(0, os.getcwd()); import rules as R
it, di, st = R.load()
items = pd.read_csv("data/to_forecast.csv").style_color.tolist(); o = di.date.max()
grp = it.grp.reindex(items)
V = json.loads(sys.argv[1])
for k, v in V.items():
    old = dict(R.P); R.P.update(v)
    F = R.forecast(o, items, it, di, st)
    R.P.clear(); R.P.update(old)
    g = F.groupby(grp).sum() / 1000
    print(k, "total", (F.sum() / 1000).round(1).to_dict(), "4m", round(F.values.sum() / 1e6, 3))
    print(g.round(1).to_string())
    print("  winter share", (g.loc["WINTER"] / g.sum()).round(3).to_dict())
