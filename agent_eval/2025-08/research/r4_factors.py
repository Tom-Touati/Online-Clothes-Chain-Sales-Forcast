import os, sys, pandas as pd, numpy as np
sys.path.insert(0, os.getcwd()); import rules as R
it, di, st = R.load()
o = di.date.max(); B = o.to_period("M"); T = [B + h for h in range(1, 5)]
f = R.seasonal_factors(it, di, st, B, T)
for k in ["web", "defl_w", "drift", "other", "ret"]:
    print(k, f[k].round(3).to_dict())
print("anc", f["anc"].round(3).to_dict(), "ret_adj", round(f["ret_adj"], 3))
print(f["grp_rel"].round(3).to_string())
items = pd.read_csv("data/to_forecast.csv").style_color.tolist()
F = R.forecast(o, items, it, di, st)
aug = di[di.m == B].groupby("style_color").units.sum().reindex(items).fillna(0)
grp = it.grp.reindex(items)
print("Aug actual", aug.sum(), "per day", round(aug.sum()/31))
print(F.sum().round(0).to_dict())
print(pd.concat([aug.groupby(grp).sum().rename("Aug"), F.groupby(grp).sum()], axis=1).round(0).to_string())
la = it.launch.reindex(items); print("launched Aug", (la >= "2025-08-01").sum(), "Jul", ((la >= "2025-07-01") & (la < "2025-08-01")).sum())
