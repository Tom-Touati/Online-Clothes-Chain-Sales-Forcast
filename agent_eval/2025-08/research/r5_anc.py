import os, sys, pandas as pd
sys.path.insert(0, os.getcwd()); import rules as R
it, di, st = R.load()
o = di.date.max(); B = o.to_period("M"); T = [B + h for h in range(1, 5)]
for w in [1, 2, 3, 4, 5, 6, 7]:
    R.P["ANCHOR_WIN"] = w; R.P["SEAS_ANCHOR"] = 1.0
    f = R.seasonal_factors(it, di, st, B, T)
    print("win", w, "LY/TY group move (anchor^1, clipped .7-1.4)", f["anc"].round(3).to_dict())
