import pandas as pd, numpy as np, sys
sys.path.insert(0, "."); sys.path.insert(0, "research")
import rules as R
from bt import it, di, st, ORIG, ITEMS, LAST
di["mm"] = di.date.dt.to_period("M")
rows = []
for o in ORIG:
    B = pd.Timestamp(o).to_period("M"); T = [B + h for h in range(1, 5) if B + h <= LAST]
    d = di[di.date <= o]
    f = R.seasonal_factors(it, d, st[st.date <= o], B, [B + h for h in range(1, 5)])
    X = di[di.style_color.isin(ITEMS[o])].groupby("mm")[["web_units", "retail", "other"]].sum()
    for t in T:
        dd = t.days_in_month / B.days_in_month
        act = X.loc[t] / X.loc[B] / dd
        ly = t - 12
        rows.append(dict(origin=o[:7], t=str(t), web_f=f["web"][ly], web_a=act.web_units, ret_f=f["ret"][ly] ** 1, ret_a=act.retail,
                         oth_f=f["other"][ly], oth_a=act.other, drift=f["drift"][ly]))
r = pd.DataFrame(rows)
for c in ["web", "ret", "oth"]: r[c + "_a/f"] = r[c + "_a"] / r[c + "_f"]
print(r.round(3).to_string())
print(r[["web_a/f", "ret_a/f", "oth_a/f"]].describe().round(3))
