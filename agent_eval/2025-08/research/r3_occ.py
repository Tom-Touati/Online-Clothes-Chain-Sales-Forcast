import pandas as pd, numpy as np, sys
sys.path.insert(0,".")
import rules_v6 as R
it, di, st = R.load()
print("BOCA first", st[st.store=="BOCA"].date.min(), st[st.store=="BOCA"].groupby(st.date.dt.to_period("W-SAT")).units.mean().round(0).to_dict())
mi = di.groupby(["style_color","m"]).units.sum().unstack().fillna(0)
def prof(base, months, tag, excl=None):
    ok = (it.launch.reindex(mi.index) < base.start_time - pd.Timedelta(days=30)) & (mi[base] > 0)
    X = mi[ok]
    occ = it.occasion.reindex(X.index).fillna("")
    has = occ.str.contains(tag, regex=False)
    if excl: has &= ~occ.str.contains(excl, regex=False)
    seg = it.seg.reindex(X.index)
    ctrl = (~occ.str.contains(tag, regex=False)) & seg.isin(seg[has].unique())
    r = []
    for m in months:
        a = X.loc[has, m].sum() / X.loc[has, base].sum(); b = X.loc[ctrl, m].sum() / X.loc[ctrl, base].sum()
        r.append(round(a / b, 2))
    return has.sum(), r
B = pd.Period("2024-08"); ms = [pd.Period(x) for x in ["2024-09","2024-10","2024-11","2024-12","2025-01"]]
for tag, ex in [("Halloween",None),("Homecoming",None),("New Year's Eve","Homecoming"),("New Year's Eve",None),("Valentine's Day",None),("4th of July",None),("Graduation",None),("St. Patrick's Day",None)]:
    print(tag, "excl", ex, "Aug-24 base Sep..Jan", prof(B, ms, tag, ex))
# this year's Jul->Aug tag move vs LY Jul->Aug
for tag in ["Halloween","Homecoming","New Year's Eve","Valentine's Day","4th of July"]:
    print(tag, "TY Jul->Aug", prof(pd.Period("2025-07"), [pd.Period("2025-08")], tag), "LY", prof(pd.Period("2024-07"), [pd.Period("2024-08")], tag))
# group LFL relative to panel: monthly moves TY vs LY
def grpmove(a, b):
    ok = (it.launch.reindex(mi.index) < a.start_time - pd.Timedelta(days=30)) & (mi[a] > 0)
    X = mi[ok]; g = it.grp.reindex(X.index)
    r = X.groupby(g)[b].sum() / X.groupby(g)[a].sum()
    return (r / (X[b].sum() / X[a].sum())).round(3).to_dict()
for a in ["2024-05","2024-06","2024-07","2024-08","2024-09","2024-10","2024-11","2024-12","2025-05","2025-06","2025-07"]:
    a = pd.Period(a); print(a, "->", a+1, grpmove(a, a+1))
# group share of panel units
for m in ["2024-08","2024-11","2024-12","2025-07","2025-08"]:
    m=pd.Period(m); s=mi[m].groupby(it.grp.reindex(mi.index)).sum(); print(m, (s/s.sum()).round(3).to_dict())
