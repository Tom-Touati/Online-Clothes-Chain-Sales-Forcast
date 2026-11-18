"""Party-tag check (Valentine / NYE, Sep-24 jump) and forecast diagnostics by group / tag / cohort."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import rules
pd.set_option("display.width", 200)
it, di, st = rules.load()
M = di.pivot_table(index="style_color", columns="m", values="units", aggfunc="sum").fillna(0)
occ = it.occasion.fillna("").reindex(M.index).fillna("")
J, S = pd.Period("2024-06"), pd.Period("2024-09")
ok = (it.launch.reindex(M.index) < J.start_time - pd.Timedelta(days=30)) & (M[J] > 0)
for t in ["Valentine", "New Year", "Homecoming"]:
    m = ok & occ.str.contains(t, regex=False)
    x = M[m]
    r = (x[S] / x[J]).sort_values()
    print(t, "n", len(x), "Sep/Jun per item median %.2f" % r.median(), "share of Sep units top-2 items %.2f"
          % (x[S].sort_values().iloc[-2:].sum() / x[S].sum()),
          "types", it.product_type.reindex(x.index).value_counts().head(4).to_dict(),
          "co-tag Homecoming", occ[m].str.contains("Homecoming").sum())
c = M[ok]
print("panel Sep/Jun-24 %.2f" % (c[S].sum() / c[J].sum()))
# forecast diagnostics with the current rules.py settings
items = pd.read_csv("data/to_forecast.csv").style_color.tolist()
o = di.date.max(); B = o.to_period("M")
f = rules.seasonal_factors(it, di, st, B, [B + h for h in range(1, 5)])
print("anc", f["anc"].round(3).to_dict(), "ret_adj %.3f" % f["ret_adj"])
print("web", f["web"].round(3).to_dict()); print("ret", f["ret"].round(3).to_dict())
F = rules.forecast(o, items, it, di, st)
jun = M.reindex(items)[pd.Period("2025-06")].fillna(0)
g = it.grp.reindex(items).values
print(F.groupby(g).sum().assign(Jun25=jun.groupby(g).sum()).round(0).to_string())
print("total", F.sum().round(0).to_dict(), "Jun25", jun.sum())
for t in ["4th of July", "Halloween", "Homecoming", "Valentine", "New Year"]:
    m = it.occasion.fillna("").reindex(items).str.contains(t, regex=False).values
    print(t, F[m].sum().round(0).to_dict(), "Jun25", jun[m].sum())
coh = it.launch.reindex(items).dt.to_period("M").astype(str).values
for c_ in ["2025-03", "2025-04", "2025-05", "2025-06"]:
    m = coh == c_
    print("cohort", c_, m.sum(), F[m].sum().round(0).to_dict(), "Jun25", jun[m].sum())
