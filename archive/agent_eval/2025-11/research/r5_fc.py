"""Forecast summary at this origin + backtest v9 vs v10 at origins Feb-Oct-25 (site levels = actuals, oracle)."""
import sys, importlib
import pandas as pd, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "research")
import rules as R10
R9 = importlib.import_module("rules_v9")
R9.SITE_WEB_PD.update(R10.SITE_WEB_PD); R9.OTHER_PD.update(R10.OTHER_PD)   # same site actuals in both

it, di, st = R10.load()
o = di.date.max(); B = o.to_period("M")
items = pd.read_csv("data/to_forecast.csv").style_color.tolist()
f = R10.seasonal_factors(it, di, st, B, [B + h for h in range(1, 5)])
for k in ["web", "ret", "other", "drift"]:
    print(k, f[k].round(3).to_dict())
print("seas_pow", round(f["seas_pow"], 3), "anc", f["anc"].round(3).to_dict(), "ret_adj", round(f["ret_adj"], 3))
print("grp_rel", f["grp_rel"].round(3).to_dict())
p = R10.forecast(o, items, it, di, st, factors=f)
nov = di[di.m == B].groupby("style_color").units.sum().reindex(items).fillna(0)
print(f"Nov actual these items {nov.sum():.0f} ({nov.sum()/30:.0f}/day)")
g = it.grp.reindex(items).fillna("CORE")
t = p.groupby(g).sum(); t["Nov act"] = nov.groupby(g).sum(); print(t.round(0).to_string())
print("total", p.sum().round(0).to_dict(), "4m", round(p.values.sum()))
occ = it.occasion.reindex(items).fillna("")
for tag in ["Halloween", "New Year's Eve", "Valentine's", "St. Patrick"]:
    h = occ.str.contains(tag, regex=False)
    print(tag, h.sum(), "Nov", nov[h].sum(), p[h].sum().round(0).to_dict())
p9 = R9.forecast(o, items, it, di, st)
print("v9 logic + v10 site plan:", p9.sum().round(0).to_dict())

# backtest
sb = pd.read_csv("feedback/scoreboard.csv")
act_m = di.groupby(["style_color", "m"]).units.sum()
def bt(mod, label):
    res = []
    for org in sorted(sb.origin.unique()):
        if org == "2025-11":
            continue
        its = sorted(sb[sb.origin == org].style_color.unique())
        O = pd.Period(org).end_time.normalize()
        pr = mod.forecast(O, its, it, di, st)
        cols = [c for c in pr.columns if pd.Period(c) <= pd.Period("2025-11")]
        a = pd.DataFrame({c: act_m.xs(pd.Period(c), level="m").reindex(its).fillna(0) for c in cols})
        P_, A = pr[cols].sum(axis=1), a.sum(axis=1)
        r = P_ / A.clip(lower=1e-9)
        res.append(dict(origin=org, wape=(P_ - A).abs().sum() / A.sum(), bias=P_.sum() / A.sum() - 1,
                        w20=((r - 1).abs() <= .2).mean(), w50=((r - 1).abs() <= .5).mean(),
                        h1=(pr[cols[0]] - a[cols[0]]).abs().sum() / a[cols[0]].sum()))
    r = pd.DataFrame(res).set_index("origin")
    print(label, "mean", r.mean().round(4).to_dict()); print(r.round(3).T.to_string())
bt(R9, "v9 ")
bt(R10, "v10")
