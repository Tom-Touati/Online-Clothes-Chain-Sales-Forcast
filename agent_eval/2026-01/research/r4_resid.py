"""Residuals of the base rules with site actuals, by channel mix / size / age / group, per h."""
import sys
import pandas as pd, numpy as np
sys.path.insert(0, "research"); sys.path.insert(0, ".")
import r3_bt as bt
over = eval(sys.argv[1]) if len(sys.argv) > 1 else {}
out, r = bt.run(over, keep=True)
print(out)
mi = bt.mi.rename(columns={"month": "origin"})
r = r.merge(mi[["style_color", "origin", "units", "retail", "drop", "online"]], on=["style_color", "origin"], how="left")
r["drop_sh"] = r["drop"] / r.units.replace(0, np.nan)
r["ret_sh"] = r.retail / r.units.replace(0, np.nan)
r["dropb"] = pd.cut(r.drop_sh, [-0.01, 0.05, 0.2, 0.5, 1.01], labels=["d<5", "d5-20", "d20-50", "d50+"])
r["retb"] = pd.cut(r.ret_sh, [-0.01, 0.0001, 0.1, 0.25, 1.01], labels=["r0", "r<10", "r10-25", "r25+"])
r["sizeq"] = r.groupby(["origin", "month"]).pred.transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
o = pd.PeriodIndex(r.origin, freq="M")
r["age"] = (o.start_time - r.style_color.map(bt.it.launch).values).days / 30.44
r["ageb"] = pd.cut(r.age, [-99, 0, 1, 2, 4, 6, 12, 999], labels=["in", "0-1", "1-2", "2-4", "4-6", "6-12", "12+"])
r["grp"] = r.style_color.map(bt.it.grp)
r["ywin"] = np.where(r.origin >= "2025-09", "Sep+", "early")
print("\nraw act/pred by h (level incl.)")
g = r.groupby("h"); print((g.actual.sum() / g.pred.sum()).round(3).to_string())
g = r.groupby(["origin"]); print((g.actual.sum() / g.pred.sum()).round(3).to_string())
tot = r.groupby(["origin", "month"])[["pred", "actual"]].transform("sum")
r["predn"] = r.pred * tot.actual / tot.pred
for col in ["dropb", "retb", "sizeq", "ageb", "grp"]:
    g = r.groupby([col, "h"], observed=True)
    t = (g.actual.sum() / g.predn.sum()).unstack().round(2)
    t2 = r[r.h == 2].groupby([col, "ywin"], observed=True).apply(lambda x: x.actual.sum() / x.predn.sum()).unstack().round(2)
    print(f"\n== {col}"); print(pd.concat([t, t2], axis=1).to_string())
# drop share effect within size quintiles (confounding check), h=3-4
x = r[r.h >= 3]
g = x.groupby(["sizeq", "dropb"], observed=True)
print("\nh>=3 sizeq x dropb"); print((g.actual.sum() / g.predn.sum()).unstack().round(2).to_string())
g = x.groupby(["sizeq", "retb"], observed=True)
print("\nh>=3 sizeq x retb"); print((g.actual.sum() / g.predn.sum()).unstack().round(2).to_string())
