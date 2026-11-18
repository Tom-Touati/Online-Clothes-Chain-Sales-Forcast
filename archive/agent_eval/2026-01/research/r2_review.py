"""Scoreboard review: act/pred relative to each (origin, month) overall ratio, by segment."""
import pandas as pd, numpy as np
import sys; sys.path.insert(0, ".")
import rules
it, di, st = rules.load()
sb = pd.read_csv("feedback/scoreboard.csv")
mi = pd.read_parquet("data/monthly_item.parquet")
sb["grp"] = sb.style_color.map(it.grp); sb["type"] = sb.style_color.map(it.product_type)
sb["tier"] = sb.style_color.map(it.tier); sb["occ"] = sb.style_color.map(it.occasion).fillna("")
o = pd.PeriodIndex(sb.origin, freq="M")
sb["age"] = ((o.start_time - sb.style_color.map(it.launch).values).days / 30.44)
sb["ageb"] = pd.cut(sb.age, [-99, 0, 1, 2, 4, 6, 12, 999], labels=["in", "0-1", "1-2", "2-4", "4-6", "6-12", "12+"])
base = mi.rename(columns={"month": "origin"})[["style_color", "origin", "units", "online", "retail", "drop"]]
sb = sb.merge(base, on=["style_color", "origin"], how="left")
sb["ret_sh"] = sb.retail / sb.units.replace(0, np.nan)
sb["chb"] = pd.cut(sb.ret_sh, [-0.01, 0.0001, 0.1, 0.25, 1.01], labels=["r0", "r<10", "r10-25", "r25+"])
sb["drop_sh"] = sb["drop"] / sb.units.replace(0, np.nan)
sb["dropb"] = pd.cut(sb.drop_sh, [-0.01, 0.05, 0.2, 0.5, 1.01], labels=["d<5", "d5-20", "d20-50", "d50+"])
sb["sizeq"] = sb.groupby(["origin", "month"]).pred.transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
tot = sb.groupby(["origin", "month"])[["pred", "actual"]].transform("sum")
sb["predn"] = sb.pred * tot.actual / tot.pred              # level-neutral prediction per (origin, month)

def tab(col, df, title):
    g = df.groupby([col, "h"], observed=True)
    r = (g.actual.sum() / g.predn.sum()).unstack().round(2)
    n = df[df.h == 1].groupby(col, observed=True).size().rename("n_h1")
    print(f"\n== {title}: act/pred (level-neutral) by h"); print(pd.concat([r, n], axis=1).to_string())

jan = sb[sb.month == "2026-01"]
for col in ["grp", "ageb", "chb", "dropb", "sizeq"]:
    tab(col, jan, f"Jan-26 targets, {col}")
for col in ["grp", "ageb", "chb", "dropb", "sizeq"]:
    tab(col, sb, f"ALL origins, {col}")
# Jan-26 h=1 by type and tags
j1 = jan[jan.h == 1]
g = j1.groupby("type"); r = pd.DataFrame({"ap": g.actual.sum() / g.predn.sum(), "n": g.size(), "act": g.actual.sum()})
print("\n== Jan h=1 by type"); print(r[r.act > 4000].round(2).sort_values("ap").to_string())
for tag in ["Halloween", "New Year's Eve", "Valentine's", "Homecoming", "Graduation", "St. Patrick", "4th of July"]:
    m = j1.occ.str.contains(tag, regex=False)
    m2 = jan.occ.str.contains(tag, regex=False)
    print(tag, m.sum(), round(j1[m].actual.sum() / j1[m].predn.sum(), 2),
          "all h Jan:", round(jan[m2].actual.sum() / jan[m2].predn.sum(), 2))
for tr in ["Party", "Evening", "Resort-Festival", "Loungewear", "Basics", "Social Daywear", "Everyday"]:
    m = j1.tier.fillna("").str.contains(tr, regex=False)
    print("tier", tr, m.sum(), round(j1[m].actual.sum() / j1[m].predn.sum(), 2))
# overall 4-month WAPE by origin
w = sb.groupby(["origin", "style_color"])[["pred", "actual"]].sum()
cnt = sb.groupby("origin").month.nunique()
w = w.reset_index(); w = w[w.origin.map(cnt) == 4]
print("\n4m WAPE by origin"); print(w.groupby("origin").apply(lambda x: pd.Series({
    "wape": (x.pred - x.actual).abs().sum() / x.actual.sum(), "bias": x.pred.sum() / x.actual.sum() - 1,
    "w20": ((x.pred - x.actual).abs() <= 0.2 * x.actual).mean()})).round(3).to_string())
