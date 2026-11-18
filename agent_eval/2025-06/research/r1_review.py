"""Review of scoreboard: June-25 misses by segment (all origins), 4-month total for origin Feb-25."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import rules
pd.set_option("display.width", 200)
it, di, st = rules.load()
sb = pd.read_csv("feedback/scoreboard.csv")
mi = pd.read_parquet("data/monthly_item.parquet")

def stats(g):
    return pd.Series(dict(n=len(g), pred=g.pred.sum(), act=g.actual.sum(), ratio=g.actual.sum() / g.pred.sum(),
                          wape=(g.pred - g.actual).abs().sum() / g.actual.sum()))

# 4-month totals per origin (complete only for 2025-02)
tot = sb.groupby(["origin", "style_color"]).agg(pred=("pred", "sum"), actual=("actual", "sum"), k=("h", "size")).reset_index()
for o, g in tot.groupby("origin"):
    e = (g.pred - g.actual).abs().sum() / g.actual.sum()
    r = g.actual / g.pred.replace(0, np.nan)
    within = ((g.pred - g.actual).abs() <= 0.2 * g.actual).mean()
    print(o, "months", g.k.max(), "WAPE %.3f bias %+.3f  +-20%% %.2f" % (e, g.pred.sum() / g.actual.sum() - 1, within))

# June-25 by segment for each origin, ratio relative to the origin's overall ratio
jun = sb[sb.month == "2025-06"].copy()
jun = jun.join(it[["product_type", "grp", "occasion", "launch"]], on="style_color")
o = pd.Timestamp("2025-05-31")
jun["age"] = pd.cut((pd.Timestamp("2025-06-01") - jun.launch).dt.days / 30.44, [-99, 1, 2, 3, 6, 12, 999],
                    labels=["<1", "1-2", "2-3", "3-6", "6-12", "12+"])
# channel mix in May
may = mi[mi.month == "2025-05"].set_index("style_color")
jun["ret_sh"] = (may.retail / may.units).reindex(jun.style_color).values
jun["mix"] = pd.cut(jun.ret_sh, [-1, 0.1, 0.25, 0.4, 1.01], labels=["web<10", "10-25", "25-40", "ret>40"])
jun["sizeq"] = jun.groupby("origin").pred.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
for col in ["grp", "age", "mix", "sizeq"]:
    t = jun.groupby(["origin", col], observed=True).apply(stats).reset_index()
    over = jun.groupby("origin").apply(stats).ratio
    t["rel"] = t.ratio / t.origin.map(over)
    print("\n== June by", col)
    print(t.pivot(index=col, columns="origin", values="rel").round(2).to_string())
pt = jun.groupby(["origin", "product_type"]).apply(stats).reset_index()
over = jun.groupby("origin").apply(stats).ratio
pt["rel"] = pt.ratio / pt.origin.map(over)
big = pt[pt.origin == "2025-05"].sort_values("act", ascending=False).head(18).product_type
print("\n== June by product type (rel), n / act for 2025-05")
print(pt[pt.product_type.isin(big)].pivot(index="product_type", columns="origin", values="rel").round(2)
      .join(pt[pt.origin == "2025-05"].set_index("product_type")[["n", "act"]]).sort_values("act", ascending=False).to_string())
tags = ["4th of July", "Graduation", "Homecoming", "Halloween", "Valentine", "New Year", "St. Patrick", "Christmas"]
print("\n== June by occasion tag (rel)")
for tg in tags:
    m = jun.occasion.fillna("").str.contains(tg, regex=False)
    if m.sum() == 0: continue
    s = jun[m].groupby("origin").apply(stats)
    print(tg, (s.ratio / over).round(2).to_dict(), "n", int(s.n.max()))
print("\n== origin 2025-05 June overall", stats(jun[jun.origin == "2025-05"]).round(3).to_dict())
# channel level: actual June per channel for origin-05 items vs May
ids = jun[jun.origin == "2025-05"].style_color
for mth in ["2025-04", "2025-05", "2025-06"]:
    x = mi[(mi.month == mth) & mi.style_color.isin(ids)][["units", "online", "retail", "drop"]].sum()
    print(mth, x.to_dict())
