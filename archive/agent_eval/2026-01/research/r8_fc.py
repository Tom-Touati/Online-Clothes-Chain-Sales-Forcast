"""Forecast summary: by group / channel / age vs January actuals of the same items."""
import sys
import pandas as pd, numpy as np
sys.path.insert(0, ".")
import rules
it, di, st = rules.load()
p = pd.read_csv("predictions.csv").pivot(index="style_color", columns="month", values="pred")
mi = pd.read_parquet("data/monthly_item.parquet")
jan = mi[mi.month == "2026-01"].set_index("style_color").reindex(p.index).fillna(0)
p["jan_act"] = jan.units
g = it.grp.reindex(p.index)
print((p.groupby(g).sum() / 1000).round(1).to_string())
print("per day:", (p.drop(columns="jan_act").sum() / [28, 31, 30, 31]).round(0).to_dict(), "Jan act/day", round(jan.units.sum() / 31))
print("items with zero total:", (p.drop(columns="jan_act").sum(axis=1) == 0).sum(), " Jan units of those:",
      jan.units[p.drop(columns="jan_act").sum(axis=1) == 0].sum())
age = (pd.Timestamp("2026-01-01") - it.launch.reindex(p.index)).dt.days / 30.44
ab = pd.cut(age, [-99, 0, 1, 2, 4, 12, 999])
print((p.groupby(ab, observed=True).sum() / 1000).round(1).to_string())
