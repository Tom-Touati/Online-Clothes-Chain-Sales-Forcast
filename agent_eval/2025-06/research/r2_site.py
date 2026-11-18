"""Site / channel / store levels: June-25 weekly, and 2024 Jun-Oct shape for comparison."""
import pandas as pd
pd.set_option("display.width", 200)
s = pd.read_parquet("data/daily_site.parquet")
s["other"] = s.units - s.web_units - s.retail
s["m"] = s.date.dt.to_period("M")
s["disc"] = s.web_discount
cols = ["units", "web_units", "retail", "other", "drop"]
print("== monthly per day")
g = s.groupby("m")
md = g[cols].sum().div(g.size(), axis=0).round(0)
md["disc"] = g.disc.mean().round(3)
print(md.to_string())
print("\n== weekly per day, 2025-05-01..")
w = s[s.date >= "2025-05-01"].set_index("date")[cols + ["disc"]]
print(w.resample("W-SAT").mean().round(0).to_string())
print("\n== weekly per day, 2024-06-01..2024-10-31")
w = s[(s.date >= "2024-06-01") & (s.date <= "2024-10-31")].set_index("date")[cols + ["disc"]]
print(w.resample("W-SAT").mean().round(2).to_string())
print("\n== June-25 daily")
print(s[s.date >= "2025-06-01"].set_index("date")[cols + ["disc"]].round(2).to_string())
c = pd.read_parquet("data/daily_channel.parquet")
c["m"] = c.date.dt.to_period("M")
print("\n== channel monthly")
print(c[c.date >= "2024-05-01"].pivot_table(index="channel", columns="m", values="units", aggfunc="sum").fillna(0).astype(int).to_string())
t = pd.read_parquet("data/daily_store.parquet")
t["m"] = t.date.dt.to_period("M")
print("\n== store per day")
g = t[t.date >= "2024-05-01"].groupby(["store", "m"])
print((g.units.sum() / g.date.nunique()).unstack().round(0).to_string())
