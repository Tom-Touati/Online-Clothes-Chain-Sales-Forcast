import pandas as pd, numpy as np
D = "data/"
s = pd.read_parquet(D + "daily_site.parquet")
s["m"] = s.date.dt.to_period("M")
s["other"] = s.units - s.web_units - s.retail
g = s.groupby("m")
pd_ = g[["units", "web_units", "retail", "other", "cancelled_units"]].sum().div(g.date.nunique(), axis=0)
pd_["disc"] = (s.web_discount * s.web_units).groupby(s.m).sum() / g.web_units.sum()
pd_["web_yoy"] = pd_.web_units / pd_.web_units.shift(12)
pd_["oth_yoy"] = pd_.other / pd_.other.shift(12)
pd_["ret_yoy"] = pd_.retail / pd_.retail.shift(12)
print(pd_.round(3).tail(26).to_string())
# January daily vs LY, day-of-week aligned (364 days)
s = s.set_index("date")
jan = s.loc["2026-01-01":"2026-01-31"]
rows = []
for d, r in jan.iterrows():
    ly = s.loc[d - pd.Timedelta(days=364)]
    rows.append((d.date(), d.day_name()[:3], int(r.web_units), int(ly.web_units), round(r.web_units / ly.web_units, 2),
                 round(r.web_discount, 3), round(ly.web_discount, 3), int(r.other), int(r.retail)))
print(pd.DataFrame(rows, columns=["date", "dow", "web", "web_ly", "yoy", "disc", "disc_ly", "other", "retail"]).to_string())
# weekly LY Jan-Jun 2025 web / other / discount
w = s.loc["2025-01-01":"2025-06-30"].resample("W")[["web_units", "other", "retail"]].mean()
w["disc"] = s.loc["2025-01-01":"2025-06-30"].web_discount.resample("W").mean()
print(w.round(3).to_string())
ch = pd.read_parquet(D + "daily_channel.parquet")
ch["m"] = ch.date.dt.to_period("M")
c = ch.pivot_table(index="m", columns="channel", values="units", aggfunc="sum").fillna(0)
print(c.tail(17).astype(int).to_string())
