from bt import run, R, it, di, st
import pandas as pd

B = pd.Period("2026-06", "M"); T = [B + h for h in range(1, 5)]
f = R.seasonal_factors(it, di, st, B, T)
print("cur_ret", f["cur_ret"], "ret_adj", f["ret_adj"])
print(f["grp_rel"].round(3).to_string()); print(f["seg_rel"].loc[["Hoodie","Sweater","Jeans","Shorts","Tank Top","Dress","Skirt","Top|LS","Pants"]].round(2).to_string())
items = pd.read_csv("data/to_forecast.csv").style_color.tolist()
P_ = R.forecast(di.date.max(), items, it, di, st, factors=f)
mi = pd.read_parquet("data/monthly_item.parquet"); j = mi[(mi.month == "2026-06") & mi.style_color.isin(items)]
print("items Jun actual", j.units.sum(), "web/ret/oth", j.online.sum(), j.retail.sum(), j["drop"].sum())
print((P_.sum() / 1000).round(1).to_dict())
g = it.grp.reindex(items).values
print(P_.groupby(g).sum().round(0).to_string()); print(j.groupby(it.grp.reindex(j.style_color).values).units.sum())

