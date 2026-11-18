import pandas as pd, numpy as np
s = pd.read_parquet("data/daily_site.parquet").set_index("date")
s["other"] = s.units - s.web_units - s.retail
def wk(a,b):
    x = s.loc[a:b]
    w = x.resample("W-SAT").agg({"web_units":"mean","other":"mean","retail":"mean","web_discount":"mean","units":"count"})
    return w.round(2)
print("LY May-Oct 2025 weekly (week ending Sat)"); print(wk("2025-05-01","2025-11-01").to_string())
print("TY Mar-Jun 2026 weekly"); print(wk("2026-03-01","2026-06-30").to_string())
print("LY Jul 2025 daily"); print(s.loc["2025-06-20":"2025-08-10", ["web_units","web_discount","other","retail"]].round(3).to_string())
print("TY Jun 2026 daily"); print(s.loc["2026-06-01":"2026-06-30", ["web_units","web_discount","other","retail"]].round(3).to_string())
