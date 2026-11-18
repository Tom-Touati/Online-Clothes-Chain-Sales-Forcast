import pandas as pd, numpy as np
s = pd.read_parquet("data/daily_site.parquet").set_index("date")
w=s.web_units
# day-of-week aligned YoY (364 days)
for a,b in [("2025-12-04","2025-12-10"),("2025-12-11","2025-12-17"),("2025-12-18","2025-12-24"),("2025-12-25","2025-12-31"),("2025-12-18","2025-12-31"),("2025-12-04","2025-12-31")]:
    ty=w[a:b].mean(); ly=w[pd.Timestamp(a)-pd.Timedelta(days=364):pd.Timestamp(b)-pd.Timedelta(days=364)].mean()
    print(a,b,round(ty),round(ly),round(ty/ly,3))
print(w["2024-12-15":"2025-01-15"].to_string())
print(s.loc["2024-12-15":"2025-01-10","web_discount"].round(3).to_string())
# LY weekly Jan-Apr 2025 web & discount
x=s["2025-01-01":"2025-05-03"][["web_units","web_discount"]].resample("W-SAT").mean().round(3)
print(x.to_string())
# 2025 Jan-Apr monthly YoY vs 2024 for reference of seasonality in previous year (regime differs)
