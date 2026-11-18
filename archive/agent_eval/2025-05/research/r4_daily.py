import pandas as pd
s=pd.read_parquet("data/daily_site.parquet").set_index("date")
s["other"]=s.units-s.web_units-s.retail
c=["units","web_units","retail","other","web_discount","cancelled_units"]
print(s.loc["2025-05-12":"2025-05-31",c].round(3).to_string())
print(s.loc["2024-05-15":"2024-06-20",c].round(3).to_string())
