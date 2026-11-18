import pandas as pd, numpy as np
s = pd.read_parquet("data/daily_site.parquet"); s["m"]=s.date.dt.to_period("M")
s["other"]=s.units-s.web_units-s.retail
g=s.groupby("m")
pdm=g[["units","web_units","retail","other","cancelled_units"]].sum().div(g.date.nunique(),axis=0).round(0)
pdm["wdisc"]=g.web_discount.mean().round(3)
print(pdm.tail(22).to_string())
# weekly Aug-Sep 2025 and 2024
for y in [2024,2025]:
    w=s[(s.date>=f"{y}-08-15")&(s.date<=f"{y}-10-31")].set_index("date")[["web_units","retail","other","web_discount"]]
    print(y); print(w.resample("W-SAT").mean().round(0).to_string())
print(s[(s.date>="2025-08-28")].set_index("date")[["web_units","retail","other","web_discount"]].round(2).to_string())
c=pd.read_parquet("data/daily_channel.parquet"); c["m"]=c.date.dt.to_period("M")
cm=c.pivot_table(index="channel",columns="m",values="units",aggfunc="sum").fillna(0)
cols=[pd.Period(x) for x in ["2024-08","2024-09","2024-10","2024-11","2024-12","2025-01","2025-07","2025-08","2025-09"]]
print(cm.reindex(columns=cols).round(0).sort_values(cols[-1],ascending=False).head(20).to_string())
st=pd.read_parquet("data/daily_store.parquet"); st["m"]=st.date.dt.to_period("M")
sm=st.pivot_table(index="store",columns="m",values="units",aggfunc="sum")/st.groupby("m").date.nunique()
print(sm.reindex(columns=cols).round(0).to_string())
