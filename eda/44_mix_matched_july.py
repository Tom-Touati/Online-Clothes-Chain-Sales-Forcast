"""Mix-matched July/June benchmark for the 3-12 month targets: 2025 peers (3-12 months at June 2025) in the same macro category x peak-position cell,
weighted by the 2026 targets' June units. Also the same for Jul-Oct/June. Compared with the final forecast (data/final_forecast_q4.csv)."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
M = pd.read_parquet("data/monthly_item.parquet"); cols = list(M.columns); it = pd.read_parquet("data/items.parquet").set_index("style_color")
start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
MAC = {"Top": "top", "Tank Top": "top", "T-Shirt": "top", "Crop Top": "top", "Corset": "top", "Strapless": "top", "Bodysuit": "top", "Shirt": "top", "Knitted Top": "top",
       "Jeans": "bottom", "Shorts": "bottom", "Skirt": "bottom", "Skort": "bottom", "Pants": "bottom", "Short Jeans": "bottom", "Sweatpants": "warm", "Hoodie": "warm", "Sweater": "warm",
       "Jacket": "warm", "Cardigan": "warm", "Dress": "dress"}
def cells(ids, y):
    k = cols.index(f"{y}-06"); pos = M.loc[ids, f"{y}-06"] / M.loc[ids, cols[:k + 1]].max(axis=1)
    return pd.DataFrame({"mac": it.product_type.reindex(ids).map(MAC).fillna("other").values, "pos": pd.cut(pos, [0, .35, .7, 10], labels=["<35%", "35-70%", ">=70%"]).astype(str).values}, index=ids)
age25 = (pd.Timestamp("2025-06-30") - start.reindex(M.index)).dt.days / 30.44
p = M.index[((age25 >= 3) & (age25 < 12)).values & (M["2025-06"] >= 50).values]; C = cells(p, 2025)
H = pd.DataFrame({"jun": M.loc[p, "2025-06"], "jul": M.loc[p, "2025-07"], "jo": M.loc[p, ["2025-07", "2025-08", "2025-09", "2025-10"]].sum(axis=1)}).join(C)
cellr = H.groupby(["mac", "pos"]).agg(n=("jun", "size"), jun=("jun", "sum"), jul=("jul", "sum"), jo=("jo", "sum")); cellr["jul_jun"] = cellr.jul / cellr.jun; cellr["jo_jun"] = cellr.jo / cellr.jun
O = pd.read_csv("data/final_forecast_q4.csv", index_col=0); t = O[(O.age_m >= 3) & (O.age_m < 12)].index
T = pd.DataFrame({"jun": M.loc[t, "2026-06"], "fjul": O.loc[t, "f_jul"], "fjo": O.loc[t, "final_total"]}).join(cells(t, 2026))
g = T.groupby(["mac", "pos"]).agg(n=("jun", "size"), jun=("jun", "sum"), fjul=("fjul", "sum"), fjo=("fjo", "sum")).join(cellr[["n", "jul_jun", "jo_jun"]], rsuffix="_2025")
g["fc_jul_jun"] = g.fjul / g.jun; g["fc_jo_jun"] = g.fjo / g.jun
print(g.round(2).to_string())
ok = g.jul_jun.notna()
print(f"\n3-12 month targets: forecast July/June {T.fjul.sum() / T.jun.sum():.3f} | mix-matched 2025 peer benchmark {(g.jun[ok] * g.jul_jun[ok]).sum() / g.jun[ok].sum():.3f} (cells with peers cover {g.jun[ok].sum() / g.jun.sum():.0%} of June units)")
print(f"3-12 month targets: forecast Jul-Oct/June {T.fjo.sum() / T.jun.sum():.3f} | mix-matched 2025 peer benchmark {(g.jun[ok] * g.jo_jun[ok]).sum() / g.jun[ok].sum():.3f}")
print(f"unmatched benchmark (age only): July/June {H.jul.sum() / H.jun.sum():.3f}, Jul-Oct/June {H.jo.sum() / H.jun.sum():.3f}")
