"""Rule-agent evaluation (D103): one data snapshot per backtest origin, cut at the origin's month end (NY dates).
Writes agent_eval/<origin>/data/ with only information available at that date, plus to_forecast.csv (the backtest item set).
Actual outcomes after the origin are never written into a snapshot; agent_eval/actuals.parquet (for scoring) sits outside them.
python eda/75_agent_snapshots.py 2026-06  builds only the live origin (D105): to_forecast = the items eligible at that origin in data/features.parquet
(the same definition as the backtest sets), and actuals.parquet is left as it is."""
import os, shutil, numpy as np, pandas as pd, pyarrow.parquet as pq, warnings
warnings.filterwarnings("ignore")
ROOT = "agent_eval"; os.makedirs(ROOT, exist_ok=True)
import sys
LIVE = sys.argv[1] if len(sys.argv) > 1 else None
ORIGINS = [LIVE] if LIVE else [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]

d = pd.read_parquet("data/daily_item_chan.parquet"); d["ny_date"] = pd.to_datetime(d.ny_date)
canc = d[d.cancelled].groupby("ny_date").qty.sum()
d = d[~d.cancelled]
d["online"] = np.where(d["class"] == "ONLINE", d.qty, 0.0); d["retail"] = np.where(d["class"] == "RETAIL", d.qty, 0.0); d["drop"] = np.where(d["class"] == "DROP", d.qty, 0.0)
d["web_q"] = np.where(d.chan == "edikted.com", d.qty, 0.0); d["web_dw"] = np.where(d.chan == "edikted.com", d.disc_w.clip(lower=0), 0.0)
d["ret_dw"] = np.where(d["class"] == "RETAIL", d.disc_w.clip(lower=0), 0.0)
di = d.groupby(["style_color", "ny_date"]).agg(units=("qty", "sum"), online=("online", "sum"), retail=("retail", "sum"), drop=("drop", "sum"),
                                              web_units=("web_q", "sum"), web_disc_sum=("web_dw", "sum")).reset_index().rename(columns={"ny_date": "date"})
site = d.groupby("ny_date").agg(units=("qty", "sum"), online=("online", "sum"), retail=("retail", "sum"), drop=("drop", "sum"), web_units=("web_q", "sum"),
                                web_disc_sum=("web_dw", "sum"), retail_disc_sum=("ret_dw", "sum"))
site["cancelled_units"] = canc.reindex(site.index).fillna(0)
site["web_discount"] = (site.web_disc_sum / site.web_units.replace(0, np.nan)).clip(0, 1)
site["retail_discount"] = (site.retail_disc_sum / site.retail.replace(0, np.nan)).clip(0, 1)
site = site.drop(columns=["web_disc_sum", "retail_disc_sum"]).reset_index().rename(columns={"ny_date": "date"})
chan = d.groupby(["ny_date", "chan"]).qty.sum().rename("units").reset_index().rename(columns={"ny_date": "date", "chan": "channel"})
r = pq.read_table("Dataset.parquet", columns=["ny_date", "class", "store", "cancelled", "qty_ordered"]).to_pandas()
r = r[(r["class"] == "RETAIL") & ~r.cancelled]; r["ny_date"] = pd.to_datetime(r.ny_date)
store = r.groupby(["ny_date", "store"]).qty_ordered.sum().rename("units").reset_index().rename(columns={"ny_date": "date"}); del r

it = pd.read_parquet("data/items.parquet")
it["published_date"] = pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
attrs = pd.read_csv("item_attributes.csv")
bt = pd.read_parquet("data/best_backtest_preds.parquet")
first = di.groupby("style_color").date.min()

# actuals for scoring (kept outside every snapshot)
di["month"] = di.date.dt.to_period("M").astype(str)
act = di.groupby(["style_color", "month"]).units.sum().rename("y").reset_index()
if not LIVE: act.to_parquet(f"{ROOT}/actuals.parquet")

for o in ORIGINS:
    end = pd.Period(o, "M").end_time.normalize()
    out = f"{ROOT}/{o}/data"; os.makedirs(out, exist_ok=True)
    x = di[di.date <= end].drop(columns="month")
    x.to_parquet(f"{out}/daily_item.parquet", index=False)
    m = x.assign(month=x.date.dt.to_period("M").astype(str)).groupby(["style_color", "month"])[["units", "online", "retail", "drop"]].sum().reset_index()
    m.to_parquet(f"{out}/monthly_item.parquet", index=False)
    site[site.date <= end].to_parquet(f"{out}/daily_site.parquet", index=False)
    chan[chan.date <= end].to_parquet(f"{out}/daily_channel.parquet", index=False)
    store[store.date <= end].to_parquet(f"{out}/daily_store.parquet", index=False)
    fs = x.groupby("style_color").date.agg(["min", "max"]).rename(columns={"min": "first_sale", "max": "last_sale"})
    ii = it[["style_color", "product_name", "product_type", "product_collection", "published_date", "color", "color_num", "occasion"]].set_index("style_color").join(fs, how="left")
    ii = ii[(ii.published_date <= end) | ii.first_sale.notna()]
    ii.loc[ii.published_date > end, "published_date"] = pd.NaT   # a publish date after the cutoff is not known yet (item sold early)
    ii.reset_index().to_csv(f"{out}/items.csv", index=False)
    attrs[attrs.style_color.isin(ii.index)].to_csv(f"{out}/item_attributes.csv", index=False)
    src = pd.read_parquet("data/features.parquet", columns=["origin", "style_color"]) if LIVE else bt
    tf = pd.DataFrame({"style_color": sorted(src[src.origin == o].style_color.unique())})
    months = [str(pd.Period(o, "M") + h) for h in range(1, 5)]
    tf.to_csv(f"{out}/to_forecast.csv", index=False)
    with open(f"{out}/README.md", "w", encoding="utf-8") as f:
        f.write(f"""# Data snapshot at {end.date()} (end of {o})

Everything here is known on {end.date()}. Nothing after that date exists in these files.
Units = quantity ordered on non-cancelled order lines, by New York calendar date. Retail store sales are class RETAIL; DROP = dropship marketplaces.

| File | Content |
|---|---|
| `daily_item.parquet` | style_color, date, units (all channels), online, retail, drop, web_units (edikted.com), web_disc_sum (sum of discount fraction over edikted.com units; web_disc_sum / web_units = unit-weighted discount) |
| `monthly_item.parquet` | style_color, month (YYYY-MM), units, online, retail, drop |
| `daily_site.parquet` | date, units, online, retail, drop, web_units, cancelled_units, web_discount, retail_discount (whole site) |
| `daily_channel.parquet` | date, channel, units |
| `daily_store.parquet` | date, store, units (retail stores) |
| `items.csv` | item master: style_color, product_name, product_type, product_collection, published_date (NY date), color, color_num (1 = first colour of the style), occasion (tags separated by '; '), first_sale, last_sale (within this snapshot) |
| `item_attributes.csv` | tagger attributes per style_color (fabric, silhouette, length, aesthetics, occasion tags ...) |
| `to_forecast.csv` | the {len(tf)} style_colors to forecast |

Target: units per item for each of {', '.join(months)} (all channels, non-cancelled, NY calendar month). Items are scored mainly on their 4-month total.
The style of an item is the part of style_color before the underscore.
""")
    print(o, "items", len(ii), "to forecast", len(tf), "daily rows", len(x))
