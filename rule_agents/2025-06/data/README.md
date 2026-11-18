# Data snapshot at 2025-06-30 (end of 2025-06)

Everything here is known on 2025-06-30. Nothing after that date exists in these files.
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
| `to_forecast.csv` | the 768 style_colors to forecast |

Target: units per item for each of 2025-07, 2025-08, 2025-09, 2025-10 (all channels, non-cancelled, NY calendar month). Items are scored mainly on their 4-month total.
The style of an item is the part of style_color before the underscore.
