"""D150: compact extracts of the order lines for the basket and regional analyses (online channel only: edikted.com + TikTok; every
online line has a US state). Non-cancelled lines.
  data/basket_pairs_month.parquet : per NY month, item pair (i < j, both among the 1,500 items) -> number of orders containing both
  data/basket_item_month.parquet  : per month and item: orders, units, orders with another item (multi-item orders)
  data/state_item_month.parquet   : per month, state and item: units (online)
Items are coded by their row in data/items.parquet (sorted style_color)."""
import sys, numpy as np, pandas as pd, time
sys.stdout.reconfigure(encoding="utf-8"); t0 = time.time()
items = np.sort(pd.read_parquet("data/items.parquet").style_color.values); code = {s: i for i, s in enumerate(items)}
d = pd.read_parquet("Dataset.parquet", columns=["order_name", "ny_date", "style_color", "qty_ordered", "class", "state", "cancelled"])
d = d[(~d.cancelled) & (d["class"] == "ONLINE")]
d["it"] = d.style_color.map(code); d = d[d.it.notna()]; d["it"] = d.it.astype(np.int16)
d["m"] = pd.PeriodIndex(pd.to_datetime(d.ny_date), freq="M").astype(str)
print("online lines", len(d), round(time.time() - t0), "s", flush=True)
S = d.groupby(["m", "state", "it"]).qty_ordered.sum().reset_index(name="units"); S.to_parquet("data/state_item_month.parquet", index=False)
o = d.groupby(["order_name", "it"]).agg(m=("m", "first"), u=("qty_ordered", "sum")).reset_index()
n = o.groupby("order_name").it.transform("size"); o["multi"] = n > 1
B = o.groupby(["m", "it"]).agg(orders=("order_name", "size"), units=("u", "sum"), multi_orders=("multi", "sum")).reset_index(); B.to_parquet("data/basket_item_month.parquet", index=False)
mo = o[o.multi][["order_name", "it", "m"]]
print("multi-item order rows", len(mo), "orders", mo.order_name.nunique(), round(time.time() - t0), "s", flush=True)
rows = []
for m, g in mo.groupby("m"):
    g = g.sort_values(["order_name", "it"]); a = g.merge(g, on="order_name", suffixes=("", "_b")); a = a[a.it < a.it_b]
    rows.append(a.groupby(["it", "it_b"]).size().reset_index(name="n").assign(m=m))
    print(m, len(a), flush=True)
P = pd.concat(rows, ignore_index=True); P.to_parquet("data/basket_pairs_month.parquet", index=False)
print("pairs rows", len(P), "total pair-orders", int(P.n.sum()), round(time.time() - t0), "s")
