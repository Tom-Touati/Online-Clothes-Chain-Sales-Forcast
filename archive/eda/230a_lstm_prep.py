"""Fork "fashion" (D404; user, 18 Nov: "train an lstm model on the gpu server that take daily data from the last 2 months and tries to compute
the next month"). Inputs for eda/230b_lstm_gpu.py:
  daily    units per item x day x channel group (web = edikted.com, tiktok, retail = own stores, partners = all DROP partners and Flip),
           cancelled lines dropped, 1 Jan 2024 .. 30 Jun 2026; and the units-weighted average web discount per item x day
  items    product type code and first-sale day per item (static context for the second version)
  eval     the month-ends to score: every backtest origin 2025-02 .. 2026-02 x its scored items (data/dec_base_nx5r26.parquet) and June 2026
           x all items, with the next calendar month's actual units (y1, NaN for July 2026)
Writes data/lstm_inputs.npz (float16 daily array + metadata)."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[~d.cancelled.astype(bool)]
d["g"] = np.where(d.chan == "edikted.com", 0, np.where(d.chan == "Tiktok", 1, np.where(d["class"] == "RETAIL", 2, 3)))
items = np.sort(pd.read_parquet("data/items.parquet").index.values if "style_color" not in pd.read_parquet("data/items.parquet").columns else pd.read_parquet("data/items.parquet").style_color.values)
days = pd.date_range("2024-01-01", "2026-06-30", freq="D"); ii = pd.Series(np.arange(len(items)), index=items); di = pd.Series(np.arange(len(days)), index=days)
d["ny_date"] = pd.to_datetime(d.ny_date); d = d[d.style_color.isin(items)]
X = np.zeros((len(items), len(days), 5), np.float32)
a = d.groupby(["style_color", "ny_date", "g"]).qty.sum().reset_index()
np.add.at(X, (ii[a.style_color].values, di[a.ny_date].values, a.g.values.astype(int)), a.qty.values)
w = d[d.chan == "edikted.com"].groupby(["style_color", "ny_date"]).agg(dw=("disc_w", "sum"), q=("qty", "sum")).reset_index()
X[ii[w.style_color].values, di[w.ny_date].values, 4] = np.clip(w.dw.values / np.maximum(w.q.values, 1), 0, 1)
it = pd.read_parquet("data/items.parquet"); it = it.set_index("style_color") if "style_color" in it.columns else it; it = it.reindex(items)
ptype = pd.Categorical(it.product_type.fillna("na")); first = pd.to_datetime(it.first_sale).fillna(pd.Timestamp("2024-01-01")).values
first_ix = np.clip((first - np.datetime64("2024-01-01")).astype("timedelta64[D]").astype(int), -3650, len(days))
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
B = pd.read_parquet("data/dec_base_nx5r26.parquet")[["origin", "style_color"]]
ev = pd.concat([B, pd.DataFrame({"origin": "2026-06", "style_color": items})]).drop_duplicates()
ev = ev[ev.origin != "2026-06"].pipe(lambda e: pd.concat([e[e.origin <= "2026-02"], ev[ev.origin == "2026-06"]]))
nxt = lambda o: str(pd.Period(o, "M") + 1)
ev["y1"] = [float(M.loc[s, nxt(o)]) if (s in M.index and nxt(o) in M.columns) else np.nan for o, s in zip(ev.origin, ev.style_color)]
print("daily array", X.shape, "| units", int(X[:, :, :4].sum()), "| eval rows", len(ev), "| origins", ev.origin.nunique(), "| y1 known", int(ev.y1.notna().sum()))
np.savez_compressed("data/lstm_inputs.npz", X=X.astype(np.float16), items=items.astype(str), day0="2024-01-01", ptype=ptype.codes.astype(np.int16), ptype_names=np.array(ptype.categories, dtype=str),
                    first_ix=first_ix.astype(np.int32), ev_origin=ev.origin.values.astype(str), ev_item=ev.style_color.values.astype(str), ev_y1=ev.y1.values.astype(np.float32))
