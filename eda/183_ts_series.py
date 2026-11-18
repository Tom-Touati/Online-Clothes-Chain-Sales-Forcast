"""D192 (fork): daily unit series per item (all channels, non-cancelled), 2024-01-01 .. 2026-06-30, for the time-series foundation
models on the remote machine. Writes data/fk_daily.npz (items, start, Y: items x days, float32)."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
items = np.sort(pd.read_parquet("data/items.parquet").style_color.values)
d = pd.read_parquet("data/daily_item_chan.parquet", columns=["style_color", "ny_date", "cancelled", "qty"]); d = d[~d.cancelled]
d["ny_date"] = pd.to_datetime(d.ny_date); D = d.groupby(["style_color", "ny_date"]).qty.sum().unstack(fill_value=0)
days = pd.date_range("2024-01-01", "2026-06-30"); D = D.reindex(index=items, columns=days, fill_value=0).fillna(0)
np.savez_compressed("data/fk_daily.npz", items=items.astype(str), start=np.array("2024-01-01"), Y=D.values.astype(np.float32))
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
chk = D.T.groupby(D.columns.to_period("M").astype(str)).sum().T.reindex(columns=["2025-06", "2025-11"]); print("check vs monthly_item:", np.abs(chk.values - M.reindex(items)[["2025-06", "2025-11"]].fillna(0).values).max())
print(D.shape, int(D.values.sum()))
