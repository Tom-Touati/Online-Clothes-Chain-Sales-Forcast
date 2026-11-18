"""Why was Nov 2025 under-forecast by a third? Site-level view: monthly units of items live, site discount, deep-discount days,
and the backtest P50 error by month, so we can see whether the miss is site-wide (campaign/calendar) or mix-driven."""
import sys, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[~d.cancelled]; d["ny_date"] = pd.to_datetime(d.ny_date)
on = d[d["class"] == "ONLINE"]
s = on.groupby("ny_date").agg(q=("qty", "sum"), dw=("disc_w", "sum")); s["disc"] = s.dw / s.q
tot = d.groupby("ny_date").qty.sum()
m = pd.DataFrame({"units_k": tot.resample("MS").sum() / 1e3, "disc": (s.dw.resample("MS").sum() / s.q.resample("MS").sum()),
                  "deep_days": s.disc.ge(0.585).resample("MS").sum(), "disc_p90": s.disc.resample("MS").quantile(0.9)})
B = pd.read_parquet("data/best_backtest_preds.parquet"); B["tm"] = (pd.PeriodIndex(B.origin, freq="M") + B.h).astype(str)
e = B.groupby(["tm", "h"]).apply(lambda g: g.pred.sum() / g.y.sum() - 1).unstack().round(3)
m.index = m.index.to_period("M").astype(str)
print(m.join(e).round(3).to_string())
