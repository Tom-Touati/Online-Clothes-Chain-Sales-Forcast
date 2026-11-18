"""D252b: day-level promotion response (user: "we want to use the different days of the week"; daily data, up to the origin only).
Site discount per day = unit-weighted discount of edikted.com online lines (disc_w / qty). Promo days = days in the 365 days up to the
origin whose site discount is in that year's top quartile.
  promo_lift     the item's mean daily units on promo days / on other days, last 180 days (log, shrunk: +3 units each side)
  promo_lift_wk  the same restricted to weekends vs weekdays of the same kind (Fri-Sun promo vs Fri-Sun other), to separate weekday mix
  bfcm_ly        the item's units over last year's Black Friday - Cyber Monday (Fri-Mon) per day / its mean daily units over the 28
                 days before Thanksgiving (log, +1 smoothing); items alive then only
  bfcm_ly_type   the same for the item's product type (all items of the type)
  dow_cv         coefficient of variation of the item's weekday profile, last 91 days (how weekday-driven the item is)
Writes data/nf3_feats.parquet."""
import sys, numpy as np, pandas as pd
from datetime import date, timedelta
sys.stdout.reconfigure(encoding="utf-8")
T = pd.read_parquet("data/fk_explain_table.parquet")[["origin", "style_color", "ptype"]]
items = np.sort(pd.read_parquet("data/items.parquet").style_color.values); ix = {s: i for i, s in enumerate(items)}
pt = pd.read_parquet("data/items.parquet").set_index("style_color").product_type.reindex(items).fillna("na").values
d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[~d.cancelled]; d["ny_date"] = pd.to_datetime(d.ny_date)
D0 = pd.Timestamp("2024-01-01"); days = pd.date_range(D0, "2026-06-30"); nd = len(days)
Dm = d.groupby(["style_color", "ny_date"]).qty.sum().unstack(fill_value=0).reindex(index=items, columns=days, fill_value=0).values.astype(np.float32)
on = d[d["class"] == "ONLINE"].groupby("ny_date")[["disc_w", "qty"]].sum().reindex(days).fillna(0); sdisc = (on.disc_w / on.qty.clip(lower=1)).values; del d
wk = days.dayofweek.values >= 4
def thanks(y): d0 = date(y, 11, 1); return d0 + timedelta((3 - d0.weekday()) % 7 + 21)
DT = pd.Series(Dm.sum(0)); TY = pd.DataFrame(Dm).groupby(pt).sum().values; tix = {t: i for i, t in enumerate(pd.Series(pt).groupby(pt).size().index)}
rows = []
for o in sorted(T.origin.unique()):
    P = pd.Period(o, "M"); e = (P.end_time.normalize() - D0).days; sub = T[T.origin == o]; idx = np.array([ix[s] for s in sub.style_color])
    R = pd.DataFrame({"origin": o, "style_color": sub.style_color.values})
    y0 = max(e - 364, 0); thr = np.quantile(sdisc[y0:e + 1], 0.75); a = max(e - 179, 0); pr = sdisc[a:e + 1] >= thr; X = Dm[idx, a:e + 1]; w_ = wk[a:e + 1]
    rate = lambda m: (X[:, m].sum(1) + 3) / (m.sum() + 3)
    R["promo_lift"] = np.log(rate(pr) / rate(~pr)) if pr.any() and (~pr).any() else np.nan
    R["promo_lift_wk"] = np.log(rate(pr & w_) / rate(~pr & w_)) if (pr & w_).any() and (~pr & w_).any() else np.nan
    th = thanks(P.year - 1 if P.month < 12 else P.year); tpos = (pd.Timestamp(th) - D0).days
    if 28 <= tpos and tpos + 4 <= e:
        bf = Dm[idx, tpos + 1:tpos + 5].sum(1) / 4; pre = Dm[idx, tpos - 28:tpos].sum(1) / 28
        R["bfcm_ly"] = np.where(pre * 28 >= 10, np.log((bf + 1) / (pre + 1)), np.nan)
        tb = TY[:, tpos + 1:tpos + 5].sum(1) / 4; tp = TY[:, tpos - 28:tpos].sum(1) / 28; tl = np.log((tb + 1) / (tp + 1))
        R["bfcm_ly_type"] = [tl[tix[pt[i]]] for i in idx]
    else: R["bfcm_ly"] = np.nan; R["bfcm_ly_type"] = np.nan
    Xw = Dm[idx, e - 90:e + 1]; dw = days.dayofweek.values[e - 90:e + 1]; prof = np.stack([Xw[:, dw == k].mean(1) for k in range(7)], 1)
    R["dow_cv"] = np.where(Xw.sum(1) >= 30, prof.std(1) / np.maximum(prof.mean(1), 1e-6), np.nan)
    rows.append(R)
NF = pd.concat(rows, ignore_index=True); NF.to_parquet("data/nf3_feats.parquet")
print(NF.drop(columns=["origin", "style_color"]).describe().T[["count", "mean", "min", "50%", "max"]].round(3).to_string())
