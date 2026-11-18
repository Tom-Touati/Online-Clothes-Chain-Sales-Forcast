"""Fork 2 (user, 18 Nov: "we need to add features like which discounts and how many days for each of the horizon months, we need to use it
directly in the models"). Known-future campaign features per origin x item for the four target months h = 1..4.
History (backtest origins): the site's realised calendar on edikted.com stands in for the plan (major campaigns are planned ahead):
  cdays_h     campaign days in month h: days whose unit-weighted edikted.com discount is >= 58.5% (D016 / D030 'deep day')
  cdisc_h     mean edikted.com discount on the month's campaign days (0 when none) -- 'which discount'
  sdisc_h     mean edikted.com discount over all days of month h (the usual rotating 30-50% plus campaigns)
  cdays_tot, cdisc_tot  over the 4 months
  dcd_h, dcd_tot   change in campaign days against the same months a year earlier (what last year's rows cannot teach the model);
                   NaN when last year is before the data start
  web_sh      the item's edikted.com share of units in the 91 days before the origin (campaigns run on the US site)
  x_cmp       dcd_tot / days in the window x web_sh (the exposure of D389v); x_h = dcd_h / days in month h x web_sh
2026-06 origin (the plan sheet): recurring windows repeat 2025's realised campaign days and depth by month (Summer Sale incl. the 4th of July
push, Back to School, Labor Day, 7-18 October ran on the same dates in 2024 and 2025, D030); the two new windows add full campaign days at
2025's mean campaign-day depth: Clearance 12-19 Aug (8 days), 24-29 Sep (6 days). Writes data/campaign_feats.parquet."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
d = pd.read_parquet("data/daily_item_chan.parquet", columns=["ny_date", "style_color", "chan", "qty", "disc_w", "cancelled"]); d = d[~d.cancelled]; d["ny_date"] = pd.to_datetime(d.ny_date)
web = d[d.chan == "edikted.com"]; S = web.groupby("ny_date").agg(q=("qty", "sum"), dw=("disc_w", "sum")); S["disc"] = S.dw / S.q; S["deep"] = S.disc >= 0.585
S["m"] = S.index.to_period("M")
M = S.groupby("m").agg(cdays=("deep", "sum"), sdisc=("disc", "mean"), n=("disc", "size"))
M["cdisc"] = S[S.deep].groupby("m").disc.mean().reindex(M.index).fillna(0)
# 2026 plan months
deep25 = S[S.deep & (S.index.year == 2025)].disc.mean()
for mo, new in [(7, 0), (8, 8), (9, 6), (10, 0)]:
    p25 = pd.Period(f"2025-{mo:02d}", "M"); p26 = pd.Period(f"2026-{mo:02d}", "M"); r = M.loc[p25]
    cd = r.cdays + new; cdisc = (r.cdisc * r.cdays + deep25 * new) / cd if cd > 0 else 0.0
    nd = p26.days_in_month; sd = r.sdisc + (deep25 - r.sdisc) * new / nd   # new campaign days replace ordinary days of 2025's month
    M.loc[p26] = dict(cdays=cd, sdisc=sd, n=nd, cdisc=cdisc)
print("months, campaign days / campaign depth / month depth:"); print(M.loc[[p for p in M.index if p.month in (7, 8, 9, 10)]].round(3).to_string())
T = pd.read_parquet("data/fk_explain_table.parquet", columns=["origin", "style_color"]); T["origin"] = T.origin.astype(str)
it_tot = d.groupby(["style_color", pd.Grouper(key="ny_date", freq="D")]).qty.sum()
rows = []
for o in sorted(T.origin.unique()):
    P = pd.Period(o, "M"); end = P.end_time.normalize(); start = end - pd.Timedelta(days=90)
    w = d[(d.ny_date >= start) & (d.ny_date <= end)]; tq = w.groupby("style_color").qty.sum(); wq = w[w.chan == "edikted.com"].groupby("style_color").qty.sum()
    sh = (wq.reindex(tq.index).fillna(0) / tq.clip(lower=1)).rename("web_sh")
    f = {}
    for h in (1, 2, 3, 4):
        m = P + h; ly = m - 12
        f[f"cdays_h{h}"] = M.cdays.get(m, np.nan); f[f"cdisc_h{h}"] = M.cdisc.get(m, np.nan); f[f"sdisc_h{h}"] = M.sdisc.get(m, np.nan)
        f[f"dcd_h{h}"] = (M.cdays.get(m, np.nan) - M.cdays.get(ly, np.nan)) if ly in M.index else np.nan
        f[f"nd_h{h}"] = m.days_in_month
    nd = sum(f[f"nd_h{h}"] for h in (1, 2, 3, 4)); f["cdays_tot"] = sum(f[f"cdays_h{h}"] for h in (1, 2, 3, 4))
    f["cdisc_tot"] = np.nansum([f[f"cdisc_h{h}"] * f[f"cdays_h{h}"] for h in (1, 2, 3, 4)]) / f["cdays_tot"] if f["cdays_tot"] > 0 else 0.0
    f["dcd_tot"] = sum(f[f"dcd_h{h}"] for h in (1, 2, 3, 4)) if all(np.isfinite(f[f"dcd_h{h}"]) for h in (1, 2, 3, 4)) else np.nan
    items = T[T.origin == o].style_color.values
    R = pd.DataFrame({"origin": o, "style_color": items}); R["web_sh"] = sh.reindex(items).fillna(sh.median() if len(sh) else 0.7).values
    for k, v in f.items(): R[k] = v
    R["x_cmp"] = R.dcd_tot / nd * R.web_sh
    for h in (1, 2, 3, 4): R[f"x_h{h}"] = R[f"dcd_h{h}"] / R[f"nd_h{h}"] * R.web_sh
    rows.append(R.drop(columns=[f"nd_h{h}" for h in (1, 2, 3, 4)]))
F = pd.concat(rows, ignore_index=True); F.to_parquet("data/campaign_feats.parquet", index=False)
print(F.groupby("origin")[["cdays_tot", "dcd_tot", "cdisc_tot"]].first().round(3).to_string()); print(F.shape)
