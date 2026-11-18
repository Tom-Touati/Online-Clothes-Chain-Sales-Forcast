"""Evidence for the two judgement adjustments: (1) 2026 campaign calendar vs the same windows in 2024/2025, (2) post-event decay for event items."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[~d.cancelled]; d["ny_date"] = pd.to_datetime(d.ny_date)
w = d[d.chan == "edikted.com"].groupby("ny_date").agg(q=("qty", "sum"), dw=("disc_w", "sum")); w["disc"] = w.dw / w.q
# like-for-like daily volume: items live before 2024-06-01 only, so growth from new items does not show up as campaign lift
it = pd.read_parquet("data/items.parquet"); it["fs"] = pd.to_datetime(it.first_sale)
old = set(it[it.fs < "2024-05-01"].style_color)
wl = d[(d.chan == "edikted.com") & d.style_color.isin(old)].groupby("ny_date").qty.sum()
w["rel"] = wl / wl.rolling(43, center=True, min_periods=15).median()

plan = [("Summer Sale (incl. 4th of July)", "06-28", "07-19"), ("Back to School", "07-20", "08-31"), ("Clearance", "08-12", "08-19"),
        ("Labor Day Sale", "09-03", "09-08"), ("Sale up to 80% (Sep)", "09-24", "09-29"), ("Sale up to 80% (Oct)", "10-07", "10-18")]
rows = []
for y in (2024, 2025):
    for n, a, b in plan:
        x = w.loc[f"{y}-{a}":f"{y}-{b}"]
        rows.append(dict(year=y, campaign=n, mean_disc=x.disc.mean(), days_ge585=int((x.disc >= .585).sum()), days=len(x), rel_units=x.rel.mean()))
print("== the 2026 campaign windows replayed on 2024 and 2025 (edikted.com; rel_units = like-for-like units / 43-day centred median)")
print(pd.DataFrame(rows).round(3).to_string(index=False))
for y in (2024, 2025):
    x = w.loc[f"{y}-07-01":f"{y}-10-31"]; print(y, "Jul-Oct mean disc by month:", x.disc.groupby(x.index.month).mean().round(3).to_dict(), "| rel units by month:", x.rel.groupby(x.index.month).mean().round(3).to_dict())
# lift as a function of realised discount (like-for-like), all days 2024-2026
w["dev"] = w.disc - w.disc.rolling(43, center=True, min_periods=15).median()
print("like-for-like lift by discount deviation from local median:"); print(w.groupby(pd.cut(w.dev, [-1, -.05, -.02, .02, .04, .06, 1]), observed=True).rel.agg(["size", "mean", "median"]).round(2))
# pull-forward: units in the 7 days after deep runs vs local median
w["deep"] = w.disc >= .585; after = w.deep.shift(1).rolling(7, min_periods=1).max().fillna(0).astype(bool) & ~w.deep
print("rel units on deep days:", round(w[w.deep].rel.mean(), 2), "| in the 7 days after a deep run:", round(w[after].rel.mean(), 2), "| other days:", round(w[~w.deep & ~after].rel.mean(), 2))

# ---- event decay: items tagged with a dated event, units by month relative to the event month
M = pd.read_parquet("data/monthly_item.parquet")
occ = it.set_index("style_color").occasion.fillna("")
for tag, ev, cols in [("4th of July", "2025-07", ["2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11"]),
                      ("4th of July", "2024-07", ["2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11"]),
                      ("St. Patrick", "2025-03", ["2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07"]),
                      ("St. Patrick", "2026-03", ["2026-02", "2026-03", "2026-04", "2026-05", "2026-06"])]:
    ids = occ[occ.str.contains(tag)].index.intersection(M.index)
    pub = pd.to_datetime(it.set_index("style_color").loc[ids, "published_date"], utc=True).dt.tz_localize(None)
    for lab, sel in [("launched <=2 months before event", pub >= pd.Period(ev).start_time - pd.DateOffset(months=2)), ("older items", pub < pd.Period(ev).start_time - pd.DateOffset(months=2))]:
        s = M.loc[ids[sel.values], cols].sum()
        if s[ev] > 0: print(f"{tag:12s} event {ev} | {lab:32s} n={int(sel.sum()):3d} | units / event month:", (s / s[ev]).round(2).to_dict())
print("World Cup-tagged items: n =", int(occ.str.contains("World Cup").sum()))
wc = ["S24840_LIGHT-GRAY-MELANGE", "S24847_NAVY", "S25039_YELLOW"]
x = d[d.style_color.isin(wc)].groupby(["style_color", pd.Grouper(key="ny_date", freq="W-SUN")]).qty.sum().unstack(0).loc["2026-05-24":]
print("World Cup targets, weekly units:"); print(x.to_string())
od = occ[occ.str.contains("World Cup")].index
print("all World Cup-tagged items, monthly units 2026:"); print(M.loc[M.index.isin(od), ["2026-03", "2026-04", "2026-05", "2026-06"]].assign(pub=it.set_index("style_color").loc[M.index[M.index.isin(od)], "published_date"].astype(str).str[:10]).to_string())
x = d[d.style_color == "S16079_DARK-GRAY"].groupby(pd.Grouper(key="ny_date", freq="MS")).qty.sum(); print("S16079_DARK-GRAY monthly:", x.loc["2025-01":].astype(int).to_dict())
