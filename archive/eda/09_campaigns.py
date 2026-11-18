import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[(~d.cancelled) & (d.chan == "edikted.com")]; d["ny_date"] = pd.to_datetime(d.ny_date)
s = d.groupby("ny_date").agg(q=("qty", "sum"), dw=("disc_w", "sum")); s["disc"] = s.dw / s.q
s["rel"] = s.q / s.q.rolling(29, center=True, min_periods=10).median()
s["tier"] = pd.cut(s.disc, [0, .5, .56, .6, 1], labels=["<50", "50-56", "56-60", "60+"])
for y in (2024, 2025):
    x = s[(s.index >= f"{y}-07-01") & (s.index <= f"{y}-10-31")]
    t = x.groupby([x.index.month, "tier"], observed=False).size().unstack(fill_value=0); t["mean_disc"] = x.groupby(x.index.month).disc.mean().round(3); t["units_k"] = (x.groupby(x.index.month).q.sum() / 1e3).round(0)
    print(f"== {y}: days per discount tier, edikted.com"); print(t.to_string())
    r = x.disc.round(2); runs = (r.ge(0.585) != r.ge(0.585).shift()).cumsum(); hi = x[r.ge(0.585)].groupby(runs).agg(start=("disc", lambda v: v.index.min().date()), end=("disc", lambda v: v.index.max().date()), days=("disc", "size"), disc=("disc", "mean"), rel_units=("rel", "mean")).round(2)
    print("deep-discount runs (>=58.5%):"); print(hi[hi.days >= 2].to_string(index=False))
plan = [("Summer Sale", "2026-06-28", "2026-07-19"), ("Back to School", "2026-07-20", "2026-08-31"), ("Clearance", "2026-08-12", "2026-08-19"), ("Labor Day", "2026-09-03", "2026-09-08"), ("Sale80", "2026-09-24", "2026-09-29"), ("Sale80", "2026-10-07", "2026-10-18")]
deep = pd.Series(0, index=pd.date_range("2026-07-01", "2026-10-31"))
for n, a, b in plan:
    if n != "Back to School": deep[(deep.index >= a) & (deep.index <= b)] = 1
print("== 2026 plan: days in a deep (60-80%+) campaign per month:", deep.groupby(deep.index.month).sum().to_dict())
for y in (2024, 2025):
    x = s[(s.index >= f"{y}-07-01") & (s.index <= f"{y}-10-31")]; print(y, "days >=58.5% per month:", x.disc.ge(0.585).groupby(x.index.month).sum().to_dict())
