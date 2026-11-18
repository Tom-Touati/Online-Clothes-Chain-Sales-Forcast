"""D070 (REJECTED variant, not used in the submission): move the 2026 July split to the same-mix age benchmark (totals unchanged).
Evidence (eda/51_rules_any_origin.py, origins May/Jun/Jul 2025): the prior-year same-mix age benchmark predicted the
first-month/origin-month ratio with errors +2.8% / -1.5% / +6.3% (mean |err| 3.5%) against the model's +15% / -6.6% / +4.0% (8.6%).
Benchmark for 2026 = mean of the 2024 and 2025 age-group July/June ratios weighted by the targets' June 2026 units, computed on the
targets without a hand rule (World Cup items and S16079 keep their rule). July of those items is scaled by one common factor;
Aug-Oct of each item are scaled down so every item total (and hence every confidence level) is unchanged."""
import pandas as pd, numpy as np, sys
sys.stdout.reconfigure(encoding="utf-8")
O = pd.read_csv("data/final_forecast_q4.csv", index_col=0); M = pd.read_parquet("data/monthly_item.parquet")
it = pd.read_parquet("data/items.parquet").set_index("style_color")
start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
grp = lambda a: np.where(a < 3, "<3m", np.where(a < 12, "3-12m", "12m+"))
R = {}
for y in (2024, 2025):
    age = (pd.Timestamp(f"{y}-06-30") - start.reindex(M.index)).dt.days / 30.44
    ids = M.index[(M[f"{y}-06"] >= 50).values & (age >= 1).values]; g = grp(age[ids].values)
    R[y] = {k: M.loc[ids[g == k], f"{y}-07"].sum() / M.loc[ids[g == k], f"{y}-06"].sum() for k in ["<3m", "3-12m", "12m+"]}
print("July/June by age group:", {y: {k: round(v, 3) for k, v in r.items()} for y, r in R.items()})
RULE = ["S24840_LIGHT-GRAY-MELANGE", "S24847_NAVY", "S25039_YELLOW", "S16079_DARK-GRAY"]
free = ~O.index.isin(RULE); jun = M.loc[O.index, "2026-06"]
g26 = pd.Series(grp(O.age_m.values), index=O.index)
bench = {y: sum(R[y][k] * jun[free & (g26 == k)].sum() for k in R[y]) / jun[free].sum() for y in R}
B = (bench[2024] + bench[2025]) / 2
cur = O.loc[free, "f_jul"].sum() / jun[free].sum(); f = B / cur
print(f"items without hand rule: July/June now {cur:.3f}; benchmark 2024 ratios {bench[2024]:.3f}, 2025 ratios {bench[2025]:.3f}, mean {B:.3f}; factor {f:.3f}")
X = O[["f_jul", "f_aug", "f_sep", "f_oct"]].astype(float); tot = X.sum(axis=1)
jul = X.f_jul.where(~free, X.f_jul * f)
rest = X[["f_aug", "f_sep", "f_oct"]].mul((tot - jul) / X[["f_aug", "f_sep", "f_oct"]].sum(axis=1), axis=0)
N = pd.concat([jul.rename("f_jul"), rest], axis=1).round(0); N["f_oct"] += tot - N.sum(axis=1)
assert (N >= 0).all().all() and (N.sum(axis=1) == O.final_total).all()
out = O.copy(); out[["f_jul", "f_aug", "f_sep", "f_oct"]] = N.values; out["july_bench_factor"] = np.where(free, f, 1.0)
out.to_csv("data/rejected_july_benchmark_variant.csv")   # rejected in D070: breaks the Aug/Jul shape
print("July/June all targets", round(O.f_jul.sum() / jun.sum(), 3), "->", round(out.f_jul.sum() / jun.sum(), 3))
print("monthly before", O[["f_jul", "f_aug", "f_sep", "f_oct"]].sum().astype(int).to_dict()); print("monthly after ", out[["f_jul", "f_aug", "f_sep", "f_oct"]].sum().astype(int).to_dict(), "total", int(out.final_total.sum()))
s = out[["f_jul", "f_aug", "f_sep", "f_oct"]].sum(); print("shares", (s / s.sum()).round(3).to_dict(), "Aug/Jul", round(s.f_aug / s.f_jul, 3), "Sep/Aug", round(s.f_sep / s.f_aug, 3))
print("max July share of an item", round((out.f_jul / out.final_total).max(), 3), "| conf unchanged:", (out.conf == O.conf).all())
