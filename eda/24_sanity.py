"""Common-sense audit of the final forecast (data/final_forecast_q.csv) against the history of the same items and of comparable items.
Read-only: prints findings, writes data/sanity_flags.csv."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
import sys
O = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/final_forecast_q.csv", index_col=0); T = list(O.index)
M = pd.read_parquet("data/monthly_item.parquet")
it = pd.read_parquet("data/items.parquet").set_index("style_color")
it["start"] = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
mo = lambda y, ms: [f"{y}-{m:02d}" for m in ms]
F = O[["f_jul", "f_aug", "f_sep", "f_oct"]]; F.columns = ["Jul", "Aug", "Sep", "Oct"]

print("== 1 PORTFOLIO")
tot = F.sum(); print("forecast by month:", tot.astype(int).to_dict(), "| total", int(tot.sum()))
print("June 2026 actual x4:", int(M.loc[T, "2026-06"].sum() * 4), "| Mar-Jun 2026 actual:", int(M.loc[T, mo(2026, [3, 4, 5, 6])].sum().sum()))
both = [s for s in T if it.loc[s, "start"] < pd.Timestamp("2025-03-01")]          # live through Mar-Oct 2025
b = M.loc[both]
r_f = F.loc[both].sum().sum() / b[mo(2026, [3, 4, 5, 6])].sum().sum(); r_25 = b[mo(2025, [7, 8, 9, 10])].sum().sum() / b[mo(2025, [3, 4, 5, 6])].sum().sum()
print(f"{len(both)} targets live since before Mar 2025: forecast Jul-Oct 26 / actual Mar-Jun 26 = {r_f:.2f}; same items, actual Jul-Oct 25 / Mar-Jun 25 = {r_25:.2f}")
print(f"   their Jul-Oct 2026 forecast vs Jul-Oct 2025 actual: {F.loc[both].sum().sum() / b[mo(2025, [7, 8, 9, 10])].sum().sum():.2f}x | Mar-Jun 26 vs Mar-Jun 25: {b[mo(2026, [3, 4, 5, 6])].sum().sum() / b[mo(2025, [3, 4, 5, 6])].sum().sum():.2f}x")
# like-for-like monthly shape: all items live before Mar 2025 (whole panel), 2025 shape
lfl = it.index[it.start < pd.Timestamp("2025-03-01")].intersection(M.index)
s25 = M.loc[lfl, mo(2025, [6, 7, 8, 9, 10])].sum(); s24 = M.loc[it.index[it.start < pd.Timestamp("2024-03-01")].intersection(M.index), mo(2024, [6, 7, 8, 9, 10])].sum()
fshape = np.r_[M.loc[T, "2026-06"].sum(), tot.values]
print("month-to-month ratios Jun>Jul>Aug>Sep>Oct | forecast (targets):", np.round(fshape[1:] / fshape[:-1], 2).tolist(),
      "| 2025 like-for-like panel:", np.round(s25.values[1:] / s25.values[:-1], 2).tolist(), "| 2024 like-for-like:", np.round(s24.values[1:] / s24.values[:-1], 2).tolist())

print("\n== 2 ITEMS: growth year on year for mature items")
A = pd.DataFrame(index=T)
A["type"] = O.ptype; A["age"] = O.age_m; A["fc"] = F.sum(axis=1); A["jun"] = M.loc[T, "2026-06"]; A["mj26"] = M.loc[T, mo(2026, [3, 4, 5, 6])].sum(axis=1)
A["mj25"] = M.loc[T, mo(2025, [3, 4, 5, 6])].sum(axis=1); A["jo25"] = M.loc[T, mo(2025, [7, 8, 9, 10])].sum(axis=1)
A["fc_vs_jo25"] = A.fc / A.jo25.replace(0, np.nan); A["spring_yoy"] = A.mj26 / A.mj25.replace(0, np.nan)
A["implied_season"] = A.fc / A.mj26; A["last_season"] = A.jo25 / A.mj25.replace(0, np.nan)
A["season_gap"] = np.log(A.implied_season / A.last_season)
m = A[A.index.isin(both)].copy()
print(f"mature targets (n={len(m)}): forecast vs last Jul-Oct median {m.fc_vs_jo25.median():.2f}x; spring 26 vs spring 25 median {m.spring_yoy.median():.2f}x")
print("forecast Jul-Oct/Mar-Jun 2026 vs the item's own 2025 ratio: median log gap", round(m.season_gap.median(), 2), "| items where forecast season ratio is >1.5x last year's:", int((m.season_gap > np.log(1.5)).sum()), "| <1/1.5x:", int((m.season_gap < -np.log(1.5)).sum()))
print(m.sort_values("season_gap")[["type", "age", "mj25", "jo25", "mj26", "fc", "spring_yoy", "fc_vs_jo25", "last_season", "implied_season"]].round(2).to_string())

print("\n== 3 YOUNG LAUNCHES: forecast vs the best month so far and vs what comparable launches did")
A["best"] = M.loc[T].max(axis=1); A["fc_avg"] = A.fc / 4; A["avg_vs_best"] = A.fc_avg / A.best
young = A[A.age < 6].sort_values("age")
print(young[["type", "age", "jun", "best", "fc", "fc_avg", "avg_vs_best"]].round(2).to_string())
# empirical: items 1-5 months old at end of June 2025 (and at other June-like origins), ratio of next-4-month average to their best month so far
Fq = pd.read_parquet("data/features.parquet"); h1 = Fq[(Fq.h == 1) & Fq.ytot.notna()].copy()
best_so_far = {}
cols = list(M.columns)
for o in h1.origin.unique():
    k = cols.index(o); best_so_far[o] = M[cols[:k + 1]].max(axis=1)
h1["best"] = [best_so_far[o][s] for o, s in zip(h1.origin, h1.style_color)]
h1["avg_vs_best"] = h1.ytot / 4 / h1.best
for lo, hi in [(1, 3), (3, 6), (6, 12), (12, 99)]:
    g = h1[(h1.age_m >= lo) & (h1.age_m < hi) & (h1.best >= 300)]
    gj = g[g.origin.isin(["2024-06", "2025-06"])]
    print(f"history, age {lo}-{hi} m, best month >= 300: next-4-month avg / best month so far: p50 {g.avg_vs_best.median():.2f}, p90 {g.avg_vs_best.quantile(.9):.2f}, p97 {g.avg_vs_best.quantile(.97):.2f}, share > 1: {(g.avg_vs_best > 1).mean():.1%} (n={len(g)}) | June origins only: p50 {gj.avg_vs_best.median():.2f}, p90 {gj.avg_vs_best.quantile(.9):.2f} (n={len(gj)})")
A["ageb"] = pd.cut(A.age, [0, 3, 6, 12, 99], labels=["1-3", "3-6", "6-12", "12+"])
A["pct_vs_history"] = np.nan
for lab, lo, hi in [("1-3", 1, 3), ("3-6", 3, 6), ("6-12", 6, 12), ("12+", 12, 99)]:
    g = h1[(h1.age_m >= lo) & (h1.age_m < hi) & (h1.best >= 300)].avg_vs_best.values; sel = A.ageb == lab
    A.loc[sel, "pct_vs_history"] = [(g <= v).mean() for v in A.loc[sel, "avg_vs_best"]]
print("targets whose forecast avg / best month sits above the 95th percentile of comparable history:"); print(A[A.pct_vs_history > .95][["type", "age", "best", "jun", "fc_avg", "avg_vs_best", "pct_vs_history"]].round(2).to_string())
print("targets below the 5th percentile:"); print(A[A.pct_vs_history < .05][["type", "age", "best", "jun", "fc_avg", "avg_vs_best", "pct_vs_history"]].round(2).to_string())

print("\n== 4 SEASONAL DIRECTION BY TYPE: forecast Oct/Jul vs 2025 like-for-like Oct/Jul")
ty = it.product_type
lflt = M.loc[lfl].groupby(ty.reindex(lfl).values)[mo(2025, [7, 10])].sum(); lflt["hist_oct_jul"] = lflt["2025-10"] / lflt["2025-07"]
fc_t = F.groupby(O.ptype).sum(); fc_t["fc_oct_jul"] = fc_t.Oct / fc_t.Jul; fc_t["n"] = O.ptype.value_counts()
X = fc_t[["n", "fc_oct_jul"]].join(lflt["hist_oct_jul"]); X["ratio"] = X.fc_oct_jul / X.hist_oct_jul
print(X.round(2).sort_values("ratio").to_string())

print("\n== 5 JULY vs JUNE continuity")
A["jul_vs_jun"] = F.Jul / A.jun.replace(0, np.nan)
j25 = M.loc[lfl, "2025-07"] / M.loc[lfl, "2025-06"].replace(0, np.nan); j25 = j25[(M.loc[lfl, "2025-06"] >= 300)]
print(f"history (items live before Mar 2025, June 2025 >= 300): July/June p10 {j25.quantile(.1):.2f} p50 {j25.median():.2f} p90 {j25.quantile(.9):.2f}")
print("targets July/June:", A.jul_vs_jun.describe(percentiles=[.1, .5, .9]).round(2).to_dict())
print(A[(A.jul_vs_jun > j25.quantile(.95)) | (A.jul_vs_jun < j25.quantile(.05))][["type", "age", "jun", "jul_vs_jun"]].join(F.Jul).round(2).to_string())

print("\n== 6 INTERVALS AND CONFIDENCE")
O["answer_pos_ok"] = (O.answer >= O.q50 - 1e-6) & (O.answer <= O.p75c + 1e-6)
print("answer between P50 and calibrated P75:", int(O.answer_pos_ok.sum()), "of", int(O.q50.notna().sum()))
O["vol6"] = M.loc[T, mo(2026, [1, 2, 3, 4, 5, 6])].pipe(lambda d: d.std(axis=1) / d.mean(axis=1))
print(O.groupby("conf").agg(n=("vol6", "size"), cv_last6_median=("vol6", "median"), iqr_ratio_median=("iqr_ratio", "median"), age_median=("age_m", "median"), level_median=("level", "median")).round(2).to_string())
print("confidence-1 items:"); print(O[O.conf == 1][["ptype", "age_m", "level", "p25c", "answer", "p75c", "vol6"]].join(A[["mj25", "jo25", "mj26", "fc"]]).round(2).to_string())
A.join(O[["conf", "p25c", "p75c", "q50", "answer"]]).to_csv("data/sanity_flags.csv")
