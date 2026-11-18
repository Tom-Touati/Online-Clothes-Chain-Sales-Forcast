"""Young items: is the model being BELOW the prior-year age-group ratio right? At every backtest origin with a known 4-month
outcome and a prior year (2025-03 .. 2026-02): for items <3 months and 3-12 months old at the origin, compare
(a) model P50 total, (b) prior-year ratio benchmark = origin-month units x (next-4-months / origin month) of the same age group
one year earlier (items with >= 50 units in that month), against the actual 4-month total."""
import pandas as pd, numpy as np, sys
sys.stdout.reconfigure(encoding="utf-8")
M = pd.read_parquet("data/monthly_item.parquet"); cols = list(M.columns); it = pd.read_parquet("data/items.parquet").set_index("style_color")
start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
Q = pd.read_parquet("data/q_backtest.parquet")
grp = lambda a: np.where(a < 3, "<3m", np.where(a < 12, "3-12m", "12m+"))
rows = []
for o, g in Q.groupby("origin"):
    k = cols.index(o)
    if k < 12 or k + 4 >= len(cols): continue
    po = pd.Period(o); ly = cols[k - 12]; nxt_ly = cols[k - 11:k - 7]
    a_ly = (po.end_time - pd.DateOffset(years=1) - start.reindex(M.index)).dt.days / 30.44
    base = M.index[(M[ly] >= 50).values & (a_ly >= 1).values]; gl = pd.Series(grp(a_ly[base].values), index=base)
    rat = {x: M.loc[gl.index[gl == x], nxt_ly].sum().sum() / M.loc[gl.index[gl == x], ly].sum() for x in ["<3m", "3-12m", "12m+"]}
    g = g.set_index("style_color"); age = (po.end_time - start.reindex(g.index)).dt.days / 30.44
    ok = (M.loc[g.index, o] >= 50).values & (age >= 1).values
    gg = pd.Series(grp(age.values), index=g.index)
    for x in ["<3m", "3-12m"]:
        ids = g.index[ok & (gg == x).values]
        if len(ids) == 0: continue
        a = g.loc[ids, "ytot"]; m = g.loc[ids, "q50"]; b = M.loc[ids, o] * rat[x]
        rows.append(dict(origin=o, group=x, n=len(ids), actual=a.sum(), model_bias=m.sum() / a.sum() - 1, bench_bias=b.sum() / a.sum() - 1,
                         model_wape=(m - a).abs().sum() / a.sum(), bench_wape=(b - a).abs().sum() / a.sum(),
                         model_vs_bench=m.sum() / b.sum() - 1, actual_vs_bench=a.sum() / b.sum() - 1))
R = pd.DataFrame(rows)
pd.set_option("display.width", 220)
for x in ["<3m", "3-12m"]:
    d = R[R.group == x]
    print(f"\n== {x}"); print(d.drop(columns="group").round(3).to_string(index=False))
    below = d.model_vs_bench < 0
    print(f"model below benchmark at {below.sum()} of {len(d)} origins; at those, actual also below benchmark at {(d.actual_vs_bench[below] < 0).sum()}"
          f" | mean bias model {d.model_bias.mean():+.3f}, benchmark {d.bench_bias.mean():+.3f} | WAPE better for model at {(d.model_wape < d.bench_wape).sum()} of {len(d)}"
          f" | mean WAPE model {d.model_wape.mean():.3f}, benchmark {d.bench_wape.mean():.3f}")
R.to_csv("data/young_ratio_test.csv", index=False)
