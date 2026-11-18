"""(a) Answer quantile by bias on season-analogous origins. (b) Realised Jul-Oct / June and monthly shape by segment in 2025 (and 2024),
against the final forecast for the targets in the same segment."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 230)
QS = np.array([0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]); QC = ["q10", "q25", "q40", "q50", "q60", "q75", "q90"]
Q = pd.read_parquet("data/q_backtest.parquet")
grid = np.round(np.arange(0.40, 0.701, 0.01), 2)
for nm, os_ in [("Jun 2025 only", ["2025-06"]), ("Jun-Jul 2025", ["2025-06", "2025-07"]), ("May-Jul 2025", ["2025-05", "2025-06", "2025-07"]), ("May-Aug 2025", ["2025-05", "2025-06", "2025-07", "2025-08"]),
                ("Sep-Oct 2025 (Black Friday windows)", ["2025-09", "2025-10"]), ("all 9 representative", [str(p) for p in pd.period_range("2025-06", "2026-02", freq="M")])]:
    g = Q[Q.origin.isin(os_)]; L = np.log1p(g[QC].values); a = g.ytot.sum()
    b = {t: np.expm1(np.array([np.interp(t, QS, r) for r in L])).sum() / a - 1 for t in grid}
    t0 = min(grid, key=lambda t: abs(b[t]))
    print(f"{nm:38s} bias at P50 {b[0.5]:+.3f} | at P61 {b[0.61]:+.3f} | zero-bias quantile {t0}")
M = pd.read_parquet("data/monthly_item.parquet"); it = pd.read_parquet("data/items.parquet").set_index("style_color")
start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
occ = it.occasion.fillna("")
SW = ["Swimwear Top", "Swimwear Bottom"]
def segs(idx):
    s = pd.Series("other", index=idx)
    s[occ.reindex(idx).str.contains("Homecoming").values] = "Homecoming"; s[occ.reindex(idx).str.contains("Halloween").values] = "Halloween"
    s[occ.reindex(idx).str.contains("4th of July").values] = "4th of July"; s[it.product_type.reindex(idx).isin(SW).values] = "swimwear"
    return s
rows = []
for y in (2024, 2025):
    live = start[start < pd.Timestamp(f"{y}-05-15")].index.intersection(M.index)
    live = live[M.loc[live, f"{y}-06"] >= 50]
    sg = segs(live); X = M.loc[live, [f"{y}-{m:02d}" for m in range(6, 11)]]; X.columns = ["Jun", "Jul", "Aug", "Sep", "Oct"]
    for s, g in X.groupby(sg.values):
        t = g.sum(); rows.append(dict(year=y, seg=s, n=len(g), JulOct_over_Jun=(t[["Jul", "Aug", "Sep", "Oct"]].sum() / t.Jun), item_median=((g[["Jul", "Aug", "Sep", "Oct"]].sum(axis=1)) / g.Jun).median(),
                                      Aug_Jul=t.Aug / t.Jul, Sep_Jul=t.Sep / t.Jul, Oct_Jul=t.Oct / t.Jul))
H = pd.DataFrame(rows)
import sys
O = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/final_forecast_q.csv", index_col=0); sgT = segs(O.index)
F = O[["f_jul", "f_aug", "f_sep", "f_oct"]]; F.columns = ["Jul", "Aug", "Sep", "Oct"]; jun = M.loc[O.index, "2026-06"]
for s, g in F.groupby(sgT.values):
    t = g.sum(); rows.append(dict(year="2026 forecast", seg=s, n=len(g), JulOct_over_Jun=t.sum() / jun[g.index].sum(), item_median=(g.sum(axis=1) / jun[g.index].replace(0, np.nan)).median(),
                                  Aug_Jul=t.Aug / t.Jul, Sep_Jul=t.Sep / t.Jul, Oct_Jul=t.Oct / t.Jul))
print("\nJul-Oct total / June, and monthly shape (month / July), items selling >= 50 in June, live by mid-May")
print(pd.DataFrame(rows).set_index(["seg", "year"]).sort_index().round(2).to_string())
print("\ntargets by segment:", sgT.value_counts().to_dict())
for s in ["swimwear", "Homecoming", "Halloween", "4th of July"]:
    ids = sgT[sgT == s].index; print(f"\n{s}:"); print(pd.DataFrame({"type": O.loc[ids, "ptype"], "age": O.loc[ids, "age_m"].round(1), "jun": jun[ids], "fc": O.loc[ids, "final_total"], "fc/jun": (O.loc[ids, "final_total"] / jun[ids]).round(2),
          "Jul": F.loc[ids, "Jul"], "Aug": F.loc[ids, "Aug"], "Sep": F.loc[ids, "Sep"], "Oct": F.loc[ids, "Oct"], "conf": O.loc[ids, "conf"]}).to_string())
