"""By age group at the June origin: history (2024, 2025) Jul-Oct/June and Jul share, forecast 2026, and June 2025 backtest bias of the P50 by group."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
src = open("eda/34_final_consistency.py", encoding="utf-8").read(); src = src[:src.index('print("== A')]; ns = {}; exec(src, ns)
O, M, group, T, WC, mo = ns["O"], ns["M"], ns["group"], ns["T"], ns["WC"], ns["mo"]; start = ns["start"]
rows = []
for y in (2024, 2025):
    ids = M.index[(M[f"{y}-06"] >= 50) & (start.reindex(M.index) < pd.Timestamp(f"{y}-06-15")).values]; g = group(ids, y)
    for k, s in g.groupby(g.values):
        x = M.loc[s.index]; t = x[mo(y, [7, 8, 9, 10])].sum()
        rows.append(dict(year=str(y), group=k, n=len(s), JulOct_Jun=t.sum() / x[f"{y}-06"].sum(), item_median=(x[mo(y, [7, 8, 9, 10])].sum(axis=1) / x[f"{y}-06"]).median(), jul_share=t.iloc[0] / t.sum()))
gT = group(pd.Index(T), 2026)
for k, s in gT.groupby(gT.values):
    f = O.loc[s.index]; rows.append(dict(year="2026 fc", group=k, n=len(s), JulOct_Jun=f.final_total.sum() / M.loc[s.index, "2026-06"].sum(), item_median=(f.final_total / M.loc[s.index, "2026-06"]).median(), jul_share=f.f_jul.sum() / f.final_total.sum()))
R = pd.DataFrame(rows); print(R.pivot(index="group", columns="year", values=["JulOct_Jun", "item_median", "jul_share"]).round(2).to_string())
Q = pd.read_parquet("data/q_backtest.parquet"); g = Q[Q.origin == "2025-06"].copy(); g["grp"] = group(pd.Index(g.style_color), 2025).values
print("\nJune 2025 origin backtest, P50 by group:")
print(g.groupby("grp").apply(lambda x: pd.Series(dict(n=len(x), bias=x.q50.sum() / x.ytot.sum() - 1, median_ratio=(x.q50 / x.ytot.clip(lower=1)).median()))).round(3).to_string())
B = pd.read_parquet("data/best_backtest_preds.parquet"); B = B[B.origin == "2025-06"]; B["grp"] = group(pd.Index(B.style_color), 2025).reindex(B.style_color).values
p = B.pivot_table(index="grp", columns="h", values=["pred", "y"], aggfunc="sum")
print("\nJune 2025 origin, monthly model: July share of Jul-Oct, predicted vs actual:"); print(pd.DataFrame({"pred_jul_share": p["pred"][1] / p["pred"].sum(axis=1), "act_jul_share": p["y"][1] / p["y"].sum(axis=1)}).round(3).to_string())
