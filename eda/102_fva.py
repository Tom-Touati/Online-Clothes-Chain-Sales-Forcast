"""Forecast value added, stage by stage from R3 (D118), on the backtest (same items as eda/86). Each stage in the standard table.
  R3                         last 3 months x 4
  R3 x type seasonality      sum over the 4 months of (mean of the last 3 months) x exp(like-for-like type seasonal index, eda/06 lfl_type)
  R3 x own seasonality       the item's own last-year ratio (eda/06 own_sea) where it has a year of history, else the type index
  quantile model             production quantile median (learns ageing, momentum, attributes, occasions on top)
  quantile + NN              geometric 50/50
  rules + quantile + NN      the three-way combination
Production hand rules (D051-D063) cannot be replayed per item on the backtest (only aggregates were kept), so the waterfall stops there."""
import numpy as np, pandas as pd, sys, importlib.util
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
T = pd.read_parquet("data/fixes_items.parquet")
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
q = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))
LQ = np.log1p(q[["q10", "q25", "q50", "q75", "q90"]].values); D = {c: LQ[:, j] - LQ[:, 2] for j, c in enumerate(["q10", "q25", "q50", "q75", "q90"])}
F = pd.read_parquet("data/features.parquet", columns=["origin", "style_color", "h", "m3", "lfl_type", "own_sea"])
F["sea_type"] = F.m3 * np.exp(F.lfl_type.fillna(0)); F["sea_own"] = F.m3 * np.exp(F.own_sea.fillna(F.lfl_type).fillna(0))
G = F.groupby(["origin", "style_color"])[["sea_type", "sea_own"]].sum().reset_index()
T = T.merge(G, on=["origin", "style_color"], how="left"); assert T.sea_type.notna().all()
COLS = ["R3", "sea_type", "sea_own", "qmodel", "q_nn", "r_q_nn"]
NM = {"R3": "1 R3", "sea_type": "2 R3 x type seasonality", "sea_own": "3 R3 x own seasonality", "qmodel": "4 quantile model", "q_nn": "5 + NN", "r_q_nn": "6 + rules (three-way)"}
REC = [o for o in sorted(T.origin.unique()) if o >= "2025-06"]
for nm, X in [("nine recent origins", T[T.origin.isin(REC)]), ("all 13 origins", T), ("recent, 100 targets", T[T.origin.isin(REC) & T.target]),
              ("recent, items 12 months+", T[T.origin.isin(REC) & (T.age == "12 months+")]), ("recent, items under 12 months", T[T.origin.isin(REC) & (T.age != "12 months+")])]:
    print(f"\n######## {nm}"); print(sl.table(X, COLS, D, NM).round(4).to_string())
