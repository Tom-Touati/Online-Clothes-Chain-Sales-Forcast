import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[~d.cancelled]; d["ny_date"] = pd.to_datetime(d.ny_date)
ids = ["S24840_LIGHT-GRAY-MELANGE", "S24847_NAVY", "S25039_YELLOW", "S23794_BLACK", "S23965_BLACK", "S23605_PINK"]
x = d[d.style_color.isin(ids) & (d.ny_date >= "2026-06-01")].groupby(["style_color", "ny_date"]).qty.sum().unstack(0).fillna(0)
print("daily units, last 14 days of June:", x.loc["2026-06-17":].mean().round(1).to_dict())
print("daily units, Jun 1-16:", x.loc[:"2026-06-16"].mean().round(1).to_dict())
print(x.resample("W-SUN").sum().to_string())
# (b) backtest: does agreement with the structured baseline sharpen the confidence-1 group?
I = pd.read_parquet("data/round3_item_totals.parquet"); F = pd.read_parquet("data/features.parquet"); F = F[F.h == 1][["origin", "style_color", "level"]]
E = I.merge(F, left_on=["origin", "sc"], right_on=["origin", "style_color"]); E = E[E.b0.notna()]
E["ape"] = (E.ens - E.a).abs() / E.a.clip(lower=1); E["gap"] = np.abs(np.log((E.ens + 1) / (E.b0 + 1)))
for nm, g in [("level>=3000", E[E.level >= 3000]), ("level>=3000 & |gap to B0|<0.3", E[(E.level >= 3000) & (E.gap < .3)]), ("level>=3000 & |gap|>=0.3", E[(E.level >= 3000) & (E.gap >= .3)]),
              ("all & |gap|<0.15", E[E.gap < .15]), ("all & |gap|>=0.5", E[E.gap >= .5])]:
    print(f"{nm:34s} n={len(g):5d} within20 {(g.ape <= .2).mean():.3f} within50 {(g.ape <= .5).mean():.3f}")
# (c) the June-origin level: model vs realised multiplier, items >=12 months old, origin 2025-06
B = pd.read_parquet("data/best_backtest_preds.parquet").groupby(["origin", "style_color"])[["pred", "y"]].sum().reset_index()
G = pd.read_parquet("data/features.parquet"); G = G[G.h == 1][["origin", "style_color", "level", "age_m"]]
B = B.merge(G, on=["origin", "style_color"])
for o in ["2025-05", "2025-06", "2025-07"]:
    b = B[(B.origin == o) & (B.age_m >= 12)]; a = B[B.origin == o]
    print(o, "items>=12m: model mult", round(b.pred.sum() / (4 * b.level.sum()), 3), "realised", round(b.y.sum() / (4 * b.level.sum()), 3), "| all items: model", round(a.pred.sum() / (4 * a.level.sum()), 3), "realised", round(a.y.sum() / (4 * a.level.sum()), 3))
