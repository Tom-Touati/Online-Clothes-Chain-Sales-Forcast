"""D295 (error-study proposal #8: event rules beyond the World Cup). For items tagged 4th of July, Halloween or Homecoming (data/items.parquet
occasion), the submitted system's backtest (E:ecnsnn_x_geo, data/geo_ecnsnn_x_preds.parquet) at origins whose 4-month window contains the
event month (4th of July: Mar-Jun origins; Halloween: Jun-Sep; Homecoming: May-Sep): bias, WAPE / R3 and the within-20% rate, against
untagged items at the same origins, by age (under / over 12 months), and the monthly shape check from the rule agents' and NN's monthly
profiles: share of the window's actual units in the event month vs the system's split (data/best_backtest_preds LightGBM profile).
Writes eda/event_rules_output.txt (stdout)."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 200)
B = pd.read_parquet("data/dec_base_ecnsnn.parquet").merge(pd.read_parquet("data/geo_ecnsnn_x_preds.parquet")[["origin", "style_color", "geo"]], on=["origin", "style_color"])
occ = pd.read_parquet("data/items.parquet").set_index("style_color").occasion.fillna("")
B["occ"] = occ.reindex(B.style_color).fillna("").values
EV = {"4th of July": ("4th of July", ["2025-03", "2025-04", "2025-05", "2025-06"], 7), "Halloween": ("Halloween", ["2025-06", "2025-07", "2025-08", "2025-09"], 10),
      "Homecoming": ("Homecoming", ["2025-05", "2025-06", "2025-07", "2025-08", "2025-09"], 9)}
BT = pd.read_parquet("data/best_backtest_preds.parquet")[["origin", "style_color", "h", "pred"]]
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
def stats(d): return pd.Series({"n": len(d), "units": d.y.sum(), "bias": d.geo.sum() / d.y.sum() - 1, "wape_r3": (d.geo - d.y).abs().sum() / (d.R3 - d.y).abs().sum(),
                                "within20": ((d.geo - d.y).abs() <= 0.2 * d.y).mean()})
for name, (tag, origins, mon) in EV.items():
    W = B[B.origin.isin(origins)].copy(); W["tagged"] = W.occ.str.contains(tag, regex=False); W["old"] = np.where(W.age_m >= 12, "12m+", "<12m")
    print(f"\n== {name}: origins {origins}, tagged item-origins {int(W.tagged.sum())}")
    print(W.groupby(["tagged", "old"]).apply(stats).round(3).to_string())
    T = W[W.tagged]
    if len(T):
        rows = []
        for (o, s), g in T.groupby(["origin", "style_color"]):
            P = pd.Period(o, "M"); tm = [str(P + h) for h in range(1, 5)]; k = [i for i, t in enumerate(tm) if int(t[-2:]) == mon]
            if not k: continue
            ya = M.reindex([s])[tm].fillna(0).values[0]; sh = BT[(BT.origin == o) & (BT.style_color == s)].sort_values("h").pred.values
            if len(sh) != 4 or ya.sum() <= 0: continue
            rows.append({"origin": o, "item": s, "actual_share_event_month": ya[k[0]] / ya.sum(), "forecast_share_event_month": sh[k[0]] / sh.sum(),
                         "actual_share_after": ya[k[0] + 1:].sum() / ya.sum(), "forecast_share_after": sh[k[0] + 1:].sum() / sh.sum()})
        R = pd.DataFrame(rows)
        if len(R): print("  monthly shape (mean shares): actual event month %.3f vs forecast %.3f | after the event: actual %.3f vs forecast %.3f (n %d)" % (
            R.actual_share_event_month.mean(), R.forecast_share_event_month.mean(), R.actual_share_after.mean(), R.forecast_share_after.mean(), len(R)))
it = pd.read_parquet("data/items.parquet").set_index("style_color"); import openpyxl
TG = [r[1].value for r in openpyxl.load_workbook("Edikted_Forecasting_Exercise.xlsx")["Forecast"].iter_rows(min_row=2) if r[1].value]
print("\n2026 targets by tag:", {t: int(occ.reindex(TG).fillna("").str.contains(t, regex=False).sum()) for t in ["4th of July", "Halloween", "Homecoming", "World Cup"]})
