"""D322 (fork; proposal #6): the month-1 update for 2026. After July 2026 has sold, re-forecasts August-October for the 100 targets with the model of
eda/month1_engine.py, trained on all 13 backtest origins (all windows closed), at half strength, with the total of the targets set by the simple scaled update (months 2-4 x (July actual / July forecast)^0.2). Backtest, months
2-4: WAPE / R3 0.3687 against 0.3952 without the update, bias +2.3%; June 2025 origin 0.644 against 0.738. The submitted workbook gives July and August-October; the updated three-month total is split over August-October in the workbook's
proportions.
Usage: python eda/month1_update_2026.py <july_actuals.csv> [out.csv]
  july_actuals.csv: style_color, units  [, own_stores, partners, web_discount]  (July 2026; order lines without cancellations; the optional
  columns are the item's own stores and partners selling it in July and its average web discount share in July; missing = unchanged from June)"""
import sys, runpy, numpy as np, pandas as pd, openpyxl
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 200)
E = runpy.run_path("eda/month1_engine.py", run_name="engine"); B = E["backtest_table"](); model = E["train"](B)
ws = openpyxl.load_workbook("Edikted_Forecasting_Exercise_filled.xlsx", data_only=True)["Forecast"]
W = pd.DataFrame([dict(style_color=x[1], name=x[2], jul=float(x[6]), aug=float(x[7]), sep=float(x[8]), oct=float(x[9])) for x in ws.iter_rows(min_row=2, values_only=True) if x[1]]).set_index("style_color")
A = pd.read_csv(sys.argv[1]); A.columns = [c.lower().strip() for c in A.columns]; A = A.set_index("style_color").reindex(W.index)
Z = pd.read_parquet("data/fk_err_master_2026.parquet").set_index("style_color").reindex(W.index); ok = Z.F.notna() & A.units.notna()
Zr = Z[ok].reset_index(); m1 = A.units[ok].values; f_m1 = W.jul[ok].values; X = E["features"](Zr, "2026-06", m1, f_m1)
with np.errstate(all="ignore"):
    now_st = E["ax"]("n_stores", Zr.style_color.values, "2026-06"); now_pa = E["ax"]("n_partners", Zr.style_color.values, "2026-06"); now_dw = E["ax"]("disc_web", Zr.style_color.values, "2026-06")
X["st1"] = (A.own_stores[ok].values - np.nan_to_num(now_st)) if "own_stores" in A else 0.0; X["pa1"] = (A.partners[ok].values - np.nan_to_num(now_pa)) if "partners" in A else 0.0; X["disc1"] = (A.web_discount[ok].values - now_dw) if "web_discount" in A else 0.0
X = X.fillna({"st1": 0.0, "pa1": 0.0, "disc1": 0.0}); X["f_rest"] = (W.aug + W.sep + W.oct)[ok].values
rest = E["update"](model, X, 0.5); lev = float((X.f_rest * ((m1 + 1) / (f_m1 + 1)) ** 0.2).sum()); rest = rest * lev / rest.sum()   # the model splits, the scaled update (exponent 0.2, the backtest's) sets the total
O = W[ok].copy(); O["july_actual"] = m1; tot = (O.aug + O.sep + O.oct).values
for c in ["aug", "sep", "oct"]: O[c + "_updated"] = np.round(rest * O[c].values / np.maximum(tot, 1))
O["aug_oct_before"] = tot.round(); O["aug_oct_updated"] = O[["aug_updated", "sep_updated", "oct_updated"]].sum(axis=1); O["change"] = O.aug_oct_updated / O.aug_oct_before - 1
print(f"targets updated: {int(ok.sum())} of {len(W)} | August-October {O.aug_oct_before.sum():,.0f} -> {O.aug_oct_updated.sum():,.0f} ({O.aug_oct_updated.sum() / O.aug_oct_before.sum() - 1:+.1%}) | July actual {m1.sum():,.0f} against {f_m1.sum():,.0f} forecast")
print(O.sort_values("change")[["name", "jul", "july_actual", "aug_oct_before", "aug_oct_updated", "change"]].round(3).head(6).to_string()); print(O.sort_values("change")[["name", "jul", "july_actual", "aug_oct_before", "aug_oct_updated", "change"]].round(3).tail(6).to_string())
if len(sys.argv) > 2: O.to_csv(sys.argv[2]); print("written", sys.argv[2])
