"""D322 (fork; proposal #6, assigned in D314): backtest of the month-1 update (eda/month1_engine.py). At every recent origin, months 2-4 of the
window are forecast four ways and scored on what sold in months 2-4:
  keep        the original forecast's months 2-4
  scaled      months 2-4 x (month-1 actual / month-1 forecast) ^ a, with a chosen on earlier closed origins (grid 0 to 1)
  model       the LightGBM update, trained on earlier closed origins (full strength and half strength)
  R3 rest     R3 x 3/4 (the reference for WAPE / R3)
Scores: WAPE / R3 (months 2-4), absolute error, bias, within +/-20%, GMRAE; recent origins and the June 2025 origin. Writes data/fk_month1_preds.parquet;
adds J["month1"]; stdout -> eda/month1_update_output.txt."""
import sys, json, runpy, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
E = runpy.run_path("eda/month1_engine.py", run_name="engine"); B = E["backtest_table"](); ORIG = sorted(B.origin.unique()); REC = [o for o in ORIG if o >= "2025-06"]
print("check: month 1 + months 2-4 = 4-month actual:", bool(np.allclose(B.m1 + B.y_rest, B.y, atol=1)), "| month-1 share of the forecast: median", round(float((B.f_m1 / B.F.clip(lower=1)).median()), 3), flush=True)
B["keep"] = B.f_rest; B["scaled"] = np.nan; B["model"] = np.nan; B["model_half"] = np.nan; B["model_half_smear"] = np.nan; alphas = {}; smear = {}
for o in REC:
    oo = pd.Period(o).ordinal; tr = B[(B.oo + 4 <= oo)]; te = (B.origin == o).values
    if len(tr) < 500: continue
    best = min(np.linspace(0, 1, 11), key=lambda a: float(np.abs(tr.f_rest * ((tr.m1 + 1) / (tr.f_m1 + 1)) ** a - tr.y_rest).sum())); alphas[o] = round(float(best), 2)
    B.loc[te, "scaled"] = (B.f_rest[te] * ((B.m1[te] + 1) / (B.f_m1[te] + 1)) ** best).values
    m = E["train"](tr); B.loc[te, "model"] = E["update"](m, B[te]); B.loc[te, "model_half"] = E["update"](m, B[te], 0.5)
    ins = E["update"](m, tr, 0.5); sm = float(tr.y_rest.clip(lower=0).sum() / max(ins.sum(), 1)); smear[o] = round(sm, 3); B.loc[te, "model_half_smear"] = B.loc[te, "model_half"].values * sm   # median-to-total correction from the training rows
B["model_half_lev"] = np.nan; mr = B.origin.isin(REC); B.loc[mr, "model_half_lev"] = (B.model_half * B.groupby("origin").scaled.transform("sum") / B.groupby("origin").model_half.transform("sum"))[mr]   # the model's split, the scaled method's total
def sc(f, Z):
    y, r3 = Z.y_rest.clip(lower=0).values, Z.r3_rest.values; ef, er = np.abs(f - y), np.abs(r3 - y)
    return dict(wape_r3=r(ef.sum() / er.sum()), abs_err=r(ef.sum() / y.sum(), 3), bias=r(f.sum() / y.sum() - 1, 3), hit20=r(float((ef <= 0.2 * y).mean()), 3), gmrae=r(float(np.exp(np.mean(np.log((ef + 1) / (er + 1))))), 3))
rows = []
for lab, c in [("keep the original months 2-4", "keep"), ("scaled by the month-1 surprise", "scaled"), ("month-1 update model", "model"), ("month-1 update model, half strength", "model_half"), ("half strength, level-corrected on the training rows", "model_half_smear"), ("half strength, total from the scaled method (recommended)", "model_half_lev")]:
    for per, m in [("recent", B.origin.isin(REC)), ("June 2025", B.origin == "2025-06")]:
        Z = B[m]; rows.append(dict(method=lab, period=per, **sc(Z[c].values, Z)))
R = pd.DataFrame(rows); print(R.to_string(index=False)); print("scaling exponent by origin:", alphas, "| level correction by origin:", smear)
po = [dict(origin=o, keep=sc(B[B.origin == o].keep.values, B[B.origin == o])["wape_r3"], half=sc(B[B.origin == o].model_half.values, B[B.origin == o])["wape_r3"], half_smear=sc(B[B.origin == o].model_half_smear.values, B[B.origin == o])["wape_r3"]) for o in REC]; print(pd.DataFrame(po).to_string(index=False))
full = E["train"](B[B.origin.isin(ORIG)]); imp = pd.Series(full.booster_.feature_importance("gain"), index=E["FEATS"]); imp = (imp / imp.sum()).sort_values(ascending=False).round(3); print("gain shares:", imp.to_dict())
B[["origin", "style_color", "m1", "f_m1", "f_rest", "y_rest", "r3_rest", "keep", "scaled", "model", "model_half", "model_half_smear", "model_half_lev"]].to_parquet("data/fk_month1_preds.parquet")
J = json.load(open("data/fk_err_analysis.json")); J["month1"] = dict(rows=rows, alphas=alphas, smear=smear, per_origin=po, importance=imp.to_dict()); json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
