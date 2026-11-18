"""Scores the frozen July-October 2026 forecast once actuals arrive (proposal 14 of the error study, D316). Reads the frozen per-target table
data/fk_expectation_2026.csv and the expectation data/fk_expectation_2026.json; prints each metric next to what the backtest expected.
Usage: python eda/score_2026_actuals.py <actuals.csv>
  long format:  style_color, month, units       (month as 2026-07 ... 2026-10)
  wide format:  style_color, jul, aug, sep, oct
Units must follow the training data's definition: order lines, cancellations excluded."""
import sys, json, hashlib, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
E = pd.read_csv("data/fk_expectation_2026.csv", index_col=0); X = json.load(open("data/fk_expectation_2026.json"))
md5 = hashlib.md5(open(X["workbook"], "rb").read()).hexdigest()
print("workbook md5", "matches the frozen one" if md5 == X["md5"] else f"CHANGED ({md5} against the frozen {X['md5']}): scoring the frozen table, not the current file")
A = pd.read_csv(sys.argv[1]); A.columns = [c.lower().strip() for c in A.columns]
if "month" in A.columns:
    A["month"] = A.month.astype(str).str[:7]; A = A.pivot_table(index="style_color", columns="month", values="units", aggfunc="sum").reindex(columns=["2026-07", "2026-08", "2026-09", "2026-10"]); A.columns = ["jul", "aug", "sep", "oct"]
else: A = A.set_index("style_color")[["jul", "aug", "sep", "oct"]]
A = A.reindex(E.index); missing = A.isna().all(axis=1); print(f"targets with actuals: {int((~missing).sum())} of {len(E)}" + (f"; missing: {', '.join(E.index[missing][:10])}" if missing.any() else ""))
Z = E[~missing].copy(); Z["y"] = A[~missing].fillna(0).sum(axis=1); f, y, r3, sea = Z.total.values.astype(float), Z.y.values, Z.R3.values, Z.seasonal_naive.values; ef, er = np.abs(f - y), np.abs(r3 - y)
M = dict(abs_wape=ef.sum() / y.sum(), wape_r3=ef.sum() / er.sum(), vs_seasonal=ef.sum() / np.abs(sea - y).sum(), bias=f.sum() / y.sum() - 1, hit20=float((ef <= 0.2 * y).mean()), hit20_units=float(y[ef <= 0.2 * y].sum() / y.sum()),
         gmrae=float(np.exp(np.mean(np.log((ef + 1) / (er + 1))))), cover=float(((y >= Z.P10.values) & (y <= Z.P90.values)).mean()), above=float((y > Z.P90.values).mean()))
LAB = dict(abs_wape="absolute error", wape_r3="WAPE ÷ R3", vs_seasonal="error ÷ seasonal naive's", bias="bias", hit20="items within ±20%", hit20_units="units within ±20%", gmrae="typical item ÷ R3 (GMRAE)", cover="inside P10–P90", above="above P90")
print(f"\n{'metric':28s} {'actual':>8s} {'expected':>9s} {'80% band (100 items)':>22s}  verdict")
for k, lab in LAB.items():
    lo, hi = X["band100"][k]; v = M[k]; verdict = "inside the band" if lo <= v <= hi else ("below the band" if v < lo else "above the band")
    print(f"{lab:28s} {v:8.3f} {X['central'][k]:9.3f} {lo:10.3f} to {hi:7.3f}  {verdict}")
mb = {m: float(E.loc[~missing, m].sum() / A.loc[~missing, m].sum() - 1) for m in ["jul", "aug", "sep", "oct"] if A.loc[~missing, m].sum() > 0}
print("\nbias by month:", {m: f"{100 * v:+.1f}%" for m, v in mb.items()})
Z["err"] = f - y; print("\nlargest misses (forecast − actual):"); print(Z.reindex(Z.err.abs().sort_values(ascending=False).index)[["name", "total", "y", "err", "P10", "P90", "hand_rule"]].head(10).to_string())
hr = Z[Z.hand_rule.fillna("") != ""]; print("\nhand-rule rows:"); print(hr[["name", "total", "model_total", "y", "hand_rule"]].to_string()) if len(hr) else None
