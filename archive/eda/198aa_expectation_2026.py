"""D316, re-frozen as D319 on candidate_nx4 (Main D350: nx3's units, the D318 confidence levels, the D317 ranges). (fork; proposal #14a of the error study, assigned in D314: "Freeze the submitted system ... write down the expectation ... score it
when actuals arrive"). Freezes what is needed to judge the July-October 2026 forecast later, and states what the backtest expects.
  frozen   the submitted workbook's md5 and its 100 rows (months, total, confidence); from the June 2026 cascade (data/final_2026_nx3_cascade.csv):
           the model's own forecast, R3, the P10 / P90 range, the buying quantity; the seasonal-naive baseline at June 2026 (last 3 months' rate
           x last year's like-for-like type ratio per target month, as eda/198e2); which rows carry a hand rule
  expected the same season in the backtest: the June 2025 origin (window July-October 2025) as the central case, the May-July 2025 origins as the
           range; and, because only 100 items will be scored, the spread of each metric over 2,000 random 100-item draws from the June 2025
           origin among items as large as the targets
Writes data/fk_expectation_2026.csv, data/fk_expectation_2026.json and FORECAST_EXPECTATION.md; adds J["expect26"]."""
import sys, json, hashlib, numpy as np, pandas as pd, openpyxl
sys.stdout.reconfigure(encoding="utf-8")
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
WB = "Edikted_Forecasting_Exercise_filled.xlsx"; md5 = hashlib.md5(open(WB, "rb").read()).hexdigest()
ws = openpyxl.load_workbook(WB, data_only=True)["Forecast"]; hdr = [c.value for c in ws[1]]; rows = [r_ for r_ in ws.iter_rows(min_row=2, values_only=True) if r_[1]]
W = pd.DataFrame([dict(style_color=x[1], name=x[2], ptype=x[3], jul=x[6], aug=x[7], sep=x[8], oct=x[9], total=x[10], confidence=x[11]) for x in rows]).set_index("style_color"); assert len(W) == 100
C = pd.read_csv("data/final_2026_nx4_cascade.csv", index_col=0).reindex(W.index)
Fe = pd.read_parquet("data/features.parquet", columns=["origin", "style_color", "h", "m3", "lfl_type"]); Fe = Fe[Fe.origin == "2026-06"]; Fe["st"] = Fe.m3 * np.exp(Fe.lfl_type.fillna(0)); sea = Fe.groupby("style_color").st.sum()
E = W.copy(); E["model_total"] = C.forecast.round(0); E["R3"] = C.R3.round(1); E["seasonal_naive"] = sea.reindex(W.index).fillna(C.R3).round(1); E["P10"] = C.P10.round(0); E["P90"] = C.P90.round(0); E["buying"] = C.buying.round(0)
WC = ["S24840_LIGHT-GRAY-MELANGE", "S25039_YELLOW", "S24847_NAVY"]; E["hand_rule"] = ""; E.loc[WC, "hand_rule"] = "World Cup cap (D033)"; E.loc["S16079_DARK-GRAY", "hand_rule"] = "out of stock: in-stock forecast kept (D034)"
assert int(E.total.sum()) == 601801, E.total.sum(); E.to_csv("data/fk_expectation_2026.csv")
# ---------- expectation from the same season in the backtest
J = json.load(open("data/fk_err_analysis.json")); D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True)
Fs = pd.read_parquet("data/features.parquet", columns=["origin", "style_color", "h", "m3", "lfl_type"]); Fs["st"] = Fs.m3 * np.exp(Fs.lfl_type.fillna(0)); D = D.merge(Fs.groupby(["origin", "style_color"]).st.sum().rename("sea").reset_index(), on=["origin", "style_color"], how="left"); D["sea"] = D.sea.fillna(D.R3)
def met(Z, F="F"):
    f, y, r3 = Z[F].values, Z.y.values, Z.R3.values; ef, er = np.abs(f - y), np.abs(r3 - y)
    return dict(abs_wape=ef.sum() / y.sum(), wape_r3=ef.sum() / er.sum(), vs_seasonal=ef.sum() / np.abs(Z.sea.values - y).sum(), bias=f.sum() / y.sum() - 1, hit20=float((ef <= 0.2 * y).mean()), hit20_units=float(y[ef <= 0.2 * y].sum() / y.sum()),
                gmrae=float(np.exp(np.mean(np.log((ef + 1) / (er + 1))))), cover=float(((y >= Z.p10.values) & (y <= Z.p90.values)).mean()), above=float((y > Z.p90.values).mean()))
# the submitted ranges are widened on the upper side (D317: lower x1.0, upper log width x1.4), so the expected coverage uses the same rule in the backtest
lF = np.log(D.F + 1); D["p10"] = np.exp(lF - np.maximum(lF - np.log(D.p10 + 1), 0.05)) - 1; D["p90"] = np.exp(lF + 1.4 * np.maximum(np.log(D.p90 + 1) - lF, 0.05)) - 1
O = {o: met(D[D.origin == o]) for o in ["2025-05", "2025-06", "2025-07"]}
lo_r3 = float(np.percentile(E.R3, 10)); pool = D[(D.origin == "2025-06") & (D.R3 >= lo_r3)].reset_index(drop=True); rng = np.random.default_rng(0)
draws = pd.DataFrame([met(pool.iloc[rng.choice(len(pool), 100, replace=False)]) for _ in range(2000)])
band = {k: [r(draws[k].quantile(0.1), 3), r(draws[k].quantile(0.9), 3)] for k in draws.columns}
EXP = dict(md5=md5, workbook=WB, total=int(E.total.sum()), months={m: int(E[m].sum()) for m in ["jul", "aug", "sep", "oct"]}, model_total=int(E.model_total.sum()), r3_total=r(E.R3.sum(), 0), sea_total=r(E.seasonal_naive.sum(), 0),
           central={k: r(v, 3) for k, v in O["2025-06"].items()}, range={k: [r(min(O[o][k] for o in O), 3), r(max(O[o][k] for o in O), 3)] for k in O["2025-06"]}, band100=band, pool_n=int(len(pool)), pool_min_r3=r(lo_r3, 0),
           backtest_cover_overall=J["coverage"]["overall"], widen_13=J["analogue"]["widen"][2]["cover"])
json.dump(EXP, open("data/fk_expectation_2026.json", "w"), indent=1); J["expect26"] = EXP; json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0)
pc = lambda x: f"{100 * x:.0f}%"; sp = lambda x: f"{100 * x:+.0f}%"; f2 = lambda x: f"{x:.2f}"
lines = [("Absolute error, Σ|F − y| ÷ Σy", "abs_wape", pc), ("WAPE ÷ R3", "wape_r3", f2), ("Error ÷ seasonal naive's error", "vs_seasonal", f2), ("Bias, ΣF ÷ Σy − 1", "bias", sp), ("Items within ±20%", "hit20", pc),
         ("Units in items within ±20%", "hit20_units", pc), ("Typical item ÷ R3 (GMRAE)", "gmrae", f2), ("Actuals inside P10–P90 (widened upper side)", "cover", pc), ("Actuals above P90", "above", pc)]
tab = "\n".join(f"| {lab} | {fmt(EXP['central'][k])} | {fmt(EXP['range'][k][0])} to {fmt(EXP['range'][k][1])} | {fmt(EXP['band100'][k][0])} to {fmt(EXP['band100'][k][1])} |" for lab, k, fmt in lines)
md = f"""# Forecast expectation, July–October 2026

Written on 17 Nov 2026, before any July–October 2026 sales are known (proposal 14 of the error study). First frozen on candidate_nx3 (D316, md5 `ac444112722d4b161a6c22191cfc2a49`). Re-frozen on candidate_nx4 (D319) after Main's D350: the units are identical; the confidence levels follow the validated rule (D318) and the P10–P90 ranges are widened on the upper side (D317). Nothing below may be tuned after the actuals arrive.

## What is frozen

- **Workbook:** `{WB}`, md5 `{md5}` (candidate_nx4, D350). 100 targets, **{EXP['total']:,} units**: July {EXP['months']['jul']:,}, August {EXP['months']['aug']:,}, September {EXP['months']['sep']:,}, October {EXP['months']['oct']:,}.
- **Per target** in `data/fk_expectation_2026.csv`: the four months and total, confidence level, the model's own total before the hand rules ({EXP['model_total']:,}), R3 ({EXP['r3_total']:,.0f}), the seasonal-naive baseline ({EXP['sea_total']:,.0f}), the P10–P90 range and the buying quantity.
- **Confidence levels** (D318): level 1 = narrowest fifth of the range, rising 91-day trend and sales in the same months last year; level 3 = widest fifth; level 2 = the rest. Targets 10 / 82 / 8. Backtest share within ±20%: 58% / 44% / 28%.
- **Ranges** (D317): the lower end as stated, the upper log width × 1.4; the expected coverage below uses the same rule in the backtest.
- **Hand rules:** the three World Cup items carry the cap of the first submission (D033, 34,463 units against the models' 53,270); `S16079_DARK-GRAY`, out of stock, keeps its in-stock forecast (D034).

## What the backtest expects

The same season in the backtest is the June 2025 origin (forecasting July–October 2025). The May and July 2025 origins give the range. Only 100 items will be scored, so the last column shows how much each number moves by chance alone. It is the 10th–90th percentile over 2,000 random draws of 100 items from the June 2025 origin, among items at least as large as the targets (R3 ≥ {EXP['pool_min_r3']:,.0f}; {EXP['pool_n']} items).

| Metric | June 2025 origin | May–July 2025 origins | Any 100 items (80% band) |
|---|---|---|---|
{tab}

## How to read the result

- **Inside the 80% band:** the forecast did what the backtest said it would.
- **Worse than the band on absolute error or bias:** something changed that the backtest did not contain, such as a new channel, a different roll-out pace, or a promotion calendar unlike 2025's. Look at the targets that drove it before blaming the models.
- **Coverage below the band:** the ranges are too narrow for this season even after the upper side was widened.
- **Bias:** a negative bias of 5–10% is expected; the backtest under-forecasts growth in this season.
- **Ratio to R3:** depends on how much the season moves. Absolute error is the steadier number.

## How to score it

```
.venv/Scripts/python.exe eda/score_2026_actuals.py <actuals.csv>
```

The actuals file needs one row per target and month (`style_color, month, units`, month as `2026-07` … `2026-10`) or one row per target (`style_color, jul, aug, sep, oct`). Use the same unit definition as the training data: order lines excluding cancellations. The script checks the workbook's md5 against the frozen one and prints every metric above with its expected range.
"""
open("reports/FORECAST_EXPECTATION.md", "w", encoding="utf-8").write(md); print(md[:400]); print("bands:", band)
