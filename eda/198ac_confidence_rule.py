"""D318 (fork; proposal #12b of the error study, assigned in D314: "State the measured hit rate with each confidence level ... and define level
1 by the validated rule: a narrow range, a rising trend and a year of history"). The rule and its rates, for Main to put in the next candidate.
  level 1   narrowest fifth of the P10-P90 range at the origin (log width, fifths within the origin over all items) AND rising 91-day trend
            AND sold in the same months last year            (the rule that held in both halves of the backtest, D278)
  level 3   widest fifth of the range
  level 2   the rest
Also a broader level 1 (two narrowest fifths). Rates: share of item totals within +/-20% on recent origins, in each half of the 13 origins, and
the unit-weighted rate. 2026: the same rule at the June 2026 origin, with the range widths of the conformal TabPFN member for all 1,423 items
(identical widths to the workbook's P10-P90) and the 2026 trend and last-year flags; compared with the workbook's current confidence column.
Writes data/fk_confidence_2026.csv; adds J["confrule"]; stdout -> eda/confidence_rule_output.txt."""
import sys, json, numpy as np, pandas as pd, openpyxl
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
r = lambda x, k=3: None if x is None or not np.isfinite(x) else round(float(x), k)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); ORIG = sorted(D.origin.unique()); H1, H2 = ORIG[:7], ORIG[7:]
D["iw"] = np.log((D.p90 + 1) / (D.p10 + 1)); D["fifth"] = D.groupby("origin")["iw"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)) + 1
D["rising"] = D.trend91.fillna(-1) > 0; D["ly"] = D.has_ly.fillna(0) == 1; D["hit"] = (D.F - D.y).abs() <= 0.2 * D.y
def levels(Z, broad=False): return np.where(((Z.fifth <= (2 if broad else 1)) & Z.rising & Z.ly).values, 1, np.where((Z.fifth == 5).values, 3, 2))
rows = []
for tag, broad in [("level 1 = narrowest fifth + rising + last year", False), ("level 1 = two narrowest fifths + rising + last year", True)]:
    D["lvl"] = levels(D, broad)
    for k in (1, 2, 3):
        m = D.lvl == k; rec = m & (D.origin >= "2025-06")
        rows.append(dict(rule=tag, level=k, share=r(float(rec.sum() / (D.origin >= "2025-06").sum())), hit_recent=r(float(D.hit[rec].mean())), hit_h1=r(float(D.hit[m & D.origin.isin(H1)].mean())), hit_h2=r(float(D.hit[m & D.origin.isin(H2)].mean())),
                         hit_units=r(float(D.y[rec & D.hit].sum() / D.y[rec].sum())), units=r(float(D.y[rec].sum() / D.y[D.origin >= "2025-06"].sum())), abs_err=r(float((D.F - D.y).abs()[rec].sum() / D.y[rec].sum()))))
R = pd.DataFrame(rows); print(R.to_string(index=False))
# 2026
Q = pd.read_parquet("data/fk_2026_pf2c.parquet").set_index("style_color"); T = pd.read_parquet("data/fk_err_master_2026.parquet").set_index("style_color")
Z = pd.DataFrame(index=Q.index); Z["iw"] = np.log((Q.q90 + 1) / (Q.q10 + 1)); Z["fifth"] = pd.qcut(Z.iw.rank(method="first"), 5, labels=False).astype(int) + 1
Z["rising"] = T.trend91.reindex(Z.index).fillna(-1) > 0; Z["ly"] = T.has_ly.reindex(Z.index).fillna(0) == 1; Z["level"] = levels(Z); Z["level_broad"] = levels(Z, True)
ws = openpyxl.load_workbook("Edikted_Forecasting_Exercise_filled.xlsx", data_only=True)["Forecast"]; W = {x[1]: (x[2], x[11]) for x in ws.iter_rows(min_row=2, values_only=True) if x[1]}
O = pd.DataFrame([dict(style_color=k, name=v[0], current=v[1]) for k, v in W.items()]).set_index("style_color"); O = O.join(Z[["iw", "fifth", "rising", "ly", "level", "level_broad"]])
HR = ["S24840_LIGHT-GRAY-MELANGE", "S25039_YELLOW", "S24847_NAVY", "S16079_DARK-GRAY"]; O.loc[O.index.isin(HR), ["level", "level_broad"]] = 3   # hand-rule rows stay at level 3
O["level"] = O.level.fillna(3).astype(int); O["level_broad"] = O.level_broad.fillna(3).astype(int)
rate = {k: R[(R.rule.str.startswith("level 1 = narrowest")) & (R.level == k)].hit_recent.iloc[0] for k in (1, 2, 3)}; O["expected_hit"] = O.level.map(rate)
O.to_csv("data/fk_confidence_2026.csv"); ct = pd.crosstab(O.current, O.level, margins=True); print(ct.to_string()); print("2026 targets by rule level:", O.level.value_counts().sort_index().to_dict(), "| broad:", O.level_broad.value_counts().sort_index().to_dict())
print("level-1 targets:", O[O.level == 1][["name", "current"]].to_string())
J = json.load(open("data/fk_err_analysis.json")); J["confrule"] = dict(rows=rows, targets=dict(rule=O.level.value_counts().sort_index().to_dict(), broad=O.level_broad.value_counts().sort_index().to_dict(), current=O.current.value_counts().sort_index().to_dict()),
                                                                  crosstab={f"{a}-{b}": int(((O.current == a) & (O.level == b)).sum()) for a in (1, 2, 3) for b in (1, 2, 3)}, level1=[dict(item=i, name=v["name"], current=int(v["current"])) for i, v in O[O.level == 1].iterrows()])
J["confrule"]["targets"] = {k: {str(a): int(b) for a, b in v.items()} for k, v in J["confrule"]["targets"].items()}; json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
