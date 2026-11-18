"""D317 (fork; proposal #11 of the error study, assigned in D314: "Widen and shift the intervals for this season ... put most of the widening
on the upper side"). Calibrates the stated P10 / P90 separately on each side, time-honestly, and gives calibrated 2026 ranges for the targets.
  widths   wl = log((F + 1) / (P10 + 1)), wu = log((P90 + 1) / (F + 1)) around the submitted forecast F (floored at 0.05)
  scaling  P10' = (F + 1) exp(-kl wl) - 1,  P90' = (F + 1) exp(ku wu) - 1, with kl, ku chosen so that 10% of training actuals fall below
           P10' and 10% above P90' (the 90th percentile of the needed factor on each side)
  training (a) every origin whose window has closed; (b) only closed origins of the same season (forecast months July-October: origins May-July)
           when at least one exists, else (a)
Scored on each recent origin: share below / inside / above, mean log width, and the interval score (width + 2/0.2 x the miss outside), lower
is better. 2026: kl, ku from all 13 origins and from the May-July 2025 origins, applied to the June 2026 cascade (data/final_2026_nx3_cascade.csv).
Writes data/fk_intervals_2026.csv; adds J["intervals"]; stdout -> eda/intervals_output.txt."""
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); ORIG = sorted(D.origin.unique()); REC = [o for o in ORIG if o >= "2025-06"]; D["oo"] = D.origin.map(lambda o: pd.Period(o, "M").ordinal); D["mo"] = D.origin.str[-2:].astype(int)
lF = np.log(D.F + 1); D["wl"] = np.maximum(lF - np.log(D.p10 + 1), 0.05); D["wu"] = np.maximum(np.log(D.p90 + 1) - lF, 0.05); D["ly"] = np.log(D.y + 1) - lF
def ks(Z): return float(np.quantile(-Z.ly / Z.wl, 0.9)), float(np.quantile(Z.ly / Z.wu, 0.9))
def apply(Z, kl, ku): return np.exp(np.log(Z.F + 1) - kl * Z.wl) - 1, np.exp(np.log(Z.F + 1) + ku * Z.wu) - 1
def score(Z, lo, hi):
    y = Z.y.values; below, above = y < lo, y > hi; ly, lh, lyy = np.log(lo + 1), np.log(hi + 1), np.log(y + 1)
    isc = (lh - ly) + 10 * np.where(below, ly - lyy, 0) + 10 * np.where(above, lyy - lh, 0)
    return dict(below=float(below.mean()), inside=float((~below & ~above).mean()), above=float(above.mean()), width=float((lh - ly).mean()), interval_score=float(isc.mean()))
rows = []; kk = []
for o in REC:
    te = D[D.origin == o]; tr = D[D.oo + 4 <= pd.Period(o).ordinal]; trs = tr[tr.mo.isin([5, 6, 7])] if o[-2:] in ("05", "06", "07") and tr.mo.isin([5, 6, 7]).any() else tr
    rows.append(dict(origin=o, method="as stated", **score(te, te.p10.values, te.p90.values)))
    for lab, Z in [("calibrated on all earlier origins", tr), ("calibrated on the same season", trs)]:
        kl, ku = ks(Z); lo, hi = apply(te, kl, ku); rows.append(dict(origin=o, method=lab, kl=kl, ku=ku, **score(te, lo.values, hi.values)))
    rows.append(dict(origin=o, method="log width x 1.3 both sides", **score(te, *[v.values for v in apply(te, 1.3, 1.3)])))
R = pd.DataFrame(rows); agg = R.groupby("method")[["below", "inside", "above", "width", "interval_score"]].mean().round(3); print(agg.to_string()); print(R[R.origin == "2025-06"].round(3).to_string(index=False))
klA, kuA = ks(D); S = D[D.origin.isin(["2025-05", "2025-06", "2025-07"])]; klS, kuS = ks(S); print("factors: all 13", round(klA, 3), round(kuA, 3), "| May-July 2025", round(klS, 3), round(kuS, 3))
C = pd.read_csv("data/final_2026_nx3_cascade.csv", index_col=0); F = C.forecast; wl = np.maximum(np.log(F + 1) - np.log(C.P10 + 1), 0.05); wu = np.maximum(np.log(C.P90 + 1) - np.log(F + 1), 0.05)
O = pd.DataFrame({"forecast": F.round(0), "final": C.final, "P10": C.P10.round(0), "P90": C.P90.round(0)})
for tag, kl, ku in [("all13", klA, kuA), ("season", klS, kuS)]:
    O[f"P10_{tag}"] = (np.exp(np.log(F + 1) - kl * wl) - 1).round(0); O[f"P90_{tag}"] = (np.exp(np.log(F + 1) + ku * wu) - 1).round(0)
# leave-one-origin-out factors and fixed asymmetric rules (the time-honest calibration above fails: the early origins were over-forecast, so
# their factors point the wrong way); recommended rule: lower side unchanged, upper log width x 1.4 (D317)
L = []
for o in ORIG:
    te = D[D.origin == o]; kl, ku = ks(D[D.origin != o]); lo, hi = apply(te, kl, ku); L.append(dict(origin=o, method="leave-one-origin-out factors", kl=kl, ku=ku, **score(te, lo.values, hi.values)))
    for kl_, ku_ in [(1.0, 1.3), (1.0, 1.4), (1.0, 1.5)]:
        lo, hi = apply(te, kl_, ku_); L.append(dict(origin=o, method=f"lower x{kl_}, upper x{ku_}", **score(te, lo.values, hi.values)))
    L.append(dict(origin=o, method="as stated", **score(te, te.p10.values, te.p90.values))); lo, hi = apply(te, 1.3, 1.3); L.append(dict(origin=o, method="log width x 1.3 both sides", **score(te, lo.values, hi.values)))
L = pd.DataFrame(L); L.to_csv("data/fk_intervals_backtest.csv", index=False); aggL = L[L.origin.isin(REC)].groupby("method")[["below", "inside", "above", "width", "interval_score"]].mean().round(3); print(aggL.to_string())
KL, KU = 1.0, 1.4; O["P10_rec"] = (np.exp(np.log(F + 1) - KL * wl) - 1).round(0); O["P90_rec"] = (np.exp(np.log(F + 1) + KU * wu) - 1).round(0)
hr = O.final.sub(O.forecast).abs() > 1   # hand-rule rows (World Cup cap): the range is moved with the capped total
for c in ["P10", "P90", "P10_rec", "P90_rec"]: O.loc[hr, c] = (O.loc[hr, c] * O.loc[hr, "final"] / O.loc[hr, "forecast"]).round(0)
O.to_csv("data/fk_intervals_2026.csv"); print(O.head(4).to_string()); print("sums:", O[["forecast", "final", "P10", "P90", "P10_rec", "P90_rec"]].sum().round(0).to_dict()); print(O.loc[["S24840_LIGHT-GRAY-MELANGE", "S25039_YELLOW", "S24847_NAVY"]].to_string())
J = json.load(open("data/fk_err_analysis.json")); J["intervals"] = dict(summary={m: {k: r(v, 3) for k, v in agg.loc[m].items()} for m in agg.index}, june2025=[{k: (r(v, 3) if isinstance(v, float) else v) for k, v in d.items()} for d in R[R.origin == "2025-06"].to_dict("records")],
                                          factors=dict(all13=[r(klA, 3), r(kuA, 3)], season=[r(klS, 3), r(kuS, 3)], loo=[r(L[L.method.str.startswith("leave")].kl.min(), 3), r(L[L.method.str.startswith("leave")].kl.max(), 3), r(L[L.method.str.startswith("leave")].ku.min(), 3), r(L[L.method.str.startswith("leave")].ku.max(), 3)], recommended=[KL, KU]), loo_summary={m: {k: r(v, 3) for k, v in aggL.loc[m].items()} for m in aggL.index}, june2025_rec={k: r(v, 3) for k, v in L[(L.origin == "2025-06") & (L.method == "lower x1.0, upper x1.4")].iloc[0].items() if k not in ("origin", "method", "kl", "ku")}, hand_rule_rows=[str(x) for x in O.index[hr]], sums={k: r(v, 0) for k, v in O[["forecast", "final", "P10", "P90", "P10_rec", "P90_rec"]].sum().items()})
json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
