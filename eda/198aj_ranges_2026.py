"""D329 (fork; Main's request for candidate_nx5, following D328): per-segment (Mondrian) calibrated range ratios for every item at June 2026.
  widths   the conformal TabPFN member's quantiles at 2026-06 (data/fk_member_quantiles.parquet, member pf2c): w = log(q50 + 1) - log(q10 + 1) etc.,
           floored at 0.02
  factors  per segment, fitted on all 13 backtest origins (as D328, which fitted them leaving one origin out): kl, ku such that 10% (P10 / P90) or
           25% (P25 / P75) of actuals fall outside on each side of the submitted forecast
  segment  age band at the origin (months since the item's first sale: under 3, 3-6, 6-12, over 12) x volume third (the recency level, i.e. the
           last three months' average monthly units, ranked within the origin over all items: low / mid / high)
  output   r10 = exp(-kl10 w10), r25 = exp(-kl25 w25), r75 = exp(ku25 w75), r90 = exp(ku10 w90); apply as P = (P50 + 1) x r - 1 (for P50 above about
           20 units this equals P50 x r within 5%)
Writes data/fk_ranges_2026.csv (with these definitions as '#' header lines; read with pandas comment='#'); stdout -> eda/ranges_2026_output.txt."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 200)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); T = pd.read_parquet("data/fk_err_master_2026.parquet").reset_index(drop=True)
Q = pd.read_parquet("data/fk_member_quantiles.parquet"); Q = Q[Q.member == "pf2c"].set_index(["origin", "style_color"])
AGE = lambda a: pd.cut(a, [-1, 3, 6, 12, 999], labels=["under 3 months", "3-6 months", "6-12 months", "over 12 months"]).astype(str)
VOL = lambda Z: Z.groupby("origin")["level"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"])).astype(str)
def widths(Z):
    q = Q.reindex(pd.MultiIndex.from_arrays([Z.origin, Z.style_color])); l50 = np.log(q.q50.values + 1); f = lambda c: np.log(q[c].values + 1)
    return {c: np.maximum(np.abs(l50 - f(c)), 0.02) for c in ["q10", "q25", "q75", "q90"]}
D["seg"] = AGE(D.age_m) + " | " + VOL(D); T["seg"] = AGE(T.age_m) + " | " + VOL(T); WB, WT = widths(D), widths(T); e = np.log(D.y.values + 1) - np.log(D.F.values + 1)
fac = {}
for sg in sorted(D.seg.unique()):
    m = (D.seg == sg).values
    fac[sg] = dict(n=int(m.sum()), kl10=float(np.quantile(-e[m] / WB["q10"][m], 0.9)), ku10=float(np.quantile(e[m] / WB["q90"][m], 0.9)), kl25=float(np.quantile(-e[m] / WB["q25"][m], 0.75)), ku25=float(np.quantile(e[m] / WB["q75"][m], 0.75)))
F = pd.DataFrame(fac).T; print(F.round(3).to_string())
K = F.reindex(T.seg.values); O = pd.DataFrame({"style_color": T.style_color, "segment": T.seg, "r10": np.exp(-K.kl10.values * WT["q10"]), "r25": np.exp(-K.kl25.values * WT["q25"]), "r75": np.exp(K.ku25.values * WT["q75"]), "r90": np.exp(K.ku10.values * WT["q90"])})
assert O[["r10", "r25", "r75", "r90"]].notna().all().all() and (O.r10 <= O.r25).all() and (O.r25 <= 1).all() and (O.r75 >= 1).all() and (O.r75 <= O.r90).all()
W = pd.read_excel("Edikted_Forecasting_Exercise_filled.xlsx", sheet_name="Forecast"); tg = set(W.iloc[:, 1].dropna().astype(str)); O["target"] = O.style_color.isin(tg)
print(O[["r10", "r25", "r75", "r90"]].describe().round(3).to_string()); print("targets:", int(O.target.sum()), "| segments:", O.segment.value_counts().to_dict())
hdr = ["# D329 (fork): per-segment (Mondrian) calibrated range ratios at the June 2026 origin, for Main's candidate_nx5 (D366).",
       "# segment = age band at the origin (months since the item's first sale: under 3, 3-6, 6-12, over 12) x volume third (recency level = the last three months' average monthly units, ranked within the origin over all 1,423 items: low / mid / high).",
       "# widths from the conformal TabPFN member's 2026-06 quantiles (log((q50+1)/(qX+1)), floored at 0.02); each side's factor per segment fitted on all 13 backtest origins around the submitted forecast (10% / 25% of actuals outside each side).",
       "# apply as P = (P50 + 1) x r - 1; for P50 above about 20 units P50 x r is within 5%. Backtest (D328, factors left one origin out): P10-P90 79.9% inside, 78-83% by segment; P25-P75 51.7%, 48-55% by segment."]
with open("data/fk_ranges_2026.csv", "w", encoding="utf-8", newline="") as fh: fh.write("\n".join(hdr) + "\n"); O.round(4).to_csv(fh, index=False)
print("written data/fk_ranges_2026.csv", len(O))
