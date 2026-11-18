"""Fork 2 (overnight goal: "optimize any other bits on the meaningful months with the new metrics"): the confidence column against the brief's own
definitions (1 = within +/-20%, 2 = within +/-50%, 3 = could be off by more than 50%), on the new nx5's time-honest backtest.
  D318    the submitted rule: level 1 = narrowest fifth of the conformal TabPFN P10-P90 log width AND rising 91-day trend AND sold in the same
          months last year; 3 = widest fifth; else 2
  ranges  from the item's calibrated ranges (D389i, leave-one-out, data/ranges_nx5r_loo.parquet): the predictive CDF interpolated in logs
          through P10 / P25 / P50 / P75 / P90; p20 = P(within +/-20%), p50 = P(within +/-50%); level 1 if p20 >= t1, 3 if p50 < t3, else 2
          (t1 / t3 chosen on the tune origins 2025-06 .. 2025-09 to maximise the share of claims met, read on the held-out origins)
Claim met: level 1 within +/-20%, level 2 within +/-50%, level 3 off by more than 50% (for 3, "could be" -- we also report its within-50% rate).
Reads: share of claims met overall and per level, level sizes; recent 9, held out, Jul-Oct 25, Mar-Jun 26. stdout -> eda/confidence_new_output.txt"""
import sys, itertools, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
G = pd.read_parquet("data/geo_nx5r_x_preds.parquet")[["origin", "style_color", "geo", "y"]].merge(pd.read_parquet("data/ranges_nx5r_loo.parquet"), on=["origin", "style_color"])
E = pd.read_parquet("data/fk_err_master.parquet")[["origin", "style_color", "trend91", "has_ly"]]
Q = pd.read_parquet("data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet").drop_duplicates(["origin", "style_color"])[["origin", "style_color", "q10", "q50", "q90"]]
G = G.merge(E, on=["origin", "style_color"], how="left").merge(Q, on=["origin", "style_color"], how="left")
F, y = G.geo.values, G.y.values; O = G.origin.values
G["iw"] = np.log((G.q90 + 1) / (G.q10 + 1)); G["fifth"] = G.groupby("origin")["iw"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)) + 1
d318 = np.where((G.fifth == 1) & (G.trend91.fillna(-1) > 0) & (G.has_ly.fillna(0) == 1), 1, np.where(G.fifth == 5, 3, 2))
# predictive CDF through the five quantiles, in logs; tails extended linearly in log to 0 / 1 at 2x the outer gaps
L = np.log1p(np.column_stack([G.p10, G.p25, F, G.p75, G.p90]).clip(min=0)); L = np.maximum.accumulate(L, axis=1)
def cdf(x):
    lx = np.log1p(np.maximum(x, 0)); out = np.empty(len(lx))
    for i in range(len(lx)):
        k = L[i]; xs = np.r_[k[0] - 2 * max(k[1] - k[0], 0.05), k, k[4] + 2 * max(k[4] - k[3], 0.05)]; ps = np.r_[0, .1, .25, .5, .75, .9, 1]
        out[i] = np.interp(lx[i], xs, ps)
    return out
p20 = cdf(1.2 * F) - cdf(0.8 * F); p50 = cdf(1.5 * F) - cdf(0.5 * F)
err = np.abs(F - y) / np.maximum(y, 1e-9)
def met(lv): return np.where(lv == 1, err <= 0.2, np.where(lv == 2, err <= 0.5, err > 0.5))
def report(lv, m, lab):
    s = f"{lab:28s} claims met {met(lv)[m].mean():.3f} |"
    for k in (1, 2, 3):
        mk = m & (lv == k)
        s += f" L{k}: {mk.sum() / m.sum():.0%} of items, within20 {np.mean(err[mk] <= .2) if mk.any() else float('nan'):.2f} within50 {np.mean(err[mk] <= .5) if mk.any() else float('nan'):.2f} |"
    print(s)
TUNE = (O >= "2025-06") & (O <= "2025-09"); HOLD = O >= "2025-10"; REC = O >= "2025-06"
best = max(itertools.product([0.3, 0.35, 0.4, 0.45, 0.5, 0.55], [0.4, 0.45, 0.5, 0.55, 0.6]), key=lambda t: met(np.where(p20 >= t[0], 1, np.where(p50 < t[1], 3, 2)))[TUNE].mean())
rng = lambda t: np.where(p20 >= t[0], 1, np.where(p50 < t[1], 3, 2))
print("chosen thresholds t1 (p20) / t3 (p50):", best)
for lab, m in [("recent 9", REC), ("held out", HOLD), ("Jul-Oct 25", O == "2025-06"), ("Mar-Jun 26", O == "2026-02")]:
    print(f"\n== {lab}"); report(d318, m, "D318 (submitted)"); report(rng(best), m, "ranges rule"); report(np.full(len(G), 2), m, "everything 2")
pd.DataFrame({"origin": O, "style_color": G.style_color, "p20": p20, "p50": p50}).to_parquet("data/confidence_new_bt.parquet", index=False)
# hybrid (added after the first read): level 1 from D318 (the validated narrow + rising + last-year rule), level 3 from the ranges (p50 < t3)
hyb = np.where(d318 == 1, 1, np.where(p50 < best[1], 3, 2))
print("\n== hybrid: D318 level 1, ranges level 3")
for lab, m in [("recent 9", REC), ("held out", HOLD), ("Jul-Oct 25", O == "2025-06"), ("Mar-Jun 26", O == "2026-02")]: report(hyb, m, lab)
