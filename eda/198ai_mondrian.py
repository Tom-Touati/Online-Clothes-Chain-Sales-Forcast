"""D328 (fork; user comment on the atlas, 'Where the error lives': "what can we do for those? why havent they improved? perhaps a separate models
and calibration for some of these categories"). Per-segment (Mondrian) calibration of the system's ranges, against the global rule of D317:
  segments  age at the origin (under 3 months, 3-6, 6-12, over 12) x volume (three bands of the recency level within the origin)
  ranges    P10-P90 and P25-P75 around the submitted forecast F, from the conformal TabPFN quantiles' log widths (as D317)
  rules     as stated | global: lower x1.0, upper log width x1.4 (D317; P25-P75 sides scaled to 25% / 25% globally) |
            Mondrian: each side's factor fitted per segment, leaving one origin out, so that 10% / 25% fall outside on each side
Scored on recent origins: inside share per segment and overall, and the interval score (width + 2/alpha x the miss outside, in logs; lower is
better). Also the share of actuals above the forecast per segment (where under-forecasting concentrates, for a per-segment buying quantile).
Writes data/fk_mondrian_backtest.csv; adds J["mondrian"]; stdout -> eda/mondrian_output.txt."""
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
r = lambda x, k=3: None if x is None or not np.isfinite(x) else round(float(x), k)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); ORIG = sorted(D.origin.unique()); rec = (D.origin >= "2025-06").values
Q = pd.read_parquet("data/fk_member_quantiles.parquet"); Q = Q[Q.member == "pf2c"].set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([D.origin, D.style_color]))
D["age_b"] = pd.cut(D.age_m, [-1, 3, 6, 12, 999], labels=["under 3 months", "3-6 months", "6-12 months", "over 12 months"]).astype(str)
D["vol_b"] = D.groupby("origin")["level"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"])).astype(str); D["seg"] = D.age_b + " | " + D.vol_b
lF = np.log(D.F.values + 1); ly = np.log(D.y.values + 1); e = ly - lF
W = {}
for lo_q, hi_q in [("q10", "q90"), ("q25", "q75")]:
    lq = np.log(Q[lo_q].values + 1) - np.log(Q.q50.values + 1); hq = np.log(Q[hi_q].values + 1) - np.log(Q.q50.values + 1)   # the member's own widths, moved onto F
    W[lo_q] = (np.maximum(-lq, 0.02), np.maximum(hq, 0.02))
def isc(lo, hi, a):
    below, above = ly < lo, ly > hi; return (hi - lo) + (2 / a) * np.where(below, lo - ly, 0) + (2 / a) * np.where(above, ly - hi, 0), below, above
out = {}; rows = []
for lo_q, a in [("q10", 0.2), ("q25", 0.5)]:
    wl, wu = W[lo_q]; tail = a / 2
    fit = lambda m: (float(np.quantile(-e[m] / wl[m], 1 - tail)), float(np.quantile(e[m] / wu[m], 1 - tail)))
    rules = {"as stated": (lF - wl, lF + wu)}
    if lo_q == "q10": rules["global (D317: upper x1.4)"] = (lF - wl, lF + 1.4 * wu)
    else:
        lo_g, hi_g = np.zeros(len(D)), np.zeros(len(D))
        for o in ORIG:
            kl, ku = fit((D.origin != o).values); m = (D.origin == o).values; lo_g[m] = lF[m] - kl * wl[m]; hi_g[m] = lF[m] + ku * wu[m]
        rules["global, fitted leaving one origin out"] = (lo_g, hi_g)
    lo_m, hi_m = np.zeros(len(D)), np.zeros(len(D)); fac = {}
    for sg in D.seg.unique():
        for o in ORIG:
            tr = ((D.seg == sg) & (D.origin != o)).values; m = ((D.seg == sg) & (D.origin == o)).values
            if not m.any(): continue
            kl, ku = fit(tr) if tr.sum() >= 50 else fit((D.origin != o).values); lo_m[m] = lF[m] - kl * wl[m]; hi_m[m] = lF[m] + ku * wu[m]
        fac[sg] = [r(v, 2) for v in fit((D.seg == sg).values)]
    rules["Mondrian: per age x volume segment, leaving one origin out"] = (lo_m, hi_m)
    for nm, (lo, hi) in rules.items():
        s, b, ab = isc(lo, hi, a); d = pd.DataFrame({"seg": D.seg, "inside": ~b & ~ab, "below": b, "above": ab, "isc": s})[rec]
        rows.append(dict(range="P10-P90" if lo_q == "q10" else "P25-P75", rule=nm, inside=r(d.inside.mean()), below=r(d.below.mean()), above=r(d.above.mean()), interval_score=r(d.isc.mean()),
                         worst_seg=d.groupby("seg").inside.mean().idxmin(), worst_inside=r(d.groupby("seg").inside.mean().min()), best_inside=r(d.groupby("seg").inside.mean().max()),
                         seg_spread=r(float(d.groupby("seg").inside.mean().std()))))
        out[(lo_q, nm)] = d.groupby("seg")[["inside", "below", "above"]].mean().round(3)
    out[(lo_q, "factors")] = fac
R = pd.DataFrame(rows); print(R.to_string(index=False))
seg = out[("q10", "as stated")].join(out[("q10", "global (D317: upper x1.4)")].add_suffix("_global")).join(out[("q10", "Mondrian: per age x volume segment, leaving one origin out")].add_suffix("_mondrian")); print("\nP10-P90 by segment (inside / below / above), recent:"); print(seg.to_string())
ab = D[rec].assign(above=(D.y > D.F)[rec]).groupby("seg").agg(above_forecast=("above", "mean"), n=("above", "size"), units=("y", "sum")); ab["units"] = (ab.units / ab.units.sum()).round(3); print("\nshare of actuals above the forecast, by segment:"); print(ab.round(3).to_string())
seg.reset_index().to_csv("data/fk_mondrian_backtest.csv", index=False)
J = json.load(open("data/fk_err_analysis.json")); J["mondrian"] = dict(rows=rows, by_segment=seg.reset_index().to_dict("records"), above_forecast=ab.reset_index().to_dict("records"), factors_p10_p90=out[("q10", "factors")], factors_p25_p75=out[("q25", "factors")])
json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0, default=float); print("saved")
