"""D365 (user, via Fork 2: "calibrate on more recent origins"). Time-honest conformal shifts for the candidate system's ranges, fitted on the
most recent closed origins only. Ranges (backtest, candidate system D361 = data/geo_ecnsnnhubrq_x_preds.parquet geo as P50):
  P25 / P75  the D364 raw weighted member quantiles x the correction multiplier (data/system_q2575_backtest.parquet; NN = the fork's calibrated)
  P10 / P90  the D317 rule: P50 x q10 / q50 and P50 x (q90 / q50)^1.4 of the conformal TabPFN member
At origin o, each side separately: s = the tau-quantile of log((y + 1) / (P_side + 1)) over the calibration set, P_side x exp(s), where the
calibration set is
  K2, K3, K4, K6   the last K origins whose windows have closed (o' + 4 <= o); fewer if fewer exist; none -> raw (no shift)
  H2, H3           every closed origin, weighted 0.5^(age in origins / half-life) (a weighted quantile)
  raw              no shift
Choice: the method with the smallest coverage error (|below - tau_lo| + |above - (1 - tau_hi)|, in points) on the earlier half of the recent
origins (2025-06 .. 2025-10), checked on the later half (2025-11 .. 2026-02); the same choice for both range pairs is not forced.
Reported: below / inside / above for recent, all 13, the June 2025 origin, the early four (they have 0 closed origins at 2025-02 .. 05 for every
method, so they fall back to raw), and both halves; and the 2026-06 shifts from the last K closed origins (2025-11 .. 2026-02 for K = 4).
Writes eda/calib_recent_output.txt (stdout) and data/calib_recent_2026.json."""
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
Q = pd.read_parquet("data/system_q2575_backtest.parquet"); key = ["origin", "style_color"]
G = pd.read_parquet("data/geo_ecnsnnhubrq_x_preds.parquet")[key + ["geo"]]; C = pd.read_parquet("data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet").drop_duplicates(key)[key + ["q10", "q50", "q90"]]
D = Q[key + ["y", "P25", "P50", "P75"]].merge(C, on=key, how="left"); D = D[D.y.notna()].reset_index(drop=True)
D["P10"] = np.minimum(D.P50 * D.q10 / D.q50.clip(lower=1), D.P50); D["P90"] = np.maximum(D.P50 * np.exp(1.4 * np.log((D.q90 / D.q50.clip(lower=1)).clip(lower=1))), D.P50)
ORIG = sorted(D.origin.unique()); oi = {o: i for i, o in enumerate(ORIG)}; ordn = {o: pd.Period(o, "M").ordinal for o in ORIG}
PAIRS = {"P25/P75": ("P25", 0.25, "P75", 0.75), "P10/P90": ("P10", 0.10, "P90", 0.90)}
def wq(v, w, t):
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w) / w.sum(); return float(np.interp(t, c, v))
def calibrate(lo_c, t_lo, hi_c, t_hi, method):
    lo, hi = D[lo_c].values.copy(), D[hi_c].values.copy(); rl = np.log((D.y + 1) / (D[lo_c] + 1)).values; rh = np.log((D.y + 1) / (D[hi_c] + 1)).values; shifts = {}
    for o in ORIG:
        closed = [p for p in ORIG if ordn[p] + 4 <= ordn[o]]; m = (D.origin == o).values
        if not closed or method == "raw": continue
        if method.startswith("K"): use = closed[-int(method[1:]):]; cal = D.origin.isin(use).values; w = np.ones(cal.sum())
        else: h = float(method[1:]); cal = D.origin.isin(closed).values; age = np.array([len(closed) - 1 - closed.index(p) for p in D.origin[cal]]); w = 0.5 ** (age / h)
        s_lo, s_hi = wq(rl[cal], w, t_lo), wq(rh[cal], w, t_hi); shifts[o] = (s_lo, s_hi)
        lo[m] = np.minimum((lo[m] + 1) * np.exp(s_lo) - 1, D.P50.values[m]); hi[m] = np.maximum((hi[m] + 1) * np.exp(s_hi) - 1, D.P50.values[m])
    return lo, hi
SUBS = {"recent": D.origin >= "2025-06", "all 13": D.origin > "", "June 2025": D.origin == "2025-06", "early 4": D.origin <= "2025-05",
        "pick half (2025-06..10)": D.origin.between("2025-06", "2025-10"), "check half (2025-11..2026-02)": D.origin >= "2025-11"}
METHODS = ["raw", "K2", "K3", "K4", "K6", "H2", "H3"]; out = {}
for pair, (lc, tl, hc, th) in PAIRS.items():
    rows = []
    for meth in METHODS:
        lo, hi = calibrate(lc, tl, hc, th, meth)
        for sn, sm in SUBS.items():
            m = sm.values; b = (D.y.values[m] < lo[m]).mean(); a = (D.y.values[m] > hi[m]).mean()
            rows.append({"range": pair, "method": meth, "set": sn, "below": round(100 * b, 1), "inside": round(100 * (1 - a - b), 1), "above": round(100 * a, 1),
                         "cov_error": round(100 * (abs(b - tl) + abs(a - (1 - th))), 1)})
    R = pd.DataFrame(rows); out[pair] = R
    piv = R.pivot_table(index="method", columns="set", values="cov_error").reindex(METHODS); print(f"\n== {pair}: coverage error in points (|below - {tl:.0%}| + |above - {1 - th:.0%}|)"); print(piv.to_string())
    best = piv["pick half (2025-06..10)"].idxmin(); out[pair + "_pick"] = best
    print(f"   picked on the earlier half: {best} | check half: {piv.loc[best, 'check half (2025-11..2026-02)']} vs raw {piv.loc['raw', 'check half (2025-11..2026-02)']}")
    print(R[R.method.isin(["raw", best]) & R.set.isin(["recent", "all 13", "June 2025", "early 4", "check half (2025-11..2026-02)"])].to_string(index=False))
# 2026-06 shifts from the last closed origins (o' + 4 <= 2026-06: 2025-02 .. 2026-02)
res = {}
for pair, (lc, tl, hc, th) in PAIRS.items():
    meth = out[pair + "_pick"]; closed = ORIG
    rl = np.log((D.y + 1) / (D[lc] + 1)).values; rh = np.log((D.y + 1) / (D[hc] + 1)).values
    if meth == "raw": s = (0.0, 0.0)
    elif meth.startswith("K"): cal = D.origin.isin(closed[-int(meth[1:]):]).values; s = (wq(rl[cal], np.ones(cal.sum()), tl), wq(rh[cal], np.ones(cal.sum()), th))
    else:
        h = float(meth[1:]); age = np.array([len(closed) - 1 - closed.index(p) for p in D.origin]); w = 0.5 ** (age / h); s = (wq(rl, w, tl), wq(rh, w, th))
    beats = all(out[pair][(out[pair].method == meth) & (out[pair].set == st)].cov_error.iloc[0] < out[pair][(out[pair].method == "raw") & (out[pair].set == st)].cov_error.iloc[0] for st in ("recent", "June 2025"))
    res[pair] = {"method": meth, "shift_lo": s[0], "shift_hi": s[1], "beats_raw_on_recent_and_june2025": bool(beats), "apply": bool(beats and meth != "raw")}
    print(f"\n2026-06 {pair}: method {meth}, log shifts lower {s[0]:+.3f} upper {s[1]:+.3f} | beats raw on recent and June 2025: {beats} -> {'APPLY' if res[pair]['apply'] else 'keep raw'}")
json.dump(res, open("data/calib_recent_2026.json", "w"), indent=1)
