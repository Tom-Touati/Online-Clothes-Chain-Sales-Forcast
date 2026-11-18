"""Fork 2 (user: "calibrate everything including the ensemble weights to last year's test months and the recent 4 months").
Calibration windows: last year's test months Jul-Oct 2025 (origin 2025-06) and the most recent four months Mar-Jun 2026 (origin 2026-02).
The submitted system = arithmetic mean of five members (rules, tree ec_ns, NN ec (fallback NN), TabPFN conformal, TabPFN 32) x the
segment / breakout correction multiplier (geo / F, eda/159). Calibrated on the two windows:
  weights  non-negative, summing to 1 (with and without the correction multiplier)
  level    one overall scale on the forecast
  ranges   P10 / P90 widths (log distance from the forecast, separate powers below / above) to 10% / 90% coverage
Honest reads: fit on one window, score the other (both ways); fit on both, score the 11 other origins. Scores WAPE / R3 with a style
bootstrap of the difference against the submitted system. Applied to 2026 (data/final_2026_nx4_cascade.csv) only as a candidate file:
data/calib2w_2026.csv. Nothing in the submission changes. Output -> eda/calib2w_output.txt"""
import sys, json, numpy as np, pandas as pd
from scipy.optimize import minimize
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True)
K = pd.MultiIndex.from_arrays([D.origin, D.style_color])
q50 = lambda f: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(K).q50.values
nn_ec = pd.read_parquet("data/nn_backtest_preds_S13ecx6.parquet").groupby(["origin", "style_color"]).q50.sum().reindex(K).values
M = np.column_stack([D.rules.values, q50("data/q_backtest_nx_ec_ns.parquet"), np.where(np.isfinite(nn_ec), nn_ec, D.nn.values),
                     q50("data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet"), q50("data/q_backtest_fktab_tabpfn_v2_dyn_e32.parquet")])
NAMES = ["rules", "tree", "nn", "pf2c", "pf2d32"]
g = pd.read_parquet("data/geo_ecnsnn_x_preds.parquet").drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(K)
cm = (g.geo / g.F.clip(lower=1e-9)).fillna(1).values
y, R3, sty = D.y.values, D.R3.values, D["style"].values
assert np.isfinite(M).all(), np.isfinite(M).sum(0)
O = D.origin.values
C1, C2 = O == "2025-06", O == "2026-02"; C = C1 | C2
REC = O >= "2025-06"; OTHER = ~C; HOLD = (O >= "2025-10") & (O <= "2026-01")
w_eq = np.ones(5) / 5
pred = lambda w, s=1.0, use_cm=True: (M @ w) * (cm if use_cm else 1) * s
wr3 = lambda f, m: float(np.abs(f - y)[m].sum() / np.abs(R3 - y)[m].sum())
def fit(m, use_cm=True, scale=False, weights=True):
    def obj(th):
        w = np.exp(th[:5]) / np.exp(th[:5]).sum() if weights else w_eq
        s = np.exp(th[5]) if scale else 1.0
        return np.abs(pred(w, s, use_cm) - y)[m].sum()
    best = None
    for st in [np.zeros(6), np.r_[np.log([.4, .15, .15, .15, .15]), 0], np.r_[np.log([.1, .3, .3, .15, .15]), 0]]:
        r = minimize(obj, st, method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-5, fatol=1e-3))
        if best is None or r.fun < best.fun: best = r
    th = best.x; w = np.exp(th[:5]) / np.exp(th[:5]).sum() if weights else w_eq
    return w, (float(np.exp(th[5])) if scale else 1.0)
rng = np.random.default_rng(0); us = np.unique(sty)
def boot(fa, fb, m, B=400):
    """95% interval of WAPE/R3(fa) - WAPE/R3(fb) over m, resampling styles"""
    df = pd.DataFrame(dict(s=sty[m], a=np.abs(fa - y)[m], b=np.abs(fb - y)[m], r=np.abs(R3 - y)[m])).groupby("s").sum()
    idx = rng.integers(0, len(df), size=(B, len(df))); A, Bb, Rr = df.a.values[idx].sum(1), df.b.values[idx].sum(1), df.r.values[idx].sum(1)
    d = A / Rr - Bb / Rr; return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
base = pred(w_eq)
print("check: submitted system WAPE/R3 recent", round(wr3(base, REC), 4), "| all 13", round(wr3(base, np.ones(len(y), bool)), 4))
print("check: correction multiplier on the two windows: with", round(wr3(base, C), 4), "without", round(wr3(pred(w_eq, use_cm=False), C), 4))
VARIANTS = {"weights": dict(), "weights + level": dict(scale=True), "level only": dict(scale=True, weights=False), "weights, no corrections": dict(use_cm=False), "weights + level, no corrections": dict(use_cm=False, scale=True)}
rows, fits = [], {}
for vn, kw in VARIANTS.items():
    uc = kw.get("use_cm", True)
    wa, sa = fit(C1, **kw); wb, sb = fit(C2, **kw); wc, sc = fit(C, **kw); fits[vn] = (wc, sc, uc)
    fa, fb, fc = pred(wa, sa, uc), pred(wb, sb, uc), pred(wc, sc, uc)
    rows.append(dict(variant=vn, weights=" ".join(f"{n} {x:.2f}" for n, x in zip(NAMES, wc)), level=round(sc, 3),
                     in_sample=round(wr3(fc, C), 4), sub_in=round(wr3(base, C), 4),
                     fit25_on26=round(wr3(fa, C2), 4), fit26_on25=round(wr3(fb, C1), 4), sub_26=round(wr3(base, C2), 4), sub_25=round(wr3(base, C1), 4),
                     other11=round(wr3(fc, OTHER), 4), sub_other11=round(wr3(base, OTHER), 4), ci_other11=boot(fc, base, OTHER),
                     hold4=round(wr3(fc, HOLD), 4), sub_hold4=round(wr3(base, HOLD), 4), ci_hold4=boot(fc, base, HOLD)))
R = pd.DataFrame(rows); print(R.to_string(index=False))
# cross-window read of each variant, both directions, against the submitted system on the same window
for r in rows: r["cross"] = round(((r["fit25_on26"] - r["sub_26"]) + (r["fit26_on25"] - r["sub_25"])) / 2, 4)
print("\nmean cross-window change (negative = better):", {r["variant"]: r["cross"] for r in rows})
# ranges: P10 / P90 as powers of the backtest's own log distance, tuned to 10% / 90% coverage on the two windows
F, p10, p90 = D.F.values, D.p10.values, D.p90.values
lo, hi = np.log(np.maximum(F, 1e-6) / np.maximum(p10, 1e-6)), np.log(np.maximum(p90, 1e-6) / np.maximum(F, 1e-6))
def pw(m, side, target):
    best = None
    for a in np.linspace(0.3, 3, 271):
        cov = float(((y >= F * np.exp(-a * lo)) if side == "lo" else (y <= F * np.exp(a * hi)))[m].mean())
        e = abs(cov - target); best = (e, a, cov) if best is None or e < best[0] else best
    return best[1]
cov = lambda a, b, m: (round(float((y < F * np.exp(-a * lo))[m].mean()), 3), round(float((y > F * np.exp(b * hi))[m].mean()), 3))
a_lo, a_hi = pw(C, "lo", 0.90), pw(C, "hi", 0.90)
a1, b1 = pw(C1, "lo", .9), pw(C1, "hi", .9); a2, b2 = pw(C2, "lo", .9), pw(C2, "hi", .9)
RG = dict(power_lo=round(a_lo, 2), power_hi=round(a_hi, 2),
          below_above_current={"Jul-Oct 25": cov(1, 1, C1), "Mar-Jun 26": cov(1, 1, C2), "other 11": cov(1, 1, OTHER)},
          below_above_calibrated={"Jul-Oct 25": cov(a_lo, a_hi, C1), "Mar-Jun 26": cov(a_lo, a_hi, C2), "other 11": cov(a_lo, a_hi, OTHER)},
          cross={"fit Jul-Oct 25, read Mar-Jun 26": cov(a1, b1, C2), "fit Mar-Jun 26, read Jul-Oct 25": cov(a2, b2, C1)})
print("\nranges (share below P10, share above P90; target 0.10 / 0.10):", json.dumps(RG, indent=1))
# 2026 candidate: the chosen variant applied to the deliverable's members; totals moved by the same ratio (caps and rounding kept)
Z = pd.read_csv("data/final_2026_nx4_cascade.csv", index_col=0)
M26 = Z[["rules", "qnx", "nn", "pf2c", "pf2d32"]].values; f_old = M26.mean(1) * Z.correction.values
out = {}
for vn, (w, s, uc) in fits.items():
    f_new = (M26 @ w) * (Z.correction.values if uc else 1) * s; out[vn] = Z.total.values * f_new / f_old
C26 = pd.DataFrame({"total_submitted": Z.total.values, **{f"total_{k.replace(' ', '_').replace(',', '')}": np.round(v) for k, v in out.items()}}, index=Z.index)
C26["P10_cal"] = np.round(Z.total.values * (Z.P10.values / Z.forecast.values) ** a_lo); C26["P90_cal"] = np.round(Z.total.values * (Z.P90.values / Z.forecast.values) ** a_hi)
C26.to_csv("data/calib2w_2026.csv")
print("\n2026 totals (100 items):", {k: int(v.sum()) for k, v in C26.items() if k.startswith("total")})
json.dump(dict(rows=rows, ranges=RG, fits={k: dict(weights=[round(float(x), 3) for x in v[0]], level=round(v[1], 3), corrections=v[2]) for k, v in fits.items()},
               totals_2026={k: int(v.sum()) for k, v in C26.items() if k.startswith("total")}), open("data/calib2w.json", "w"), indent=1, default=str)
# cross-window reads with style-bootstrap intervals (added after the first run)
print("\ncross-window, 95% style-bootstrap interval of the change against the submitted system:")
XR = {}
for vn, kw in VARIANTS.items():
    uc = kw.get("use_cm", True); wa, sa = fit(C1, **kw); wb, sb = fit(C2, **kw)
    XR[vn] = {"fit Jul-Oct 25 -> Mar-Jun 26": [round(wr3(pred(wa, sa, uc), C2) - wr3(base, C2), 4), boot(pred(wa, sa, uc), base, C2)],
              "fit Mar-Jun 26 -> Jul-Oct 25": [round(wr3(pred(wb, sb, uc), C1) - wr3(base, C1), 4), boot(pred(wb, sb, uc), base, C1)],
              "weights from Mar-Jun 26": " ".join(f"{n} {x:.2f}" for n, x in zip(NAMES, wb)) + f" level {sb:.3f}"}
    print(vn, XR[vn])
J = json.load(open("data/calib2w.json")); J["cross_ci"] = XR; json.dump(J, open("data/calib2w.json", "w"), indent=1, default=str)
