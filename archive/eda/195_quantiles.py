"""D364 (user, via Fork 2: "all the models should output 0.25 quantile and 0.75 quantile, not only median"). P25 / P75 for every member and
for the candidate system (D361: Huber tree + rules + new NN + conformal TabPFN + TabPFN dyn + rc_deepfm at a quarter weight, corrections).
Members' 4-month P25 / P75:
  tree    quantile LightGBM at alpha 0.25 / 0.75 on the ec_ns inputs (eda/187_losses.py q25 / q75; data/q_backtest_ls_q25*.parquet)
  rules   point forecasts only: P_tau = (rules + 1) x exp(q_tau) - 1, q_tau = the tau-quantile of log((y + 1) / (rules + 1)) on earlier
          origins (o' + 4 <= o), by age band (< 6, 6-12, 12+ months) x volume tercile, shrunk to the all-items quantile by n / (n + 50)
  nn      the NN's monthly q25 / q75 summed over the four months (data/nn_backtest_preds_S13ecx6, finalSec + finalSec_s10 for 2026)
  pf2c, pf2d32   their own q25 / q75 (conformal TabPFN; TabPFN dyn 32)
  rc      Fork 2's pinball versions (data/rc_recsys_preds_q25 / _q75, f_honest_deepfm) if present, else its median (counted)
System: the members' P25 (and P75) averaged with the candidate's weights (1, 1, 1, 1, 1, 0.25), times the item's correction multiplier
(geo / F of data/geo_ecnsnnhubrq_x_preds.parquet; 2026: data/geo_ecnsnnhubrq_x_2026.parquet). Then a time-honest per-side calibration:
at each origin, s_lo = the 0.25-quantile of log((y + 1) / (P25 + 1)) and s_hi = the 0.75-quantile of log((y + 1) / (P75 + 1)) over earlier
origins, P25 x exp(s_lo), P75 x exp(s_hi) (unadjusted where no earlier origin is closed). DECISION (D364): the calibration is reported as a
diagnostic only. It learns from the early origins, which over-forecast, and moves recent coverage the wrong way (as D317 found for P10 / P90),
so the delivered P25 / P75 are the raw weighted quantiles x the correction multiplier (columns P25 / P75; the calibrated ones are P25_cal / P75_cal).
Coverage reported: share below P25, above P75 and inside, raw and calibrated, recent / all 13 / June 2025, plus the pinball loss.
Writes data/system_q2575_backtest.parquet, data/system_q2575_2026.parquet and eda/q2575_output.txt (stdout)."""
import sys, os, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
B = pd.read_parquet("data/dec_base_ecnsnnhubrq.parquet"); key = ["origin", "style_color"]; K = pd.MultiIndex.from_frame(B[key])
q = lambda f, c: pd.read_parquet(f).drop_duplicates(key).set_index(key)[c].reindex(K).values
nnq = lambda f, c: pd.read_parquet(f).groupby(key)[c].sum().reindex(K).values
oo = pd.PeriodIndex(B.origin, freq="M").astype(int).values
B["age_b"] = pd.cut(B.age_m, [-1, 6, 12, 999], labels=False); B["vol_t"] = B.groupby("origin").R3.transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False))
def rules_q(D, train_mask_fn, tau):
    out = np.full(len(D), np.nan); r = np.log((D.y + 1) / (D.rules + 1))
    for o in D.origin.unique():
        m = (D.origin == o).values; tr = train_mask_fn(o) & np.isfinite(r.values)
        if tr.sum() < 200: continue
        g = np.quantile(r.values[tr], tau); seg = pd.DataFrame({"s": D.age_b.astype(str).values + "_" + D.vol_t.astype(str).values, "r": r.values})
        qs = seg[tr].groupby("s").r.agg(lambda v: np.quantile(v, tau)); n = seg[tr].groupby("s").size(); qs = (n * qs + 50 * g) / (n + 50)
        out[m] = (D.rules.values[m] + 1) * np.exp(seg[m].s.map(qs).fillna(g).values) - 1
    return np.clip(out, 0, None)
for tau, c in [(0.25, "q25"), (0.75, "q75")]:
    B[f"rules_{c}"] = rules_q(B, lambda o: oo + 4 <= pd.Period(o).ordinal, tau)
    B[f"tree_{c}"] = q(f"data/q_backtest_ls_{c}.parquet", "q50"); B[f"nn_{c}"] = nnq("data/nn_backtest_preds_S13ecx6.parquet", c)
    if os.environ.get("NNCAL", "1") == "1":   # D364b: the fork's D327 calibrated NN quantiles (factors fitted leaving one origin out: not strictly time-honest in the backtest)
        _fq = pd.read_parquet("data/fk_member_quantiles.parquet"); _fq = _fq[_fq.member == "nn"].drop_duplicates(key).set_index(key)[f"{c}_cal"]; B[f"nn_{c}"] = _fq.reindex(K).values
    B[f"pf2c_{c}"] = q("data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet", c); B[f"pf2d32_{c}"] = q("data/q_backtest_fktab_tabpfn_v2_dyn_e32.parquet", c)
    f = f"data/rc_recsys_preds_{c}.parquet"
    B[f"rc_{c}"] = q(f, "f_honest_deepfm") if os.path.exists(f) else B.rc.values
    print(f"{c}: rc quantile file {'used' if os.path.exists(f) else 'MISSING, median used'}")
W = np.array([1, 1, 1, 1, 1, 0.25]); MEMS = ["rules", "tree", "nn", "pf2c", "pf2d32", "rc"]
G = pd.read_parquet("data/geo_ecnsnnhubrq_x_preds.parquet").set_index(key).reindex(K); mult = (G.geo / G.F.clip(lower=1e-9)).fillna(1).values
B["P50"] = G.geo.values
def wavg(cols):
    X = np.column_stack([np.clip(B[c].values, 0, None) for c in cols]); ok = np.isfinite(X)
    return np.where(ok.any(1), np.nansum(np.where(ok, X, 0) * W, 1) / np.maximum((ok * W).sum(1), 1e-9), np.nan)
for c in ["q25", "q75"]: B[f"P{c[1:]}_raw"] = wavg([f"{m}_{c}" for m in MEMS]) * mult
B["P25_raw"] = np.minimum(B.P25_raw, B.P50); B["P75_raw"] = np.maximum(B.P75_raw, B.P50)
B["P25"] = B.P25_raw; B["P75"] = B.P75_raw; B["P25_cal"] = B.P25_raw; B["P75_cal"] = B.P75_raw; lo = np.log((B.y + 1) / (B.P25_raw + 1)); hi = np.log((B.y + 1) / (B.P75_raw + 1)); S = {}
for o in sorted(B.origin.unique()):
    tr = (oo + 4 <= pd.Period(o).ordinal) & np.isfinite(lo.values); m = (B.origin == o).values
    if tr.sum() < 200: continue
    s_lo, s_hi = np.quantile(lo.values[tr], 0.25), np.quantile(hi.values[tr], 0.75); S[o] = (s_lo, s_hi)
    B.loc[m, "P25_cal"] = np.minimum((B.P25_raw[m] + 1) * np.exp(s_lo) - 1, B.P50[m]); B.loc[m, "P75_cal"] = np.maximum((B.P75_raw[m] + 1) * np.exp(s_hi) - 1, B.P50[m])
pin = lambda y, p, t: np.mean(np.maximum(t * (y - p), (t - 1) * (y - p)))
print("\n== coverage (share of actual 4-month totals below P25 / inside / above P75; target 25 / 50 / 25) and pinball / mean units")
for nm, msk in [("recent", B.origin >= "2025-06"), ("all 13", B.origin > ""), ("June 2025 origin", B.origin == "2025-06"), ("early 4", B.origin <= "2025-05")]:
    D = B[msk]
    for v in ["raw", "cal"]:
        p25, p75 = (D.P25_raw, D.P75_raw) if v == "raw" else (D.P25_cal, D.P75_cal)
        below, above = (D.y < p25).mean(), (D.y > p75).mean()
        print(f"  {nm:16s} {v}: below {below:.1%} | inside {1 - below - above:.1%} | above {above:.1%} | pinball {(pin(D.y, p25, 0.25) + pin(D.y, p75, 0.75)) / 2 / D.y.mean():.4f} | median width P75/P25 {np.median((p75 + 1) / (p25 + 1)):.2f}")
print("\n== members alone, recent: share below their P25 / above their P75")
R = B[B.origin >= "2025-06"]
for m in MEMS: print(f"  {m:7s} below {(R.y < R[f'{m}_q25']).mean():.1%} | above {(R.y > R[f'{m}_q75']).mean():.1%}")
B[key + ["y", "P25", "P50", "P75", "P25_cal", "P75_cal"] + [f"{m}_{c}" for m in MEMS for c in ("q25", "q75")]].to_parquet("data/system_q2575_backtest.parquet", index=False)
# 2026-06
B26 = pd.read_parquet("data/dec_base_ecnsnnhubrq26.parquet"); B26 = B26[B26.origin == "2026-06"].copy(); K26 = B26.style_color.values
B26["age_b"] = pd.cut(B26.age_m, [-1, 6, 12, 999], labels=False); B26["vol_t"] = pd.qcut(B26.R3.rank(method="first"), 3, labels=False)
ALL = pd.concat([B, B26], ignore_index=True); oo2 = pd.PeriodIndex(ALL.origin, freq="M").astype(int).values
for tau, c in [(0.25, "q25"), (0.75, "q75")]:
    ALL[f"rules_{c}"] = rules_q(ALL, lambda o: oo2 + 4 <= pd.Period(o).ordinal, tau); r26 = ALL[ALL.origin == "2026-06"].set_index("style_color")
    B26[f"rules_{c}"] = r26[f"rules_{c}"].reindex(K26).values
    s1 = lambda f, c_: pd.read_parquet(f).drop_duplicates("style_color").set_index("style_color")[c_].reindex(K26).values
    B26[f"tree_{c}"] = s1(f"data/q_backtest_ls_{c}_2026.parquet", "q50")
    B26[f"nn_{c}"] = (sum(pd.read_parquet(f"data/{f}.parquet").groupby("style_color")[c].sum() for f in ["nn_backtest_preds_finalSec", "nn_backtest_preds_finalSec_s10"]) / 2).reindex(K26).values
    if os.environ.get("NNCAL", "1") == "1":
        _fq = pd.read_parquet("data/fk_member_quantiles.parquet"); _fq = _fq[(_fq.member == "nn") & (_fq.origin == "2026-06")].drop_duplicates("style_color").set_index("style_color")[f"{c}_cal"]; B26[f"nn_{c}"] = _fq.reindex(K26).values
    B26[f"pf2c_{c}"] = s1("data/fk_2026_pf2c.parquet", c); B26[f"pf2d32_{c}"] = s1("data/fk_2026_pf2d32.parquet", c)
    f = f"data/rc_recsys_preds_{c}.parquet"
    if os.path.exists(f):
        x = pd.read_parquet(f); x = x[x.origin == "2026-06"].drop_duplicates("style_color").set_index("style_color").f_honest_deepfm; B26[f"rc_{c}"] = x.reindex(K26).values
    else: B26[f"rc_{c}"] = B26.rc.values
G26 = pd.read_parquet("data/geo_ecnsnnhubrq_x_2026.parquet").set_index("style_color").reindex(K26); m26 = (G26.geo / G26.F.clip(lower=1e-9)).fillna(1).values
X = lambda c: np.column_stack([np.clip(B26[f"{m}_{c}"].values, 0, None) for m in MEMS])
def wavg26(c):
    A = X(c); ok = np.isfinite(A); return np.where(ok.any(1), np.nansum(np.where(ok, A, 0) * W, 1) / np.maximum((ok * W).sum(1), 1e-9), np.nan)
lo_all, hi_all = lo[np.isfinite(lo)].values, hi[np.isfinite(hi)].values; s_lo, s_hi = np.quantile(lo_all, 0.25), np.quantile(hi_all, 0.75)
P50 = G26.geo.values; P25 = np.minimum(wavg26("q25") * m26, P50); P75 = np.maximum(wavg26("q75") * m26, P50)   # D364: raw (calibration not applied, see the docstring)
pd.DataFrame({"style_color": K26, "P25": P25, "P50": P50, "P75": P75}).to_parquet("data/system_q2575_2026.parquet", index=False)
print(f"\n2026-06: {len(K26)} items; raw ranges (the calibration, not applied, would shift in logs) lower {s_lo:+.3f}, upper {s_hi:+.3f}; median width P75/P25 {np.nanmedian((P75 + 1) / (P25 + 1)):.2f}")
