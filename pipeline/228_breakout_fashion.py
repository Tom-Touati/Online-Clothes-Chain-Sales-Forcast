"""Fable 5.2 fork (user, 18 Nov: "try these" -- idea 5: "Put the two stable signals into the correction layer"). The breakout / fading
residual model of the corrections (pipeline/179b_breakout.py section 4: LightGBM median + L2 on the signals, trained at each origin on windows
ended by it, applied as F x exp(0.5 x rhat), origin total kept) refitted on the submitted system's base (data/dec_base_nx5r26.parquet,
signals from data/breakout_preds_nx5r26_x.parquet) with fashion signals added. 179b is not edited: this is the same loop in its own file.
  v0     the stored signals                                                                 [must reproduce the stored rhat]
  two    + the two signals that tracked the tree's residual at all nine recent origins (D205): photo / text neighbours' last-year
         ramp (nc_sea4) and the last-year ramp of items sharing the item's attribute values (fa_mean_sea4)
  nbfa   + photo / text neighbours (newest turn, momentum, last-year ramp) and attribute-cohort trends (ramp, momentum, LLM-profile
         alignment, share within type)
  llm    + the 14 language-model scores and their season interactions (score x share of the target months in that season)
  lin    v0's rhat plus a small ridge on what it leaves: (e - rhat) on the two standardised signals, fitted on closed windows only
         (two coefficients, alpha 100: a heavily constrained use of a 0.2-0.3 correlation)
Combined as the submitted corrections: geo = sqrt(segment x residual), each origin's total kept (segment from
data/bias_seg_preds_nx5r26.parquet). WAPE / R3 on recent 9 / held out / all 13 / Jul-Oct 25 / Mar-Jun 26 / new items, style bootstrap
against v0. stdout -> pipeline/breakout_fashion_output.txt"""
import os, sys, numpy as np, pandas as pd, lightgbm as lgb, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
K = pd.read_parquet("data/breakout_preds_nx5r26_x.parquet"); B = pd.read_parquet("data/dec_base_nx5r26.parquet")[["origin", "style_color", "y", "R3", "F", "e", "style"]]
drop = {"origin", "style_color", "brk", "rhat"}; SIG = [c for c in K.columns if c not in drop and not c.startswith(("seg_", "res_")) and c != "F"]
B = B.merge(K[["origin", "style_color", "rhat"] + [c for c in SIG if c not in B.columns]].rename(columns={"rhat": "rhat_stored"}), on=["origin", "style_color"])
S = pd.read_parquet("data/bias_seg_preds_nx5r26.parquet")[["origin", "style_color", "vol_age_disc_0.5n"]]; B = B.merge(S, on=["origin", "style_color"])
B = B.merge(pd.read_parquet("data/fk_nnc_feats.parquet")[["origin", "style_color", "nc_acc1", "nc_mom3", "nc_sea4"]], on=["origin", "style_color"], how="left")
FA = ["fa_mean_sea4", "fa_mean_mom", "fa_llm_yoy", "fa_in_type"]; B = B.merge(pd.read_parquet("data/fk_fash_feats.parquet")[["origin", "style_color"] + FA], on=["origin", "style_color"], how="left")
LLM = ["warmth", "coverage", "formality", "fit_tightness", "statement", "staple", "summer", "winter", "back_to_school", "party", "halloween", "festival", "layering", "versatility"]
X = pd.read_parquet("data/fk_explain_table.parquet")[["origin", "style_color"] + LLM].drop_duplicates(["origin", "style_color"]); B = B.merge(X, on=["origin", "style_color"], how="left")
oo = pd.PeriodIndex(B.origin, freq="M").asi8; mo = (oo % 12) + 1; XI = []
for k, months in {"summer": [6, 7, 8], "winter": [12, 1, 2], "halloween": [10], "back_to_school": [8, 9], "party": [11, 12], "festival": [4, 5, 6]}.items():
    B[f"sx_{k}"] = pd.to_numeric(B[k], errors="coerce") * np.mean([np.isin(((mo - 1 + h) % 12) + 1, months) for h in (1, 2, 3, 4)], axis=0); XI.append(f"sx_{k}")
print("signals", len(SIG), "| coverage of the added: nc_sea4", round(B.nc_sea4.notna().mean(), 2), "fa_mean_sea4", round(B.fa_mean_sea4.notna().mean(), 2), flush=True)
P = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10, verbose=-1, n_jobs=int(os.environ.get("NJOBS", 3)))
ORIG = sorted(B.origin.unique()); known = B.y.notna().values
LYG = os.environ.get("LYGATE") == "1"   # the last-year-removed gate: training windows o - 12 +/- 1 dropped from both the residual model and the ridge
gate = lambda o: ~((oo >= pd.Period(o).ordinal - 13) & (oo <= pd.Period(o).ordinal - 11)) if LYG else np.ones(len(B), bool)
def resid(sig):
    r = np.zeros(len(B))
    for o in ORIG:
        tr = (oo + 4 <= pd.Period(o).ordinal) & known & gate(o); m = (B.origin == o).values
        if B.origin[tr].nunique() < 3: continue
        Xt = B.loc[tr, sig].astype(float); yt = B.e[tr].clip(-2, 2); wt = np.sqrt(B.y[tr] + B.F[tr] + 1)
        r[m] = np.mean([lgb.LGBMRegressor(objective=ob, **({"alpha": 0.5} if ob == "quantile" else {}), **P).fit(Xt, yt, sample_weight=wt).predict(B.loc[m, sig].astype(float)) for ob in ("quantile", "l2")], axis=0)
    return r
def lin(r0, cols, alpha=100.0):
    r = r0.copy(); co = []
    for o in ORIG:
        tr = (oo + 4 <= pd.Period(o).ordinal) & known & gate(o); m = (B.origin == o).values
        if B.origin[tr].nunique() < 3: continue
        mu, sd = B.loc[tr, cols].mean(), B.loc[tr, cols].std().replace(0, 1); Z = lambda d: ((d[cols] - mu) / sd).fillna(0).clip(-3, 3).values
        g = Ridge(alpha=alpha).fit(Z(B[tr]), (B.e[tr].clip(-2, 2) - r0[tr]).values, sample_weight=np.sqrt(B.y[tr] + B.F[tr] + 1)); r[m] = r0[m] + g.predict(Z(B[m])); co.append(g.coef_)
    return r, np.round(np.mean(co, axis=0), 4), np.round(np.min(co, axis=0), 4), np.round(np.max(co, axis=0), 4)
def geo(r):
    res = B.F.values * np.exp(0.5 * r); resn = res * B.groupby("origin").F.transform("sum").values / pd.Series(res).groupby(B.origin.values).transform("sum").values
    g = np.sqrt(B["vol_age_disc_0.5n"].clip(lower=0).values * np.clip(resn, 0, None)); return g * B.groupby("origin").F.transform("sum").values / pd.Series(g).groupby(B.origin.values).transform("sum").values
y, R3, O = B.y.values, B.R3.values, B.origin.values; new = pd.read_parquet("data/dec_base_nx5r26.parquet").age_m.fillna(99).values < 6; sty = B["style"].values; rng = np.random.default_rng(0)
wr = lambda f, m: np.abs(f - y)[m].sum() / np.abs(R3 - y)[m].sum()
def ci(a, b, m):
    d = pd.DataFrame(dict(s=sty[m], a=np.abs(a - y)[m], b=np.abs(b - y)[m], r=np.abs(R3 - y)[m])).groupby("s").sum()
    i = rng.integers(0, len(d), size=(400, len(d))); v = (d.a.values[i].sum(1) - d.b.values[i].sum(1)) / d.r.values[i].sum(1); return f"[{np.percentile(v, 2.5):+.4f},{np.percentile(v, 97.5):+.4f}]"
SETS = [("recent", (O >= "2025-06") & known), ("held out", (O >= "2025-10") & known), ("all 13", known), ("JulOct25", O == "2025-06"), ("MarJun26", O == "2026-02"), ("new rec", (O >= "2025-06") & known & new), ("old rec", (O >= "2025-06") & known & ~new)]
r0 = resid(SIG); print("LYGATE" if LYG else "normal", "| v0 reproduces the stored rhat: max abs diff", float(np.abs(r0 - B.rhat_stored.fillna(0).values)[known].max()), "| stored geo check recent", round(wr(geo(B.rhat_stored.fillna(0).values), SETS[0][1]), 4), flush=True)
g0 = geo(r0)
def show(nm, r):
    g = geo(r); d = B[(O >= "2025-06") & known]; rho = pd.Series(r[(O >= "2025-06") & known]).corr(d.e.reset_index(drop=True), method="spearman")
    print(f"{nm:5s} " + " ".join(f"{lab} {wr(g, m):.4f}" for lab, m in SETS) + f" | Spearman(rhat, e) {rho:+.3f} | vs v0: recent {ci(g, g0, SETS[0][1])} held out {ci(g, g0, SETS[1][1])} all {ci(g, g0, SETS[2][1])}", flush=True)
show("v0", r0)
show("two", resid(SIG + ["nc_sea4", "fa_mean_sea4"]))
show("nbfa", resid(SIG + ["nc_acc1", "nc_mom3", "nc_sea4"] + FA))
show("llm", resid(SIG + LLM + XI))
for cols, a in [(["nc_sea4", "fa_mean_sea4"], 100.0), (["nc_sea4", "fa_mean_sea4"], 1000.0), (["nc_sea4"], 100.0)]:
    r, c, lo, hi = lin(r0, cols, a); show("lin", r); print("      ridge on", cols, "alpha", a, "| mean coefficients", c, "range over origins", lo, hi, flush=True)
d = B[(O >= "2025-06") & known].copy(); d["left"] = d.e - r0[(O >= "2025-06") & known]
print("Spearman of what v0 leaves (e - rhat) with: " + ", ".join(f"{c} {d[['left', c]].corr(method='spearman').iloc[0, 1]:+.3f}" for c in ["nc_sea4", "fa_mean_sea4", "nc_acc1", "nc_mom3", "fa_mean_mom", "fa_llm_yoy"]))
# the adopted-candidate version for 2026 (added after the reads above): v0's residual + the ridge on nc_sea4 (alpha 100), applied at every
# origin incl. 2026-06 (trained on closed windows only); writes the 2026 corrected totals beside the current ones
if os.environ.get("OUT2026") == "1" and not LYG:
    r1, c1, lo1, hi1 = lin(r0, ["nc_sea4"], 100.0); g_new, g_cur = geo(r1), geo(r0); m26 = (O == "2026-06")
    out = pd.DataFrame({"style_color": B.style_color.values[m26], "F": B.F.values[m26], "geo_current": g_cur[m26], "geo": g_new[m26], "nc_sea4": B.nc_sea4.values[m26]})
    out.to_parquet("data/geo_nx5r_fash_2026.parquet", index=False)
    print("2026-06 rows", int(m26.sum()), "| coefficient at 2026-06 fit:", c1, "| multiplier new / current 10/50/90%:", np.round(np.nanpercentile(out.geo / out.geo_current, [10, 50, 90]), 4),
          "| min / max:", round(float((out.geo / out.geo_current).min()), 4), round(float((out.geo / out.geo_current).max()), 4), "| total kept:", round(float(out.geo.sum() / out.geo_current.sum()), 6))
# within-origin version (added after the 2026 check: pooled across origins the coefficient cancels to -0.001 at 2026-06, because the
# feature's level shifts between origins; within each origin its rank agrees with the residual in 10 of 12 windows)
for c_ in ["nc_sea4", "fa_mean_sea4"]:
    B[c_ + "_wz"] = B.groupby("origin")[c_].transform(lambda s_: (s_ - s_.mean()) / (s_.std() if s_.std() > 0 else 1.0))
def linz(r0, cols, alpha=100.0):
    r = r0.copy(); co = []
    for o in ORIG:
        tr = (oo + 4 <= pd.Period(o).ordinal) & known & gate(o); m = (B.origin == o).values
        if B.origin[tr].nunique() < 3: continue
        Z = lambda d: d[cols].fillna(0).clip(-3, 3).values
        g = Ridge(alpha=alpha, fit_intercept=False).fit(Z(B[tr]), (B.e[tr].clip(-2, 2) - r0[tr]).values, sample_weight=np.sqrt(B.y[tr] + B.F[tr] + 1)); r[m] = r0[m] + g.predict(Z(B[m])); co.append((o, np.round(g.coef_, 4)))
    return r, co
for cols in (["nc_sea4_wz"], ["nc_sea4_wz", "fa_mean_sea4_wz"]):
    rz, co = linz(r0, cols); show("linz", rz); print("      within-origin ridge on", cols, "| coefficient by origin:", [(o, list(c)) for o, c in co][-6:], flush=True)
    if os.environ.get("OUT2026") == "1" and not LYG and cols == ["nc_sea4_wz"]:
        g_new, g_cur = geo(rz), geo(r0); m26 = (O == "2026-06")
        out = pd.DataFrame({"style_color": B.style_color.values[m26], "F": B.F.values[m26], "geo_current": g_cur[m26], "geo": g_new[m26], "nc_sea4": B.nc_sea4.values[m26]})
        out.to_parquet("data/geo_nx5r_fash_2026.parquet", index=False)
        bt = known   # the backtest counterpart of data/geo_nx5r_x_preds.parquet (origin x style_color x F, geo) for the ranges calibration
        pd.DataFrame({"origin": O[bt], "style_color": B.style_color.values[bt], "F": B.F.values[bt], "y": y[bt], "R3": R3[bt], "geo": g_new[bt], "geo_current": g_cur[bt]}).to_parquet("data/geo_nx5r_fash_x_preds.parquet", index=False)
        print("2026 (within-origin): multiplier new / current 10/50/90%:", np.round(np.nanpercentile(out.geo / out.geo_current, [10, 50, 90]), 4), "| min / max", round(float((out.geo / out.geo_current).min()), 4), round(float((out.geo / out.geo_current).max()), 4), flush=True)

if os.environ.get("DBG") == "1":   # size of the ridge's adjustment by origin (diagnostic)
    r1, *_ = lin(r0, ["nc_sea4"], 100.0); d = pd.DataFrame({"o": O, "dr": r1 - r0, "nc": B.nc_sea4.values})
    print(d.groupby("o").agg(dr_sd=("dr", "std"), dr_absmax=("dr", lambda s: s.abs().max()), nc_cov=("nc", lambda s: s.notna().mean()), nc_mean=("nc", "mean"), nc_sd=("nc", "std")).round(4).to_string())
    ratio = pd.Series(geo(r1) / geo(r0)); print("geo ratio sd by origin:", ratio.groupby(O).std().round(5).to_dict())
