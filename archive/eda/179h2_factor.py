"""D242: factor-model ("PCA-like hierarchical") forecast correction of the final system F (p2c32a; data/dec_base.parquet).
At each origin o: surprise panel (as D241) over the 24 months up to o, items with >= 12 surprises; SVD -> item loadings (global k = 3) and
monthly factor scores; within each look cluster (D238 labels at o) the PC1 of the global residual -> look loading and look factor.
Projected item surprise for the 4 target months:
  phi_seas  sum over target months and factors of loading x the factor's value 12 months earlier (the factor's seasonal repeat; D241:
            PC1's next-4-month mean is R^2 0.81 predictable from last year)
  phi_mom   4 x sum over factors of loading x the factor's mean over the last 3 months (momentum; PC3 R^2 0.51)
  phi_look  the same seasonal + momentum projection with the look-level factor
Diagnostics: Spearman of each phi with F's log residual e (recent, items with y >= 20), raw and partial on own_sea.
Correction: e ~ c0 + c1 phi_seas + c2 phi_mom + c3 phi_look fitted on earlier origins only (o' + 4 <= o; F x exp(fit)), full and half;
and momentum only without intercept (F_mom, F_mom_h).
Standard table and bootstrap against F. Writes eda/factor_output.txt (stdout)."""
import sys, contextlib, io, importlib.util, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
B = pd.read_parquet("data/dec_base.parquet")
items = np.sort(pd.read_parquet("data/items.parquet").style_color.values); n = len(items); ix = {s: i for i, s in enumerate(items)}
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; M = M.reindex(items).fillna(0); months = list(M.columns)
ptype = pd.read_parquet("data/items.parquet").set_index("style_color").product_type.reindex(items).fillna("na").values
MT = M.groupby(ptype).sum(); U = M.values; first = np.argmax(U > 0, 1); first[U.sum(1) == 0] = len(months)
S = np.full(U.shape, np.nan)
for k in range(3, len(months)):
    base = U[:, k - 3:k].mean(1); live = (first <= k - 3) & (base > 0)
    step = pd.Series(MT.iloc[:, k - 12].values / np.maximum(MT.iloc[:, k - 15:k - 12].mean(1).values, 1e-9), index=MT.index).reindex(ptype).fillna(1).values if k >= 15 else np.ones(n)
    S[:, k] = np.where(live, np.log((U[:, k] + 1) / (base * step + 1)), np.nan)
rows = []
for o in sorted(B.origin.unique()):
    ko = months.index(o); lo = max(ko - 23, 3); W = S[:, lo:ko + 1]; ok = np.isfinite(W).sum(1) >= 12
    if ok.sum() < 30 or W.shape[1] < 12:
        rows.append(pd.DataFrame({"origin": o, "style_color": items, "phi_seas": np.nan, "phi_mom": np.nan, "phi_look": np.nan})); continue
    X = W[ok] - np.nanmean(W[ok], 1, keepdims=True); X = np.nan_to_num(X)
    u, s, vt = np.linalg.svd(X, full_matrices=False); K = 3; L = u[:, :K] * s[:K] / np.sqrt(X.shape[1]); Fs = vt[:K].T * np.sqrt(X.shape[1])   # months x K
    nm = W.shape[1]; seas = np.zeros(K)
    for h in range(1, 5):
        j = nm - 1 + h - 12                                                  # same month last year, within the window
        if 0 <= j < nm: seas += Fs[j]
    mom = Fs[-3:].mean(0) * 4
    phi_s = np.full(n, np.nan); phi_m = np.full(n, np.nan); phi_l = np.full(n, np.nan)
    idx = np.where(ok)[0]; phi_s[idx] = L @ seas; phi_m[idx] = L @ mom
    R = X - (u[:, :K] * s[:K]) @ vt[:K]; lk = B[B.origin == o].set_index("style_color").look.reindex(items[idx]).fillna("none").values
    for g in np.unique(lk):
        m = lk == g
        if m.sum() < 4: continue
        uu, ss, vv = np.linalg.svd(R[m], full_matrices=False); lo_ = uu[:, 0] * ss[0] / np.sqrt(nm); fo = vv[0] * np.sqrt(nm)
        sl_ = sum(fo[nm - 1 + h - 12] for h in range(1, 5) if 0 <= nm - 1 + h - 12 < nm); phi_l[idx[m]] = lo_ * (sl_ + fo[-3:].mean() * 4) / 2
    rows.append(pd.DataFrame({"origin": o, "style_color": items, "phi_seas": phi_s, "phi_mom": phi_m, "phi_look": phi_l}))
PH = pd.concat(rows); B = B.merge(PH, on=["origin", "style_color"], how="left")
REC = B[(B.origin >= "2025-06") & (B.y >= 20)]
print("== diagnostics: Spearman with the final system's log residual (recent, y >= 20) | partial on own_sea | coverage")
for c in ["phi_seas", "phi_mom", "phi_look"]:
    d = REC[np.isfinite(REC[c]) & np.isfinite(REC.own_sea)]; z = d.own_sea.values; rz = lambda v: v - np.polyval(np.polyfit(z, v, 1), z)
    print(f"  {c:9s} {spearmanr(REC[c], REC.e, nan_policy='omit')[0]:+.3f} | {spearmanr(rz(d[c].values), rz(d.e.values))[0]:+.3f} | {np.isfinite(REC[c]).mean():.2f}")
for c in ["phi_seas", "phi_mom", "phi_look"]: B[c] = B[c].fillna(0)
oo = pd.PeriodIndex(B.origin, freq="M").astype(int); B["F_fac"] = B.F; B["F_fac_h"] = B.F; B["F_mom"] = B.F; B["F_mom_h"] = B.F
for o in sorted(B.origin.unique()):
    tr = (oo + 4 <= pd.Period(o).ordinal) & (B.y >= 1)
    if B.origin[tr].nunique() < 3: continue
    A = np.column_stack([np.ones(tr.sum()), B.phi_seas[tr], B.phi_mom[tr], B.phi_look[tr]]); w = np.sqrt(B.y[tr] + B.F[tr] + 1)
    c = np.linalg.lstsq(A * w.values[:, None], B.e[tr].values * w.values, rcond=None)[0]; m = (B.origin == o).values
    adj = c[0] + c[1] * B.phi_seas[m] + c[2] * B.phi_mom[m] + c[3] * B.phi_look[m]; B.loc[m, "F_fac"] = B.F[m] * np.exp(adj); B.loc[m, "F_fac_h"] = B.F[m] * np.exp(0.5 * adj)
    cm = np.linalg.lstsq((B.phi_mom[tr].values * w.values)[:, None], B.e[tr].values * w.values, rcond=None)[0][0]   # momentum only, no intercept
    B.loc[m, "F_mom"] = B.F[m] * np.exp(cm * B.phi_mom[m]); B.loc[m, "F_mom_h"] = B.F[m] * np.exp(0.5 * cm * B.phi_mom[m])
    print("  origin", o, "coefficients", np.round(c, 3), "| momentum-only slope", round(cm, 3))
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); Bi = B.set_index(["origin", "style_color"]).reindex(key)
for c in ["F", "F_fac", "F_fac_h", "F_mom", "F_mom_h"]: T[c] = Bi[c].values
REC_ = [o for o in ORIGINS if o >= "2025-06"]
for nm, m in [("nine recent origins", T.origin.isin(REC_)), ("all 13 origins", np.ones(len(T), bool))]:
    print(f"\n######## {nm}"); print(sl.table(T[m], ["F", "F_fac", "F_fac_h", "F_mom", "F_mom_h"], D).round(4).to_string())
F_ = T[T.origin.isin(REC_)]; print("bootstrap vs F, recent:", {c: np.round(sl.boot(F_, c, "F"), 4).tolist() for c in ["F_fac", "F_fac_h", "F_mom", "F_mom_h"]})
