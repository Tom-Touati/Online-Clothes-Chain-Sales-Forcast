"""D241 (user: "try PCA like hierarchical"): hierarchical PCA of the monthly sales-surprise panel.
Surprise s_im = log((u_im + 1) / (mean of the 3 previous months x the product type's seasonal step + 1)) (as D189), months 2024-07 ..
2026-06; items with >= 18 surprises; each item centred, missing months set to 0.
Layers: L0 global PCA (k = 1..5); L1 PCA within each grouping of the L0 residual, grouping = product type, or look cluster (D238 labels at
2026-02), or type x look; L2 the remaining item-specific part. Share of total variance at each layer.
Stability: loadings estimated on the first and the second half of the months; |correlation| of item loadings (global PC1-3, and the mean
of within-group PC1 per group). Predictability of factor scores: autocorrelation at lag 1, 4 and 12 months; R^2 of each global factor's
next-4-month mean from its value 12 months earlier (seasonal) and from its last 3 months (momentum).
Writes eda/hpca_output.txt (stdout), data/hpca_factors.parquet and eda/figs/hpca_factors.png."""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")
items = np.sort(pd.read_parquet("data/items.parquet").style_color.values); n = len(items)
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; M = M.reindex(items).fillna(0); months = list(M.columns)
ptype = pd.read_parquet("data/items.parquet").set_index("style_color").product_type.reindex(items).fillna("na").values
MT = M.groupby(ptype).sum(); U = M.values; first = np.argmax(U > 0, 1); first[U.sum(1) == 0] = len(months)
S = np.full(U.shape, np.nan)
for k in range(3, len(months)):
    base = U[:, k - 3:k].mean(1); live = (first <= k - 3) & (base > 0)
    step = pd.Series(MT.iloc[:, k - 12].values / np.maximum(MT.iloc[:, k - 15:k - 12].mean(1).values, 1e-9), index=MT.index).reindex(ptype).fillna(1).values if k >= 15 else np.ones(n)
    S[:, k] = np.where(live, np.log((U[:, k] + 1) / (base * step + 1)), np.nan)
cols = [k for k, m in enumerate(months) if "2024-07" <= m <= "2026-06"]; X = S[:, cols]; mlab = [months[k] for k in cols]
keep = np.isfinite(X).sum(1) >= 18; X = X[keep]; it_k = items[keep]; ty = ptype[keep]
X = X - np.nanmean(X, 1, keepdims=True); X = np.nan_to_num(X); TOT = (X ** 2).sum()
B = pd.read_parquet("data/dec_base.parquet"); look = B[B.origin == "2026-02"].set_index("style_color").look.reindex(it_k).fillna("none").values
print(f"panel: {X.shape[0]} items x {X.shape[1]} months")
U0, s0, V0 = np.linalg.svd(X, full_matrices=False)
print("L0 global: cumulative share of variance, k = 1..5:", np.round(np.cumsum(s0[:5] ** 2) / TOT, 3))
K0 = 3; R0 = X - (U0[:, :K0] * s0[:K0]) @ V0[:K0]
def within(R, grp, k=1):
    expl = 0.0; Rr = R.copy()
    for gname in np.unique(grp):
        m = grp == gname
        if m.sum() < 3: continue
        u, s, v = np.linalg.svd(R[m], full_matrices=False); expl += (s[:k] ** 2).sum(); Rr[m] = R[m] - (u[:, :k] * s[:k]) @ v[:k]
    return expl / TOT, Rr
print(f"L1 after {K0} global factors (share of total variance explained by 1 factor per group):")
for nm, grp in [("product type", ty), ("look cluster", look), ("type x look", np.char.add(np.char.add(ty.astype(str), "|"), look.astype(str)))]:
    e, Rr = within(R0, grp); print(f"  {nm:12s} {e:.3f} | groups {len(np.unique(grp))} | item-specific left {(Rr ** 2).sum() / TOT:.3f}")
# random grouping control with the same group sizes as looks
rng = np.random.default_rng(0); e_r, _ = within(R0, rng.permutation(look)); print(f"  random (look sizes) {e_r:.3f}")
# stability of loadings across halves
h = X.shape[1] // 2; A1 = np.linalg.svd(X[:, :h], full_matrices=False); A2 = np.linalg.svd(X[:, h:], full_matrices=False)
print("stability, |corr| of item loadings between halves, global PC1-3:", [round(abs(np.corrcoef(A1[0][:, j], A2[0][:, j])[0, 1]), 3) for j in range(3)])
stab = []
for gname in np.unique(look):
    m = look == gname
    if m.sum() < 5: continue
    a = np.linalg.svd(R0[m, :h], full_matrices=False)[0][:, 0]; b = np.linalg.svd(R0[m, h:], full_matrices=False)[0][:, 0]; stab.append(abs(np.corrcoef(a, b)[0, 1]))
print(f"stability of within-look PC1 loadings between halves: median |corr| {np.median(stab):.3f} over {len(stab)} looks")
# predictability of the global factor scores
Fsc = pd.DataFrame((V0[:5].T * s0[:5]), index=mlab, columns=[f"PC{j+1}" for j in range(5)])
for c in Fsc.columns[:3]:
    f = Fsc[c].values; ac = lambda L: np.corrcoef(f[L:], f[:-L])[0, 1]
    nxt = pd.Series(f).rolling(4).mean().shift(-4).values; ly = pd.Series(f).shift(11).rolling(4).mean().values; mo = pd.Series(f).rolling(3).mean().values
    ok1 = np.isfinite(nxt) & np.isfinite(ly); ok2 = np.isfinite(nxt) & np.isfinite(mo)
    r2s = np.corrcoef(nxt[ok1], ly[ok1])[0, 1] ** 2 if ok1.sum() > 4 else np.nan; r2m = np.corrcoef(nxt[ok2], mo[ok2])[0, 1] ** 2 if ok2.sum() > 4 else np.nan
    print(f"  {c}: autocorr lag1 {ac(1):+.2f} lag4 {ac(4):+.2f} lag12 {ac(12):+.2f} | R^2 next-4-month mean from last year {r2s:.2f}, from last 3 months {r2m:.2f}")
Fsc.to_parquet("data/hpca_factors.parquet")
fig, ax = plt.subplots(2, 1, figsize=(11, 6.5))
for c in Fsc.columns[:3]: ax[0].plot(range(len(mlab)), Fsc[c], marker="o", ms=3, label=c)
ax[0].set_xticks(range(0, len(mlab), 2)); ax[0].set_xticklabels(mlab[::2], rotation=45, fontsize=7); ax[0].axhline(0, color="grey", lw=0.5); ax[0].legend(); ax[0].set_title("global surprise factors (scores)")
top = pd.Series(np.abs(U0[:, 0]), index=it_k).sort_values(ascending=False).head(12).index
ld = pd.DataFrame({"PC1": U0[:, 0], "type": ty}, index=it_k).groupby("type").PC1.mean().sort_values()
ax[1].barh(ld.index[-10:].tolist() + ld.index[:10].tolist(), list(ld.values[-10:]) + list(ld.values[:10])); ax[1].set_title("mean PC1 loading by product type (10 lowest and highest)"); ax[1].tick_params(labelsize=7)
plt.tight_layout(); plt.savefig("eda/figs/hpca_factors.png", dpi=120); print("figure saved")
