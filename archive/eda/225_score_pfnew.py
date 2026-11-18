"""Fork 2 (overnight goal: "try to improve the new-item version ... use the gpu"): new-item versions of the TabPFN 32 member (pf2d32) from the GPU
(3.94.95.98, eda/196c_tabpfn_new.py: v2, feat=dyn, 32 estimators, test rows = items under 6 months only):
  ctrl   the stored settings (reproduces pf2d32: median relative difference 0.06%)
  dup3   new items' context rows duplicated 3x (TabPFN has no sample weights; duplication weighs them in context)
  only   context = new items' rows only
Each swapped in for pf2d32 on items under 6 months inside the new nx5 (data/dec_base_nx5r26.parquet members, rc at a quarter, the routed nx5's
correction multiplier held fixed). WAPE / R3 on new items and overall, recent / held out / focus windows, member alone, style bootstrap vs ctrl."""
import sys, os, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
B = pd.read_parquet("data/dec_base_nx5r26.parquet"); B = B[B.y.notna()].reset_index(drop=True)
G = pd.read_parquet("data/geo_nx5r_x_preds.parquet")[["origin", "style_color", "F", "geo"]].rename(columns={"F": "Fg"}); B = B.merge(G, on=["origin", "style_color"])
B["corr"] = (B.geo / B.Fg.clip(lower=1e-9)).fillna(1); K = pd.MultiIndex.from_frame(B[["origin", "style_color"]])
new = B.age_m.fillna(99).values < 6; O = B.origin.values; y, R3 = B.y.values, B.R3.values; sty = B["style"].values; rng = np.random.default_rng(0)
W = np.array([1, 1, 1, 1, 1, .25]); MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32", "rc"]
def sysf(pfd):
    X = B[MEM].clip(lower=0).values.copy(); X[:, 4] = np.where(new & np.isfinite(pfd), pfd, X[:, 4]); return (X * W).sum(1) / W.sum() * B["corr"].values
w = lambda f, m: np.abs(f - y)[m].sum() / np.abs(R3 - y)[m].sum()
def ci(a, b, m):
    d = pd.DataFrame(dict(s=sty[m], a=np.abs(a - y)[m], b=np.abs(b - y)[m], r=np.abs(R3 - y)[m])).groupby("s").sum()
    i = rng.integers(0, len(d), size=(400, len(d))); v = (d.a.values[i].sum(1) - d.b.values[i].sum(1)) / d.r.values[i].sum(1); return f"[{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}]"
V = {}
for nm in ["ctrl", "dup3", "only"]:
    f = f"data/pfnew_{nm}.parquet"
    if os.path.exists(f): V[nm] = pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).q50.reindex(K).values
S = {nm: sysf(v) for nm, v in V.items()}
SETS = [("new, recent", (O >= "2025-06") & new), ("new, held out", (O >= "2025-10") & new), ("new, Jul-Oct 25", (O == "2025-06") & new), ("new, Mar-Jun 26", (O == "2026-02") & new),
        ("all, recent", O >= "2025-06"), ("all, held out", O >= "2025-10"), ("all, all 13", O > "0")]
for nm in V:
    print(f"== {nm}: member alone on new items recent {w(np.nan_to_num(V[nm]), (O >= '2025-06') & new & np.isfinite(V[nm])):.4f}")
    for lab, m in SETS: print(f"   {lab:16s} {w(S[nm], m):.4f}" + (f"  vs ctrl {ci(S[nm], S['ctrl'], m)}" if nm != "ctrl" else ""))
