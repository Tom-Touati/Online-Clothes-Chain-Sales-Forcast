"""Fork 2 (D385+): score recommender variants in the system. Members = rules, Huber tree (dec_base_ecnsnnhub qdyn), nn, pf2c, pf2d32
(+ the variant at weight W), with the submitted correction multiplier F/F0. WAPE / R3 on recent nine, all 13, held-out 2025-10..2026-02, early four.
Usage: python eda/212_score_rc_variants.py file:col[:weight] ..."""
import sys, numpy as np, pandas as pd
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); K = D[["origin", "style_color"]]
Hb = K.merge(pd.read_parquet("data/dec_base_ecnsnnhub.parquet")[["origin", "style_color", "qdyn"]], how="left").qdyn.values
M = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; top = D[M].max(1).values; y = D.y.values; ar = np.abs(D.R3 - y).values
rec = (D.origin >= "2025-06").values; al = np.ones(len(D), bool); er = ~rec; ho = (D.origin >= "2025-10").values; corr = (D.F / D.F0).values
X = D[M].values.copy(); X[:, 1] = Hb
w = lambda f, m: round(float(np.abs(f - y)[m].sum() / ar[m].sum()), 4); b = lambda f, m: round(100 * (f[m].sum() - y[m].sum()) / y[m].sum(), 1)
def get(fn, c):
    v = K.merge(pd.read_parquet(fn)[["origin", "style_color", c]], how="left")[c].astype(float).values; return np.minimum(np.where(np.isnan(v), D.R3, v), 3 * top)
print(f"{'variant':52s} alone rec/all  bias | system rec / all / held / early(before corr)")
s = X.mean(1); print(f"{'five members (Huber tree)':52s} {'':18s} | {w(s*corr,rec)} {w(s*corr,al)} {w(s*corr,ho)} {w(s,er)}")
for a in sys.argv[1:]:
    p = a.split(":"); fn, c = p[0], p[1]; W = float(p[2]) if len(p) > 2 else 1.0; f = get(fn, c); s = (X.sum(1) + W * f) / (5 + W)
    print(f"{c + ' x' + str(W):52s} {w(f,rec)} {w(f,al)} {b(f,rec):+} | {w(s*corr,rec)} {w(s*corr,al)} {w(s*corr,ho)} {w(s,er)}")
