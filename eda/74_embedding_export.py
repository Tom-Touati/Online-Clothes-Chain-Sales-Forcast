"""Explorer support (D102): re-run eda/70 (one NN, seed 0, fitted on all rows known at the June 2026 origin) and export
the full product-type cosine-similarity matrix plus a 2-D PCA of every item's hierarchy vector (macro+type+style+item).
Writes data/explorer/ptype_sim.csv and data/explorer/item_vec2d.csv."""
import sys, numpy as np, pandas as pd
src = open("eda/70_embedding_map.py", encoding="utf-8").read()
g = {"__name__": "__main__"}; exec(compile(src, "eda/70", "exec"), g)
S, iv, items = g["S"], g["iv"], g["items"]
S.round(4).to_csv("data/explorer/ptype_sim.csv")
X = iv - iv.mean(0); U, s, Vt = np.linalg.svd(X, full_matrices=False); P2 = X @ Vt[:2].T
pd.DataFrame({"x": P2[:, 0], "y": P2[:, 1]}, index=items.index).round(4).to_csv("data/explorer/item_vec2d.csv")
print("explained share of first two PCs:", (s[:2] ** 2 / (s ** 2).sum()).round(3))
