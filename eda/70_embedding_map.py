"""What the hierarchy embeddings learned: one NN (eda/57, inter=0, seed 0) fitted on all rows known at the June 2026 origin;
product-type vectors (macro + type) compared by cosine similarity; nearest neighbours per type and for a few target styles."""
import sys, numpy as np, pandas as pd
sys.argv = ["x", "inter=0", "seeds=1", "threads=7"]
import importlib.util; spec = importlib.util.spec_from_file_location("nnm", "eda/57_embed_nn.py"); nnm = importlib.util.module_from_spec(spec); spec.loader.exec_module(nnm)
import torch
nnm.RET_NET = True; F = nnm.F; oo = pd.Period("2026-06").ordinal
tr = F[(F.o_ord + F.h <= oo) & F.y.notna()]; te = F[F.origin == "2026-06"].head(5)
_, net = nnm.fit_predict(tr, te, 0)
items = F.drop_duplicates("style_color").set_index("style_color")
with torch.no_grad():
    xc = torch.tensor(items[[c + "_i" for c in nnm.CATS]].values).to(nnm.DEV)
    tv = (net.E["macro"](xc[:, 0]) + net.E["ptype"](xc[:, 1])).cpu().numpy()
    iv = sum(net.E[c](xc[:, nnm.CATS.index(c)]) for c in nnm.HIER).cpu().numpy()
P = pd.DataFrame(tv, index=items.ptype.astype(str).values).groupby(level=0).first()
n = items.ptype.astype(str).value_counts(); P = P[n.reindex(P.index) >= 8]
U = P.values / np.linalg.norm(P.values, axis=1, keepdims=True); S = pd.DataFrame(U @ U.T, index=P.index, columns=P.index)
print("== product types: 3 nearest by learned vector (types with >= 8 items)")
for t in S.index: print(f"{t:16s} ->", ", ".join(f"{k} {v:.2f}" for k, v in S.loc[t].drop(t).nlargest(3).items()))
sv = np.linalg.norm(iv - tv, axis=1)       # how far each item moved from its type (style + item levels)
print("\n== items whose own vectors moved furthest from their type (learned item-specific behaviour):")
x = pd.DataFrame({"ptype": items.ptype.astype(str), "dist": sv}, index=items.index).sort_values("dist", ascending=False)
print(x.head(12).round(3).to_string())
T = pd.read_csv("data/targets.csv").style_color; print("\ntargets: median distance", round(float(x.loc[x.index.isin(T), "dist"].median()), 3), "| all items", round(float(x.dist.median()), 3))
