"""Out-of-fold Matryoshka embeddings (D113, fixing the target leakage of eda/93 embeddings as tree features).
At each origin the styles are split into FOLDS groups (fixed random split; siblings stay together, since their sales move together
and overlap in time). One eda/93 model is trained per fold on the rows of the other folds' styles, and every item gets its vector
from the model that never saw its style. The fold models' spaces are aligned to fold 0 by orthogonal Procrustes fitted on the
group-level tables they all train (macro, type, tagger attributes, collection, colour, occasion tags), one rotation per Matryoshka
block (dims 0-1, 2-3, 4-7, 8-15), so the nested-prefix property survives the alignment. The same rotation is applied to every
table's vector. Test rows use their own fold's model too, so training and test rows have the same kind of vector.
Writes data/mrl_emb_oof.parquet (same layout as data/mrl_emb.parquet). Usage: python eda/95_matryoshka_oof.py [folds=5] [eda/93 args]"""
import sys, time, numpy as np, pandas as pd, torch
src = open("eda/93_matryoshka_emb.py", encoding="utf-8").read().split("out, t0, WAPES")[0]
exec(src)
FOLDS = int(ARGS.get("folds", 5)); OUT = ARGS.get("out", "data/mrl_emb_oof.parquet")
BLOCKS = [(0, 2), (2, 4), (4, 8), (8, 16)]
styles = np.array(sorted(F["style"].unique())); rs = np.random.default_rng(0); FOLD = dict(zip(styles, rs.permutation(len(styles)) % FOLDS))
item_fold = np.array([FOLD[s.split("_")[0]] for s in IX.style_color])
GROUP_T = ["macro", "ptype"] + ATTR
def train(tr, seed=0):
    mu, sd = tr[NUM].mean(), tr[NUM].std().replace(0, 1)
    A = tensors(tr, mu, sd); y = torch.tensor(tr.rlog.values, dtype=torch.float32)
    torch.manual_seed(seed); m = MRL(len(NUM)); opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=WD); n = len(y)
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for b in range(0, n, 512):
            i = perm[b:b + 512]; P, parts = m(*[a[i] for a in A])
            loss = sum((p - y[i]).abs().mean() for p in P) / len(K) + sum(PS * PEN[c] * parts[c].pow(2).sum(1).mean() for c in HIER)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m, mu, sd
def anchors(m): return np.vstack([m.E[t].weight.detach().numpy() for t in GROUP_T] + [m.T.weight.detach().numpy()])
def rotation(Af, A0):
    R = np.zeros((D, D))
    for a, b in BLOCKS:
        U, _, Vt = np.linalg.svd(Af[:, a:b].T @ A0[:, a:b]); R[a:b, a:b] = U @ Vt
    return R
out, t0, W = [], time.time(), []
for o in ORIGS:
    oo = pd.Period(o).ordinal; tr = F[(F.o_ord + F.h <= oo) & F.y.notna()]; te = F[F.origin == o]
    trf = tr["style"].map(FOLD).values; tef = te["style"].map(FOLD).values
    V = {nm: np.zeros((len(IX), D)) for nm in ["v"] + TABLES}; A0 = None; pred = np.zeros((len(te), len(K))); fit = []
    for f in range(FOLDS):
        m, mu, sd = train(tr[trf != f])
        with torch.no_grad():
            ci = tensors(IX.assign(tmonth=1, h=1), mu, sd); v, parts = m.vec(ci[0], ci[1], ci[2])
            if (tef == f).any():
                P, _ = m(*tensors(te[tef == f], mu, sd)); pred[tef == f] = torch.stack(P, 1).numpy()
        Af = anchors(m)
        if A0 is None: A0, R = Af, np.eye(D)
        else: R = rotation(Af, A0)
        fit.append(np.linalg.norm(Af @ R - A0) / np.linalg.norm(A0))
        sel = item_fold == f
        V["v"][sel] = v.numpy()[sel] @ R
        for c in TABLES: V[c][sel] = parts[c].numpy()[sel] @ R
    yh = (te.level.values[:, None] + 1) * np.exp(pred) - 1
    tot = pd.DataFrame(yh, index=te.style_color.values).groupby(level=0).sum(); act = te.groupby("style_color").y.sum().reindex(tot.index)
    wape = {f"k={k}": float((tot[j] - act).abs().sum() / act.sum()) for j, k in enumerate(K)}; W.append(wape)
    E = pd.DataFrame({"origin": o, "style_color": IX.style_color.values})
    for nm in ["v"] + TABLES:
        lab = "item" if nm == "style_color" else nm
        E = pd.concat([E, pd.DataFrame(V[nm], columns=[f"{lab}_{d}" for d in range(D)])], axis=1)
    out.append(E)
    print(o, f"| out-of-fold head 4-month WAPE {({k: round(x, 3) for k, x in wape.items()})} | alignment residual (folds 1-4) {np.round(fit[1:], 2).tolist()} | {time.time() - t0:.0f}s", flush=True)
pd.concat(out, ignore_index=True).to_parquet(OUT, index=False)
print("mean out-of-fold head WAPE:", {k: round(float(np.mean([w[k] for w in W])), 3) for k in W[0]})
