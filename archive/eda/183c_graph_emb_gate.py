"""D208 (fork; user: "try a graph embedder for the styles"): an inductive graph neural network (GraphSAGE-style) for style
embeddings, judged on the same cart gate as D164/D166/D180 (eda/170, 172, 177v: held-out styles, 3 origins x 5 folds, past and
future cart PPMI, style bootstrap against the raw product features and against v6_all, the best content embedding).
Node features: the v6_all input blocks (product attributes / collection / name words / LLM scores, region, temperature, event timing,
CLIP image, mpnet text, CLIP text), each through its own small tower as in eda/170.
Graphs (no held-out item ever has a cart edge):
  content  k = 10 nearest neighbours of every item by raw image + text cosine (content is known for every item, cart-free)
  cart     for training items only: each item's 10 highest-PPMI partners among training items, past 12 months
Layers: two mean-aggregation GraphSAGE layers (self and neighbour weights), 32-d unit output. Objective as v3/v4/v6 (MSE to the
PPMI-SVD cosine + InfoNCE on cart pairs among training items). For a held-out item, the embedding comes from its own content and from
its content neighbours, whose representations the cart objective has shaped, so cart behaviour propagates to cold items.
Variants: g_content (content graph), g_both (content + cart graph), g_cart (cart graph only: held-out items are isolated, so this is
close to an MLP), v6_all (reference). Writes eda/graph_emb_gate_output.txt (stdout) and data/graph_emb_gate_results.csv."""
import os, sys
src = open("eda/177v_content_gate.py", encoding="utf-8").read(); _k = sys.argv; sys.argv = sys.argv[:1]
exec(src[:src.index("res = []; t0 = time.time()")]); sys.argv = _k
GRUN = sys.argv[1:] or ["v6_all", "g_content", "g_both", "g_cart"]
KN = 10


class GSAGE(nn.Module):
    def __init__(s, dims, d=32, hid=64):
        super().__init__()
        s.g = nn.ModuleList([nn.Sequential(nn.Linear(k, 32), nn.GELU()) for k in dims])
        s.inp = nn.Linear(32 * len(dims), hid); s.drop = nn.Dropout(0.1)
        s.s1, s.n1 = nn.Linear(hid, hid), nn.Linear(hid, hid, bias=False); s.s2, s.n2 = nn.Linear(hid, d), nn.Linear(hid, d, bias=False)
    def forward(s, xs, A):
        h = nn.functional.gelu(s.inp(s.drop(torch.cat([g(x) for g, x in zip(s.g, xs)], 1))))
        h = nn.functional.gelu(s.s1(h) + s.n1(A @ h))
        return nn.functional.normalize(s.s2(h) + s.n2(A @ h), dim=1)


def knn_adj(S, k, rows=None, cols=None):
    """Row-normalised adjacency: each row in `rows` links to its k most similar items in `cols` (other styles only)."""
    S = S.copy(); np.fill_diagonal(S, -9); S[sty[:, None] == sty[None, :]] = -9
    if cols is not None: S[:, ~cols] = -9
    A = np.zeros_like(S, dtype=np.float32); idx = np.argsort(-S, 1)[:, :k]
    for i in (np.where(rows)[0] if rows is not None else range(len(S))):
        j = idx[i][S[i, idx[i]] > -9]; A[i, j] = 1.0
    return A


CONT = rown(np.hstack([rown(IMG + 1e-6), rown(TXT + 1e-6)])); SC = CONT @ CONT.T


def train_g(blocks, tr, Ppast, okp, kind, steps=None, seed=0):
    steps = steps or STEPS; torch.manual_seed(seed); np.random.seed(seed)
    trm = np.zeros(n, bool); trm[tr] = True
    A = np.zeros((n, n), np.float32)
    if kind in ("content", "both"): A += knn_adj(SC, KN)
    if kind in ("cart", "both"): A += knn_adj(Ppast, KN, rows=trm, cols=trm)
    A = A / np.maximum(A.sum(1, keepdims=True), 1); At = torch.tensor(A)
    xs = [torch.tensor(b, dtype=torch.float32) for b in blocks]; net = GSAGE([b.shape[1] for b in blocks])
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
    Pt = Ppast[np.ix_(tr, tr)]; ok = np.outer(okp[tr], okp[tr]) & (sty[tr][:, None] != sty[tr][None, :])
    us, ss, _ = randomized_svd(rown(Ppast), 24, random_state=0); SV = rown(us * np.sqrt(ss))[tr]; Scos = torch.tensor(SV @ SV.T, dtype=torch.float32)
    okt = torch.tensor(ok); trt = torch.tensor(tr)
    rws = np.where(ok.any(1))[0]; prob = np.where(ok, Pt, 0); prob = prob / np.maximum(prob.sum(1, keepdims=True), 1e-9)
    for _ in range(steps):
        net.train(); h = net(xs, At)[trt]; sim = h @ h.T
        loss = ((sim - Scos)[okt] ** 2).mean()
        a = np.random.choice(rws, min(256, len(rws)), replace=False); pos = np.array([np.random.choice(len(tr), p=prob[i]) for i in a])
        logits = sim[a] / 0.1; mask = torch.tensor(~ok[a] | (np.arange(len(tr))[None, :] == a[:, None]))
        logits = logits.masked_fill(mask, -1e4); logits[torch.arange(len(a)), torch.tensor(pos)] = (sim[a, torch.tensor(pos)] / 0.1)
        loss = loss + nn.functional.cross_entropy(logits, torch.tensor(pos))
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): return net(xs, At).numpy()


BLK = ["prod", "reg", "temp4", "evt4", "img", "txt", "ctx"]
res = []; t0 = time.time()
for o in ORIG:
    REGION, TEMP, EVENTS = inputs(o); TEMP2, EVT2 = inputs2(o); TR = truth(o)
    B = {"prod": np.hstack([ATT, COLL, NAME, LLM]), "reg": REGION, "temp4": np.hstack([TEMP, TEMP2]), "evt4": np.hstack([EVENTS, EVT2]), "img": IMG, "txt": TXT, "ctx": CTX}
    Hs0 = {"raw product": rown(zs(PROD_RAW)), "raw image+text": CONT}
    alive = M.reindex(columns=[Pm(o, -k) for k in range(12)]).fillna(0).values.sum(1) > 0
    styles = np.unique(sty); fold = dict(zip(styles, np.random.default_rng(0).permutation(len(styles)) % 5)); f = np.array([fold[s] for s in sty])
    for k in range(NF):
        tr = np.where(alive & (f != k))[0]; te = np.where(alive & (f == k))[0]; Hs = dict(Hs0)
        for v in GRUN:
            Hs[v] = train([B[b] for b in BLK], tr, *TR["past"], "nce") if v == "v6_all" else train_g([B[b] for b in BLK], tr, *TR["past"], v[2:])
        for tgt, (P_, ok_) in TR.items():
            for nm_, H in Hs.items():
                d = score(H, te, tr, P_, ok_); d["model"] = nm_; d["target"] = tgt; d["origin"] = o; d["fold"] = k; res.append(d)
        print(o, "fold", k, f"{time.time() - t0:.0f}s", flush=True)
R = pd.concat(res); R.to_csv("data/graph_emb_gate_results.csv", index=False)
print("\n== mean over held-out items (3 origins x 5 folds): Spearman with cart PPMI | top-20 precision | chance")
print(R.groupby(["target", "model"])[["rho", "prec", "chance"]].mean().round(4).to_string())
rng = np.random.default_rng(0)
for ref in ["raw product", "v6_all"]:
    print(f"\n== paired bootstrap over held-out styles, model minus {ref} (95% interval)")
    for tgt in ["past", "future"]:
        base = R[(R.target == tgt) & (R.model == ref)].set_index(["origin", "style"])[["rho", "prec"]].groupby(level=[0, 1]).mean()
        for m in [x for x in R.model.unique() if x != ref]:
            x = R[(R.target == tgt) & (R.model == m)].set_index(["origin", "style"])[["rho", "prec"]].groupby(level=[0, 1]).mean()
            dlt = (x - base.reindex(x.index)).dropna(); ix = np.arange(len(dlt))
            bs = np.array([dlt.values[rng.integers(0, len(ix), len(ix))].mean(0) for _ in range(1000)]); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            print(f"  {tgt:6s} {m:16s} rho {dlt.rho.mean():+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}] | prec {dlt.prec.mean():+.4f} [{lo[1]:+.4f}, {hi[1]:+.4f}]")
