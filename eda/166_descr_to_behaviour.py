"""D160 (user request): fashion-descriptor embeddings trained to predict behaviour. The descriptors must first predict cart
co-occurrence and regional correlation between items; only then are their learned vectors appended to the observed co-occurrence
and regional vectors and used in the model.
At each origin o (data up to o only):
  observed cart vectors   PPMI of 12 months of carts (context smoothing 0.75), SVD 24 dims, normalised; S_cart = cosine
  observed region vectors each item's state mix over 12 months as log(item share / site share) (shrunk, items >= 100 units),
                          SVD 8 dims, normalised; S_reg = cosine (two items sell in the same states)
  descriptors x_i         tagger attributes (one-hot, as eda/153) + the 14 LLM scores (D151), standardised
  model                   MLP x -> 64 -> heads h_cart (16 dims) and h_reg (8 dims), unit-normalised; loss = mean over observed pairs
                          (i != j) of (h_i . h_j - S_ij)^2 for each head
TEST (gate): 5 style-grouped folds at origins 2025-06, 2025-10 and 2026-02: train on 4 folds' items, predict each held-out item's
similarity to the training items; Pearson r with the observed similarity and top-20 neighbour precision, against raw descriptor
cosine similarity (no learning) and chance.
FEATURES (every origin, model trained on all observed items): combined vector = [cart obs (else h_cart), region obs (else h_reg),
h_cart, h_reg] with block weights 0.4 / 0.2 / 0.25 / 0.15; neighbour features from its 20 nearest other-style neighbours alive at o,
as eda/153: dp_nb_sea4, dp_nb_lvl, dp_nb_mom1, dp_nb_mom3, dp_nb_yoy.
Writes eda/descr_behaviour_output.txt (via stdout) and data/descr_behaviour_feats.parquet."""
import sys, glob, time, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.utils.extmath import randomized_svd
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4)
items = np.sort(pd.read_parquet("data/items.parquet").style_color.values); n = len(items); sty = pd.Series(items).str.split("_").str[0].values
src = open("eda/153_basket_emb.py", encoding="utf-8").read(); exec(src[src.index("it = pd.read_parquet"):src.index("P = pd.read_parquet")])
Lr = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("data/llm/scores_part*.csv"))]).drop_duplicates("style_color").set_index("style_color").reindex(items)
X = np.hstack([ATT, Lr.values]); X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-6); X = X.astype(np.float32)
P = pd.read_parquet("data/basket_pairs_month.parquet"); CM = {}
for m, g in P.groupby("m"):
    c = np.zeros((n, n), np.float32); c[g.it.values, g.it_b.values] = g.n.values; CM[m] = c + c.T
S = pd.read_parquet("data/state_item_month.parquet"); states = sorted(S.state.unique()); si = {s: i for i, s in enumerate(states)}
mons = sorted(S.m.unique()); mi = {m: i for i, m in enumerate(mons)}
UST = np.zeros((len(mons), n, len(states)), np.float32); UST[S.m.map(mi).values, S.it.values, S.state.map(si).values] = S.units.values
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; M = M.reindex(items).fillna(0)
Pm = lambda o, k: str(pd.Period(o, "M") + k); U = lambda o, ks: M.reindex(columns=[Pm(o, k) for k in ks]).fillna(0).values
rown = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)


def observed(o):
    C = sum(CM[m] for m in CM if Pm(o, -11) <= m <= o); r = C.sum(1); tot = max(r.sum(), 1); ctx = r ** 0.75; ctx = ctx / max(ctx.sum(), 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"): pmi = np.nan_to_num(np.maximum(np.log((C / tot) / np.outer(r / tot, ctx)), 0), posinf=0, neginf=0)
    u, s, _ = randomized_svd(rown(pmi), 24, random_state=0); cart = rown(u * np.sqrt(s)); okc = r >= 20
    u12 = UST[[mi[m] for m in mons if Pm(o, -11) <= m <= o]].sum(0); u12 = u12[:, u12.sum(0) >= 200]; site = u12.sum(0) / max(u12.sum(), 1); tot_i = u12.sum(1, keepdims=True)
    D = np.log((u12 + 20 * site) / (tot_i + 20) / np.maximum(site, 1e-9)); D = D - (D * site).sum(1, keepdims=True)
    u, s, _ = randomized_svd(D * np.sqrt(site), 8, random_state=0); reg = rown(u * s); okr = tot_i[:, 0] >= 100
    return cart, okc, reg, okr


class Net(nn.Module):
    def __init__(s, d):
        super().__init__(); s.b = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.1)); s.hc = nn.Linear(64, 16); s.hr = nn.Linear(64, 8)
    def forward(s, x): z = s.b(x); return nn.functional.normalize(s.hc(z), dim=1), nn.functional.normalize(s.hr(z), dim=1)


def fit(tr_idx, cart, okc, reg, okr, steps=400, seed=0):
    torch.manual_seed(seed); net = Net(X.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
    xt = torch.tensor(X[tr_idx]); Sc = torch.tensor(cart[tr_idx] @ cart[tr_idx].T); Sr = torch.tensor(reg[tr_idx] @ reg[tr_idx].T)
    mc = torch.tensor(np.outer(okc[tr_idx], okc[tr_idx]) & ~np.eye(len(tr_idx), dtype=bool)); mr = torch.tensor(np.outer(okr[tr_idx], okr[tr_idx]) & ~np.eye(len(tr_idx), dtype=bool))
    net.train()
    for _ in range(steps):
        hc, hr = net(xt); loss = ((hc @ hc.T - Sc)[mc] ** 2).mean() + ((hr @ hr.T - Sr)[mr] ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): hc, hr = net(torch.tensor(X)); return hc.numpy(), hr.numpy()


def gate(o):
    cart, okc, reg, okr = observed(o); alive = U(o, range(-11, 1)).sum(1) > 0
    styles = np.unique(sty); fold = dict(zip(styles, np.random.default_rng(0).permutation(len(styles)) % 5)); f = np.array([fold[s] for s in sty])
    Xr = rown(X); res = {"cart": [], "reg": []}
    for k in range(5):
        tr = np.where(alive & (f != k))[0]; te = np.where(alive & (f == k))[0]
        hc, hr = fit(tr, cart, okc, reg, okr)
        for nm, H, Obs, ok in [("cart", hc, cart, okc), ("reg", hr, reg, okr)]:
            a = te[ok[te]]; b = tr[ok[tr]]
            if len(a) < 10: continue
            pred = H[a] @ H[b].T; raw = Xr[a] @ Xr[b].T; obs = Obs[a] @ Obs[b].T
            same = sty[a][:, None] == sty[b][None, :]; pred[same] = raw[same] = obs[same] = np.nan
            v = np.isfinite(obs)
            top = lambda Smat: np.argsort(-np.nan_to_num(Smat, nan=-9), axis=1)[:, :20]
            to = top(obs); prec = lambda Smat: np.mean([len(set(x) & set(y)) / 20 for x, y in zip(top(Smat), to)])
            res[nm].append((np.corrcoef(pred[v], obs[v])[0, 1], np.corrcoef(raw[v], obs[v])[0, 1], prec(pred), prec(raw), 20 / len(b)))
    for nm, r in res.items():
        r = np.array(r).mean(0)
        print(f"  {o} {nm:4s} held-out items: r(learned) {r[0]:.3f} | r(raw descriptors) {r[1]:.3f} | top-20 precision learned {r[2]:.3f} raw {r[3]:.3f} chance {r[4]:.3f}", flush=True)


print("== GATE: can the descriptors predict cart co-occurrence and regional correlation for items they never saw?")
for o in ["2025-06", "2025-10", "2026-02"]: gate(o)
print("\n== features at every origin", flush=True)
rows, t0 = [], time.time()
for o in [str(p) for p in pd.period_range("2024-03", "2026-06", freq="M")]:
    cart, okc, reg, okr = observed(o); alive = U(o, range(-11, 1)).sum(1) > 0
    hc, hr = fit(np.where(alive)[0] if alive.sum() > 50 else np.arange(n), cart, okc, reg, okr)
    V = np.hstack([np.sqrt(0.4) * np.where(okc[:, None], np.hstack([cart[:, :16]]), hc), np.sqrt(0.2) * np.where(okr[:, None], reg, hr), np.sqrt(0.25) * hc, np.sqrt(0.15) * hr])
    V = rown(V); live3 = U(o, [-2, -1, 0]).sum(1) > 0
    Sm = V @ V.T; Sm[:, ~live3] = -9; Sm[sty[:, None] == sty[None, :]] = -9
    nb = np.argsort(-Sm, axis=1)[:, :20]; w = np.clip(np.take_along_axis(Sm, nb, 1), 0, None)
    u0 = U(o, [0])[:, 0]; u3 = U(o, [-3, -2, -1]); l3 = U(o, [-2, -1, 0]); p3 = U(o, [-5, -4, -3]); y3 = U(o, [-14, -13, -12]); yt = U(o, [-11, -10, -9, -8])
    ws = w.sum(1); wm = lambda x: np.where(ws > 0, (w * x[nb]).sum(1) / np.maximum(ws, 1e-9), np.nan)
    yoy = np.where(y3.sum(1) > 0, np.log((l3.sum(1) + 1) / (y3.sum(1) + 1)), np.nan); wy = w * np.isfinite(yoy)[nb]
    sy3 = (w * y3.sum(1)[nb]).sum(1); syt = (w * yt.sum(1)[nb]).sum(1)
    rows.append(pd.DataFrame({"origin": o, "style_color": items, "dp_nb_sea4": np.where(sy3 > 0, np.log((syt + 1) / (sy3 * 4 / 3 + 1)), np.nan),
                              "dp_nb_lvl": np.log1p(wm(l3.mean(1))), "dp_nb_mom1": wm(np.log((u0 + 1) / (u3.mean(1) + 1))), "dp_nb_mom3": wm(np.log((l3.sum(1) + 1) / (p3.sum(1) + 1))),
                              "dp_nb_yoy": np.where(wy.sum(1) > 0, (wy * np.nan_to_num(yoy)[nb]).sum(1) / np.maximum(wy.sum(1), 1e-9), np.nan),
                              "dp_obs_cart": okc.astype(int), "dp_obs_reg": okr.astype(int)}))
    print(o, f"cart observed {okc.sum()} | region observed {okr.sum()} | {time.time() - t0:.0f}s", flush=True)
pd.concat(rows, ignore_index=True).to_parquet("data/descr_behaviour_feats.parquet", index=False); print("saved")
