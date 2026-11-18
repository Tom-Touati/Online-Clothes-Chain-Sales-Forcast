"""D199 (fork): the embedding gate (user rule: an embedding enters the model only if it predicts shopping-cart co-occurrence better
than a naive raw-product-feature baseline). Time-honest and future-facing: at each backtest origin o, every candidate built from data
up to o is asked to predict the co-purchase structure of the NEXT three months (months o+1 .. o+3, which no candidate saw):
  target   cosine of PPMI-SVD (24 dims) vectors from carts in o+1 .. o+3, items with >= 20 orders there and alive at o
  raw      cosine of the raw product features (tagger one-hots, product type, macro, colour; eda/153 attribute block)  <- baseline
  static   cosine of the PPMI-SVD vectors from carts in o-11 .. o
  dynamic  cosine of the dynamic co-occurrence network's vectors v(o) (eda/181)
  chronos2 / bolt   cosine of the Chronos-2 / Chronos-Bolt encoder embeddings of the item's weekly sales (eda/184 allorig)
Pairs of the same style are excluded. Metrics: Pearson r with the target over all pairs; top-20 neighbour precision; chance = 20 / n.
Also the same restricted to items with < 6 months of history at o (young items, where history-free candidates should matter).
Writes eda/emb_gate_output.txt (stdout)."""
import sys, os, glob, numpy as np, pandas as pd
from sklearn.utils.extmath import randomized_svd
sys.stdout.reconfigure(encoding="utf-8")
items = np.sort(pd.read_parquet("data/items.parquet").style_color.values); n = len(items); sty = pd.Series(items).str.split("_").str[0].values
src = open("eda/153_basket_emb.py", encoding="utf-8").read(); exec(src[src.index("it = pd.read_parquet"):src.index("P = pd.read_parquet")])
rown = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
RAW = rown(ATT)
P = pd.read_parquet("data/basket_pairs_month.parquet"); CM = {}
for m, g in P.groupby("m"):
    c = np.zeros((n, n), np.float32); c[g.it.values, g.it_b.values] = g.n.values; CM[m] = c + c.T
first = pd.to_datetime(pd.read_parquet("data/items.parquet").set_index("style_color").reindex(items).first_sale)
Pm = lambda o, k: str(pd.Period(o, "M") + k)
def ppmi_vec(ms):
    C = sum(CM[m] for m in ms if m in CM); r = C.sum(1); tot = max(r.sum(), 1); ctx = r ** 0.75; ctx = ctx / max(ctx.sum(), 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"): pmi = np.nan_to_num(np.maximum(np.log((C / tot) / np.outer(r / tot, ctx)), 0), posinf=0, neginf=0)
    u, s, _ = randomized_svd(rown(pmi), 24, random_state=0); return rown(u * np.sqrt(s)), r
def load_npz(path):
    if not os.path.exists(path): return None
    return np.load(path, allow_pickle=True)
DYN = load_npz("data/fk_cooc_dyn.npz"); C2 = load_npz("data/fk_tsfm_emb_chronos2_all.npz"); BO = load_npz("data/fk_tsfm_emb_bolt_all.npz")
def from_npz(Z, o, keyed_items=True):
    if Z is None or o not in Z.files: return None
    E = np.zeros((n, Z[o].shape[1]), np.float32)
    if keyed_items and (o + "_items") in Z.files:
        idx = pd.Index(items).get_indexer(Z[o + "_items"].astype(str)); E[idx] = Z[o].astype(np.float32)
    else: E[:] = Z[o].astype(np.float32)
    return rown(E)
res = []
for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
    tgt, rf = ppmi_vec([Pm(o, k) for k in (1, 2, 3)]); stat, rp = ppmi_vec([Pm(o, -k) for k in range(12)])
    alive = first.values <= pd.Period(o, "M").end_time
    young = alive & (first.values > (pd.Period(o, "M").end_time - pd.Timedelta(days=183)))
    cand = {"raw": RAW, "static": stat, "dynamic": from_npz(DYN, o, False), "chronos2": from_npz(C2, o), "bolt": from_npz(BO, o)}
    for grp, msk in [("all", alive & (rf >= 20)), ("young", young & (rf >= 20))]:
        idx = np.where(msk)[0]
        if len(idx) < 30: continue
        Tt = tgt[idx] @ tgt[idx].T; same = sty[idx][:, None] == sty[idx][None, :]; np.fill_diagonal(same, True); Tt[same] = np.nan
        top_t = np.argsort(-np.nan_to_num(Tt, nan=-9), 1)[:, :20]; v = np.isfinite(Tt)
        for nm, E in cand.items():
            if E is None: continue
            Sx = E[idx] @ E[idx].T; Sx[same] = np.nan
            if not np.isfinite(Sx[v]).all() or np.nanstd(Sx) == 0: continue
            r = np.corrcoef(Sx[v], Tt[v])[0, 1]; top = np.argsort(-np.nan_to_num(Sx, nan=-9), 1)[:, :20]
            prec = np.mean([len(set(a) & set(b)) / 20 for a, b in zip(top, top_t)])
            res.append(dict(origin=o, items=grp, cand=nm, n=len(idx), r=r, p20=prec, chance=20 / len(idx)))
    print(o, "done", flush=True)
R = pd.DataFrame(res)
for grp in ["all", "young"]:
    G = R[R["items"] == grp]
    if len(G): print(f"\n== {grp} items: mean over origins (n items {G.n.mean():.0f}; chance {G.chance.mean():.3f})"); print(G.groupby("cand")[["r", "p20"]].mean().round(3).sort_values("r", ascending=False).to_string())
print("\n== per origin, r, all items"); print(R[R["items"] == "all"].pivot(index="origin", columns="cand", values="r").round(3).to_string())
