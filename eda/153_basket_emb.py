"""D150: co-purchase (shopping-cart) item and style embeddings, time-honest, and rotation-invariant neighbour features for the trees.
At every origin o (2024-03 .. 2026-06), from online orders in the 12 months up to o only:
  1. Pair counts C_ij = orders containing both items; PPMI with context smoothing (alpha 0.75), rows L2-normalised.
  2. Attribute block: one-hot product type, macro group, primary colour, silhouette, fabric, weight, print, length, neckline, sleeve,
     rise, occasion tier, and the multi-valued tags (detail, trend, aesthetics, occasion), optionally the LLM scores (D151) as
     standardised columns; rows L2-normalised and weighted by GAMMA (co-purchase gets weight 1, so it dominates).
  3. Truncated SVD of [PPMI | GAMMA x attributes] to 32 dims; rows normalised = item vectors. Style vector = mean of its colours.
     Items without baskets yet (launches) sit in the space through their attributes.
Neighbour features per item at o (top K = 20 cosine neighbours among items that sold in the last 3 months, other styles only,
weights = cosine similarity clipped at 0); all from monthly units (all channels) up to o:
  nb_mom1  log((u[o] + 1) / (mean u[o-3..o-1] + 1))                          neighbours' 1-month momentum
  nb_mom3  log((sum u[o-2..o] + 1) / (sum u[o-5..o-3] + 1))                  3-month momentum
  nb_yoy   log((sum u[o-2..o] + 1) / (sum u[o-14..o-12] + 1)), neighbours alive a year ago
  nb_sea4  log of the neighbours' summed units in [o-11 .. o-8] / their summed units in [o-14 .. o-12] x 3/4   (last year's ramp
           from the origin window into the 4 target months, for the item's co-purchase cluster)
  nb_lvl   log1p of the neighbours' weighted mean monthly units over the last 3 months
  attach   the item's share of orders (last 3 months) that contain another item
  sty_nb_sim  mean similarity of the top-K neighbours (how embedded the item is)
Variants: cp (GAMMA 0.5), attr (attributes only: no PPMI block), cpl (cp + LLM scores in the attribute block, if present).
Writes data/basket_feats_<variant>.parquet and data/basket_emb_<variant>.npz (item vectors at each origin, for maps)."""
import sys, glob, numpy as np, pandas as pd, time
from sklearn.utils.extmath import randomized_svd
sys.stdout.reconfigure(encoding="utf-8")
VARS = sys.argv[1:] or ["cp", "attr"]
K, DIM, WIN = 20, 32, 12
items = np.sort(pd.read_parquet("data/items.parquet").style_color.values); n = len(items); sty = pd.Series(items).str.split("_").str[0].values
it = pd.read_parquet("data/items.parquet").set_index("style_color").reindex(items)
F0 = pd.read_parquet("data/features.parquet", columns=["style_color", "macro"]).drop_duplicates("style_color").set_index("style_color").macro
A = pd.read_csv("item_attributes.csv").set_index("style_color").reindex(items)
blocks = [pd.get_dummies(it.product_type.fillna("na"), prefix="pt"), pd.get_dummies(F0.reindex(items).fillna("na"), prefix="mac")]
for c in ["color_primary", "silhouette", "fabric_feel", "fabric_weight", "print_pattern", "length", "neckline", "sleeve", "rise", "occasion_tier"]:
    v = A[c].fillna("na"); top = v.value_counts(); v = v.where(v.isin(top[top >= 5].index), "rare"); blocks.append(pd.get_dummies(v, prefix=c))
for c in ["detail_tags", "trend_tags", "aesthetics", "occasion_tags"]:
    tags = A[c].fillna("").str.split(r"\s*\|\s*"); allt = pd.Series([t for ts in tags for t in ts if t]).value_counts()
    keep = allt[allt >= 5].index; blocks.append(pd.DataFrame([[int(t in set(ts)) for t in keep] for ts in tags], columns=[f"{c}_{t}" for t in keep], index=items))
ATT = pd.concat([b.reset_index(drop=True) for b in blocks], axis=1).astype(float).values
LLM = None
if glob.glob("data/llm/scores_part*.csv"):
    L = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("data/llm/scores_part*.csv"))]).drop_duplicates("style_color").set_index("style_color").reindex(items)
    LLM = ((L - L.mean()) / L.std()).fillna(0).values / 2
print("attribute columns", ATT.shape[1], "| LLM scores", None if LLM is None else LLM.shape[1], flush=True)
P = pd.read_parquet("data/basket_pairs_month.parquet"); B = pd.read_parquet("data/basket_item_month.parquet")
months = sorted(P.m.unique()); CM = {}
for m, g in P.groupby("m"):
    c = np.zeros((n, n), np.float32); c[g.it.values, g.it_b.values] = g.n.values; CM[m] = c + c.T
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; M = M.reindex(items).fillna(0)
Pm = lambda o, k: str(pd.Period(o, "M") + k)
U = lambda o, ks: M.reindex(columns=[Pm(o, k) for k in ks]).fillna(0).values      # units in months o+k


def rown(X): return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def embed(o, var):
    win = [m for m in months if Pm(o, -WIN + 1) <= m <= o]
    C = sum(CM[m] for m in win) if win else np.zeros((n, n), np.float32)
    r = C.sum(1); tot = r.sum(); blocks = []
    if var != "attr" and tot > 0:
        ctx = r ** 0.75; ctx = ctx / ctx.sum()
        with np.errstate(divide="ignore", invalid="ignore"): pmi = np.log((C / tot) / (np.outer(r / tot, ctx)))
        blocks.append(rown(np.nan_to_num(np.maximum(pmi, 0), posinf=0, neginf=0)))
    att = ATT if (var != "cpl" or LLM is None) else np.hstack([ATT, LLM])
    blocks.append((0.5 if var != "attr" else 1.0) * rown(att))
    X = np.hstack(blocks); u, s, _ = randomized_svd(X, DIM, random_state=0); V = rown(u * np.sqrt(s))
    return V, r


out = {v: [] for v in VARS}; vecs = {v: {} for v in VARS}; t0 = time.time()
ORIG = [str(p) for p in pd.period_range("2024-03", "2026-06", freq="M")]
for o in ORIG:
    u0 = U(o, [0])[:, 0]; u3 = U(o, [-3, -2, -1]); l3 = U(o, [-2, -1, 0]); p3 = U(o, [-5, -4, -3]); y3 = U(o, [-14, -13, -12]); yt = U(o, [-11, -10, -9, -8])
    alive = l3.sum(1) > 0
    mom1 = np.log((u0 + 1) / (u3.mean(1) + 1)); mom3 = np.log((l3.sum(1) + 1) / (p3.sum(1) + 1))
    yoy = np.where(y3.sum(1) > 0, np.log((l3.sum(1) + 1) / (y3.sum(1) + 1)), np.nan); lvl = l3.mean(1)
    bm = B[(B.m >= Pm(o, -2)) & (B.m <= o)].groupby("it")[["orders", "multi_orders"]].sum().reindex(range(n))
    attach = (bm.multi_orders / bm.orders).values
    for v in VARS:
        V, r = embed(o, v); vecs[v][o] = V.astype(np.float16)
        S = V @ V.T; S[:, ~alive] = -9; S[sty[:, None] == sty[None, :]] = -9
        nb = np.argsort(-S, axis=1)[:, :K]; w = np.clip(np.take_along_axis(S, nb, 1), 0, None); w[np.take_along_axis(S, nb, 1) < -1] = 0
        ws = w.sum(1); wm = lambda x: np.where(ws > 0, (w * x[nb]).sum(1) / np.maximum(ws, 1e-9), np.nan)
        yo = np.nan_to_num(yoy); wy = w * (~np.isnan(yoy))[nb]
        nb_yoy = np.where(wy.sum(1) > 0, (wy * yo[nb]).sum(1) / np.maximum(wy.sum(1), 1e-9), np.nan)
        sy3 = (w * y3.sum(1)[nb]).sum(1); syt = (w * yt.sum(1)[nb]).sum(1)
        nb_sea4 = np.where(sy3 > 0, np.log((syt + 1) / (sy3 * 4 / 3 + 1)), np.nan)
        out[v].append(pd.DataFrame({"origin": o, "style_color": items, "nb_mom1": wm(mom1), "nb_mom3": wm(mom3), "nb_yoy": nb_yoy, "nb_sea4": nb_sea4,
                                    "nb_lvl": np.log1p(wm(lvl)), "attach": attach, "nb_sim": np.where(ws > 0, ws / K, np.nan), "basket_orders": np.log1p(r)}))
    print(o, f"alive {alive.sum()} | {time.time() - t0:.0f}s", flush=True)
for v in VARS:
    pd.concat(out[v], ignore_index=True).to_parquet(f"data/basket_feats_{v}.parquet", index=False)
    np.savez_compressed(f"data/basket_emb_{v}.npz", items=items, **{o: x for o, x in vecs[v].items()})
print("saved", VARS)
