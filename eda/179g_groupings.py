"""D250 (user: "the true challenge is to know how to decompose, the levels"): which grouping of items holds the most shared error,
at a matched number of groups? Final system F = p2c32a (data/dec_base.parquet), recent origins.
Groupings at each origin:
  macro, product type, look (D238 cart + content clusters), type x look;
  co-surprise clusters: k-means on the items' standardised monthly surprise rows (as D241/D242, the 24 months up to the origin, items
  with >= 12 surprises); items without enough history take the cluster of their nearest look peer with history (else their own look);
  co-surprise within type: the same, run separately inside each product type;
  random groups with the same sizes (control, 5 draws).
  K = 8, 20, 40, 100, 250 where the grouping allows it.
Measures: (1) oracle rescaling -- F scaled so each group's 4-month total equals the actual; WAPE / R3. (2) share of F's log residual
variance (weights sqrt(y + F)) explained by group means. Higher = more of the error is shared within groups (decomposable there).
Writes eda/groupings_output.txt (stdout)."""
import sys, numpy as np, pandas as pd
from sklearn.cluster import KMeans
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 200)
B = pd.read_parquet("data/dec_base.parquet"); B = B[B.origin >= "2025-06"].copy()
items = np.sort(pd.read_parquet("data/items.parquet").style_color.values); ix = {s: i for i, s in enumerate(items)}; n = len(items)
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; M = M.reindex(items).fillna(0); months = list(M.columns)
ptype = pd.read_parquet("data/items.parquet").set_index("style_color").product_type.reindex(items).fillna("na").values
MT = M.groupby(ptype).sum(); U = M.values; first = np.argmax(U > 0, 1); first[U.sum(1) == 0] = len(months)
S = np.full(U.shape, np.nan)
for k in range(3, len(months)):
    base = U[:, k - 3:k].mean(1); live = (first <= k - 3) & (base > 0)
    step = pd.Series(MT.iloc[:, k - 12].values / np.maximum(MT.iloc[:, k - 15:k - 12].mean(1).values, 1e-9), index=MT.index).reindex(ptype).fillna(1).values if k >= 15 else np.ones(n)
    S[:, k] = np.where(live, np.log((U[:, k] + 1) / (base * step + 1)), np.nan)
def cosurprise(o, D, K, within=None):
    ko = months.index(o); W = S[:, max(ko - 23, 3):ko + 1]; idx = np.array([ix[s] for s in D.style_color]); Wi = W[idx]; ok = np.isfinite(Wi).sum(1) >= 12
    X = np.nan_to_num(Wi - np.nanmean(Wi, 1, keepdims=True)); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    lab = np.full(len(D), -1, dtype=object); grp = D[within].values if within else np.zeros(len(D))
    for g in np.unique(grp):
        m = (grp == g) & ok; k = max(1, min(int(round(K * (grp == g).sum() / len(D))) if within else K, m.sum()))
        if m.sum() == 0: continue
        lab[m] = [f"{g}|{c}" for c in KMeans(k, n_init=4, random_state=0).fit_predict(X[m])]
    # no history: nearest look peer with a cluster (most common cluster in the item's look), else its look
    lk = D.look.values
    for i in np.where(lab == -1)[0]:
        peers = lab[(lk == lk[i]) & (lab != -1)]
        lab[i] = pd.Series(peers).mode().iloc[0] if len(peers) else f"look_{lk[i]}"
    return lab
def score(lab):
    D = B.assign(g=lab); gs = D.groupby(["origin", "g"]); fac = (gs.y.transform("sum") + 1e-9) / (gs.F.transform("sum") + 1e-9)
    w = np.sqrt(D.y + D.F + 1); gm = (D.e * w).groupby([D.origin, D.g]).transform("sum") / w.groupby([D.origin, D.g]).transform("sum")
    om = (D.e * w).groupby(D.origin).transform("sum") / w.groupby(D.origin).transform("sum")
    return (D.F * fac - D.y).abs().sum() / (D.R3 - D.y).abs().sum(), (w * (gm - om) ** 2).sum() / (w * (D.e - om) ** 2).sum(), D.groupby("origin").g.nunique().mean()
rows = []; rng = np.random.default_rng(0)
B["tl"] = B.ptype + "|" + B.look
fixed = {"macro": B.macro.values, "product type": B.ptype.values, "look": B.look.values, "type x look": B.tl.values}
def rand_like(lab):
    out = np.empty(len(B), dtype=object)
    for o in B.origin.unique():
        m = (B.origin == o).values; out[m] = [f"{o}|{v}" for v in rng.permutation(lab[m])]
    return out
for nm, lab in fixed.items():
    lab = np.array([f"{o}|{v}" for o, v in zip(B.origin, lab)], dtype=object); w, v, k = score(lab)
    rr = [score(rand_like(lab)) for _ in range(5)]
    rows.append(dict(grouping=nm, groups=round(k), oracle_wape_r3=w, resid_var_share=v, random_oracle=np.mean([r[0] for r in rr]), random_var_share=np.mean([r[1] for r in rr])))
for K in (8, 20, 40, 100, 250):
    for nm, within in [("co-surprise", None), ("co-surprise within type", "ptype")]:
        lab = np.empty(len(B), dtype=object)
        for o in B.origin.unique():
            m = (B.origin == o).values; lab[m] = [f"{o}|{x}" for x in cosurprise(o, B[m], K, within)]
        w, v, k = score(lab); rr = [score(rand_like(lab)) for _ in range(5)]
        rows.append(dict(grouping=f"{nm} K={K}", groups=round(k), oracle_wape_r3=w, resid_var_share=v, random_oracle=np.mean([r[0] for r in rr]), random_var_share=np.mean([r[1] for r in rr])))
    print("done K", K, flush=True)
T = pd.DataFrame(rows); T["oracle_gain_over_random"] = T.random_oracle - T.oracle_wape_r3; T["var_share_over_random"] = T.resid_var_share - T.random_var_share
print(f"F as is: WAPE / R3 {(B.F - B.y).abs().sum() / (B.R3 - B.y).abs().sum():.4f}")
print(T.sort_values("groups").round(4).to_string(index=False))
