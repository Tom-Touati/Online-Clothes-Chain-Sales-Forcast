"""D164 (user goal): a style/fashion embedding that predicts shopping-cart co-occurrence significantly better than the raw product
features. Inputs (all knowable at the origin; region / temperature / event inputs come from the item's own sales, never from carts):
  product   tagger attributes one-hot (type, macro, colour, silhouette, fabric, weight, print, length, neckline, sleeve, rise,
            occasion tier, detail / trend / aesthetic / occasion tags), product collection, product-name words (>= 5 items), LLM scores
  region    the item's state mix over 12 months (log share / site share, shrunk), SVD 8 dims; warm- and cold-state skew
  temp      temperature preference across states (beta_T, D157) and the mean normal temperature where it sells
  events    for each of 16 US events, the item's sales lift in months with that event (24 months up to the origin); its calendar
            profile (12 log shares); days from first sale to the next occurrence of each event; age
EVALUATION: origins 2025-06, 2025-10, 2026-02; 5 style-grouped folds (held-out styles never train). Truth = PPMI of cart pairs among
items with >= 20 multi-item cart appearances, in (a) the 12 months up to the origin and (b) the next 3 months (future carts).
For each held-out item, against every training item of another style: Spearman between predicted similarity and PPMI, and top-20
precision (overlap with the 20 highest-PPMI partners). Paired bootstrap over held-out styles of (model - raw product features).
BASELINES: raw product features cosine; raw everything (product + region + temp + events, standardised) cosine; chance.
MODELS (argv picks; default all): v1 = D160 (attributes + LLM, MSE to PPMI-SVD cosine); v2 = all inputs, same objective;
v3 = all inputs, InfoNCE on cart pairs (positives sampled by PPMI) with an MSE term; ablations of v3 without region / temp / events.
Writes eda/style_emb_gate_output.txt (stdout) and data/style_emb_gate_results.csv."""
import sys, glob, time, re, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.utils.extmath import randomized_svd
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(6)
import os
ORIG = os.environ.get("GATE_ORIGINS", "2025-06,2025-10,2026-02").split(","); NF = int(os.environ.get("GATE_FOLDS", 5)); STEPS = int(os.environ.get("GATE_STEPS", 500))
RUN = sys.argv[1:] or ["v1", "v2", "v3", "v3_noreg", "v3_notemp", "v3_noevt", "v3_prodonly"]
items = np.sort(pd.read_parquet("data/items.parquet").style_color.values); n = len(items); sty = pd.Series(items).str.split("_").str[0].values
src = open("eda/153_basket_emb.py", encoding="utf-8").read(); exec(src[src.index("it = pd.read_parquet"):src.index("P = pd.read_parquet")])
coll = it.product_collection.fillna("na"); top = coll.value_counts(); coll = coll.where(coll.isin(top[top >= 5].index), "rare")
COLL = pd.get_dummies(coll).values.astype(float)
words = [re.findall(r"[a-z]+", str(s).lower())[1:] for s in it.product_name]   # drop the first word (the given name)
wc = pd.Series([w for ws in words for w in set(ws)]).value_counts(); vocab = list(wc[wc >= 5].index)
NAME = np.array([[int(v in set(ws)) for v in vocab] for ws in words], float)
Lr = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("data/llm/scores_part*.csv"))]).drop_duplicates("style_color").set_index("style_color").reindex(items)
LLM = ((Lr - Lr.mean()) / Lr.std()).fillna(0).values
PROD_RAW = np.hstack([ATT, COLL])                                      # the naive baseline: raw product features
print("attribute cols", ATT.shape[1], "| collections", COLL.shape[1], "| name words", NAME.shape[1], flush=True)
P = pd.read_parquet("data/basket_pairs_month.parquet"); CM = {}
for m, g in P.groupby("m"):
    c = np.zeros((n, n), np.float32); c[g.it.values, g.it_b.values] = g.n.values; CM[m] = c + c.T
S = pd.read_parquet("data/state_item_month.parquet"); states = sorted(S.state.unique()); si = {s: i for i, s in enumerate(states)}
mons = sorted(S.m.unique()); mi = {m: i for i, m in enumerate(mons)}
UST = np.zeros((len(mons), n, len(states)), np.float32); UST[S.m.map(mi).values, S.it.values, S.state.map(si).values] = S.units.values
W = pd.read_parquet("data/weather_state_month.parquet"); TNORM = W.groupby("state").tmean_norm.mean().reindex(states).fillna(W.tmean_norm.mean()).values
WF = pd.read_parquet("data/weather_event_feats.parquet")
RF = pd.read_parquet("data/region_feats.parquet")
EV = pd.read_parquet("data/events_month.parquet").set_index("m"); EVN = list(EV.columns)
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; M = M.reindex(items).fillna(0)
first = pd.to_datetime(it.first_sale).values
Pm = lambda o, k: str(pd.Period(o, "M") + k)
rown = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
zs = lambda A: (A - np.nanmean(A, 0)) / np.maximum(np.nanstd(A, 0), 1e-6)


def ppmi(C):
    r = C.sum(1); tot = max(r.sum(), 1); ctx = r ** 0.75; ctx = ctx / max(ctx.sum(), 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"): p = np.log((C / tot) / np.outer(r / tot, ctx))
    return np.nan_to_num(np.maximum(p, 0), posinf=0, neginf=0), r


def inputs(o):
    """Region, temperature and event inputs at origin o, from sales up to o."""
    w12 = [mi[m] for m in mons if Pm(o, -11) <= m <= o]; u12 = UST[w12].sum(0); keep = u12.sum(0) >= 200; u = u12[:, keep]
    site = u.sum(0) / max(u.sum(), 1); tot = u.sum(1, keepdims=True)
    D = np.log((u + 20 * site) / (tot + 20) / site); D = D - (D * site).sum(1, keepdims=True)
    us, ss, _ = randomized_svd(D * np.sqrt(site), 8, random_state=0); REG = us * ss
    mix = (u12 + 20 * (u12.sum(0) / max(u12.sum(), 1))) / (u12.sum(1, keepdims=True) + 20); TMIX = mix @ TNORM
    wf = WF[WF.origin == o].set_index("style_color").reindex(items); rf = RF[RF.origin == o].set_index("style_color").reindex(items)
    REGION = np.column_stack([REG, rf.clim_warm_rel.values, rf.clim_cold_rel.values])
    TEMP = np.column_stack([wf.beta_T.values, TMIX])
    ms = [Pm(o, -k) for k in range(23, -1, -1)]; U = M.reindex(columns=ms).fillna(0).values; live = np.cumsum(U, 1) > 0
    ub = np.where(live, U, np.nan); mean_u = np.nanmean(ub, 1, keepdims=True)
    E = EV.reindex(ms).fillna(0).values                                  # months x events
    aff = np.log((np.nan_to_num(ub) @ E + 1) / (np.nan_to_num(mean_u) * (live.astype(float) @ E) + 1))
    cal = np.zeros((n, 12)); cm = np.array([int(m[-2:]) - 1 for m in ms])
    for k in range(12): cal[:, k] = np.nansum(np.where(cm[None, :] == k, ub, np.nan), 1)
    cal = np.log((cal + 1) / (cal.sum(1, keepdims=True) / 12 + 1))
    od = pd.Period(o, "M").to_timestamp(how="end"); fs = pd.to_datetime(first)
    age = np.array([(od - f).days if pd.notna(f) else np.nan for f in fs]) / 365
    nxt = []
    for e in EVN:   # days from first sale to the next start of the event's month window
        em = [pd.Period(m, "M").to_timestamp() for m in EV.index[EV[e] > 0]]
        nxt.append([min([(x - f).days for x in em if x >= f], default=np.nan) / 365 if pd.notna(f) else np.nan for f in fs])
    EVENTS = np.column_stack([aff, cal, np.array(nxt).T, age])
    return np.nan_to_num(zs(REGION)), np.nan_to_num(zs(TEMP)), np.nan_to_num(zs(EVENTS))


def truth(o):
    past, r_p = ppmi(sum(CM[m] for m in CM if Pm(o, -11) <= m <= o))
    fut, r_f = ppmi(sum(CM[m] for m in CM if Pm(o, 1) <= m <= Pm(o, 3)))
    return {"past": (past, r_p >= 20), "future": (fut, r_f >= 20)}


class Tower(nn.Module):
    def __init__(s, dims, d=32):
        super().__init__()
        s.g = nn.ModuleList([nn.Sequential(nn.Linear(k, 32), nn.GELU()) for k in dims])
        s.out = nn.Sequential(nn.Dropout(0.1), nn.Linear(32 * len(dims), 64), nn.GELU(), nn.Linear(64, d))
    def forward(s, xs): return nn.functional.normalize(s.out(torch.cat([g(x) for g, x in zip(s.g, xs)], 1)), dim=1)


def train(blocks, tr, Ppast, okp, objective, steps=None, seed=0):
    steps = steps or STEPS
    torch.manual_seed(seed); np.random.seed(seed)
    xs = [torch.tensor(b[tr], dtype=torch.float32) for b in blocks]; net = Tower([b.shape[1] for b in blocks])
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
    Pt = Ppast[np.ix_(tr, tr)]; ok = np.outer(okp[tr], okp[tr]) & (sty[tr][:, None] != sty[tr][None, :])
    us, ss, _ = randomized_svd(rown(Ppast), 24, random_state=0); SV = rown(us * np.sqrt(ss))[tr]; Scos = torch.tensor(SV @ SV.T, dtype=torch.float32)
    okt = torch.tensor(ok); Ptt = torch.tensor(Pt, dtype=torch.float32)
    rows = np.where(ok.any(1))[0]; prob = np.where(ok, Pt, 0); prob = prob / np.maximum(prob.sum(1, keepdims=True), 1e-9)
    for _ in range(steps):
        net.train(); h = net(xs); sim = h @ h.T
        loss = ((sim - Scos)[okt] ** 2).mean()
        if objective == "nce":
            a = np.random.choice(rows, min(256, len(rows)), replace=False); pos = np.array([np.random.choice(len(tr), p=prob[i]) for i in a])
            logits = sim[a] / 0.1; mask = torch.tensor(~ok[a] | (np.arange(len(tr))[None, :] == a[:, None]))
            logits = logits.masked_fill(mask, -1e4); logits[torch.arange(len(a)), torch.tensor(pos)] = (sim[a, torch.tensor(pos)] / 0.1)
            loss = loss + nn.functional.cross_entropy(logits, torch.tensor(pos))
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): return net([torch.tensor(b, dtype=torch.float32) for b in blocks]).numpy()


def score(H, te, tr, P, ok):
    """Per held-out item: Spearman of predicted similarity with PPMI over valid training partners, and top-20 precision."""
    out = []
    for i in te:
        if not ok[i]: continue
        j = tr[ok[tr] & (sty[tr] != sty[i])]
        if len(j) < 50 or P[i, j].max() <= 0: continue
        pr = H[i] @ H[j].T; t = P[i, j]
        top_t = set(j[np.argsort(-t)[:20]]); top_p = set(j[np.argsort(-pr)[:20]])
        out.append((sty[i], spearmanr(pr, t)[0], len(top_t & top_p) / 20, 20 / len(j)))
    return pd.DataFrame(out, columns=["style", "rho", "prec", "chance"])


CFG = {"v1": ("mse", ["prodllm"]), "v2": ("mse", ["prod", "reg", "temp", "evt"]), "v3": ("nce", ["prod", "reg", "temp", "evt"]),
       "v3_noreg": ("nce", ["prod", "temp", "evt"]), "v3_notemp": ("nce", ["prod", "reg", "evt"]), "v3_noevt": ("nce", ["prod", "reg", "temp"]),
       "v3_prodonly": ("nce", ["prod"])}
res = []; t0 = time.time()
for o in ORIG:
    REGION, TEMP, EVENTS = inputs(o); TR = truth(o)
    B = {"prodllm": np.hstack([ATT, LLM]), "prod": np.hstack([ATT, COLL, NAME, LLM]), "reg": REGION, "temp": TEMP, "evt": EVENTS}
    raw_prod = rown(zs(PROD_RAW)); raw_all = rown(np.hstack([zs(PROD_RAW), LLM, REGION, TEMP, EVENTS]))
    alive = M.reindex(columns=[Pm(o, -k) for k in range(12)]).fillna(0).values.sum(1) > 0
    styles = np.unique(sty); fold = dict(zip(styles, np.random.default_rng(0).permutation(len(styles)) % 5)); f = np.array([fold[s] for s in sty])
    for k in range(NF):
        tr = np.where(alive & (f != k))[0]; te = np.where(alive & (f == k))[0]
        Hs = {"raw product": raw_prod, "raw everything": raw_all}
        for v in RUN:
            obj, blk = CFG[v]; Hs[v] = train([B[b] for b in blk], tr, *TR["past"], obj)
        for tgt, (P_, ok_) in TR.items():
            for nm, H in Hs.items():
                d = score(H, te, tr, P_, ok_); d["model"] = nm; d["target"] = tgt; d["origin"] = o; d["fold"] = k; res.append(d)
        print(o, "fold", k, f"{time.time() - t0:.0f}s", flush=True)
R = pd.concat(res); R.to_csv("data/style_emb_gate_results.csv", index=False)
print("\n== mean over held-out items (3 origins x 5 folds): Spearman with cart PPMI | top-20 precision | chance")
print(R.groupby(["target", "model"])[["rho", "prec", "chance"]].mean().round(4).to_string())
rng = np.random.default_rng(0)
print("\n== paired bootstrap over held-out styles, model minus raw product features (95% interval)")
for tgt in ["past", "future"]:
    base = R[(R.target == tgt) & (R.model == "raw product")].set_index(["origin", "style"])[["rho", "prec"]].groupby(level=[0, 1]).mean()
    for m in [x for x in R.model.unique() if x != "raw product"]:
        x = R[(R.target == tgt) & (R.model == m)].set_index(["origin", "style"])[["rho", "prec"]].groupby(level=[0, 1]).mean()
        dlt = (x - base.reindex(x.index)).dropna(); ix = np.arange(len(dlt))
        bs = np.array([dlt.values[rng.integers(0, len(ix), len(ix))].mean(0) for _ in range(1000)])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        print(f"  {tgt:6s} {m:16s} rho {dlt.rho.mean():+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}] | prec {dlt.prec.mean():+.4f} [{lo[1]:+.4f}, {hi[1]:+.4f}]")
