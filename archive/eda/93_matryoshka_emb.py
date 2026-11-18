"""Matryoshka hierarchical attribute embeddings (D113), one model per backtest origin, CPU.
Item vector v = sum of embedding tables (D = 16):
  hierarchy: macro group, product type, style, item; deeper levels start at zero and carry a stronger L2 penalty
             (so an unseen item falls back to its style, then its type);
  attributes: the 7 tagger attributes, collection, base colour, occasion tags (mean of tag vectors; 'none' if untagged).
Matryoshka training: four heads read the nested prefixes v[:2], v[:4], v[:8], v[:16] (each with the target-month and horizon
embeddings and the numeric features), and the loss is the mean L1 over the four heads, so the leading dimensions must carry the
most useful information on their own. Target: log((y + 1) / (level + 1)) per item-month, as the monthly models.
Training rows: outcome known by the origin (o_ord + h <= origin). Test-origin head accuracy (4-month WAPE by prefix) is printed.
Writes data/mrl_emb.parquet: origin, style_color, v0..v15 and every table's own vector <table>_0..15 (prefixes are aligned:
v[:k] = the sum of the tables' first k dimensions). Usage: python eda/93_matryoshka_emb.py [seeds=1] [epochs=20]"""
import sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(8)
ARGS = dict(a.split("=") for a in sys.argv[1:]); EPOCHS = int(ARGS.get("epochs", 20)); SEEDS = int(ARGS.get("seeds", 1)); WD = float(ARGS.get("wd", 1e-5)); PS = float(ARGS.get("pen", 1)); DROP = float(ARGS.get("drop", 0))
ORIGS = ARGS["origins"].split(",") if "origins" in ARGS else [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]; OUT = ARGS.get("out", "data/mrl_emb.parquet")
D, K = 16, [2, 4, 8, 16]
F = pd.read_parquet("data/features.parquet")
it = pd.read_parquet("data/items.parquet").set_index("style_color")
F["style"] = F.style_color.str.split("_").str[0]
F["coll"] = it.product_collection.reindex(F.style_color).fillna("none").values
col = it.color.fillna("none").str.replace("-", " ").str.split().str[-1]; top = col.value_counts().head(25).index
F["colour"] = col.where(col.isin(top), "other").reindex(F.style_color).values
F["rlog"] = np.log((F.y + 1) / (F.level + 1))
F["l_age"] = np.log1p(F.age_m.clip(lower=0)); F["l_live"] = np.log1p(F.days_live)
NUM = [f"lagr{k}" for k in range(1, 7)] + ["lvl", "f_r7", "f_r28", "trend91", "peak_r", "sh_ret", "sh_drop", "sh_tt", "disc_item", "disc_rel",
                                          "lfl_type", "lfl_macro", "lfl_occ", "own_sea", "color_num", "l_age", "l_live"]
HIER = ["macro", "ptype", "style", "style_color"]; PEN = {"macro": 0.0, "ptype": 1e-5, "style": 1e-4, "style_color": 1e-3}
ATTR = [c for c in F.columns if c.startswith("a_")] + ["coll", "colour"]
TABLES = HIER + ATTR + ["occ"]
# vocabularies over all items (attributes are item master data)
VOC = {c: {v: i for i, v in enumerate(sorted(F[c].astype(str).unique()))} for c in HIER + ATTR}
tags = it.occasion.fillna("none").str.split(";").apply(lambda s: [t.strip() for t in s])
TAGV = {t: i for i, t in enumerate(sorted({t for s in tags for t in s}))}; MAXT = 3
def tag_idx(scs):
    out = np.full((len(scs), MAXT), len(TAGV)); m = np.zeros((len(scs), MAXT), np.float32)
    for r, s in enumerate(scs):
        tt = tags.get(s, ["none"])[:MAXT]
        for j, t in enumerate(tt): out[r, j] = TAGV[t]; m[r, j] = 1
    return out, m
class MRL(nn.Module):
    def __init__(self, nnum):
        super().__init__()
        self.E = nn.ModuleDict({c: nn.Embedding(len(VOC[c]), D) for c in HIER + ATTR}); self.T = nn.Embedding(len(TAGV) + 1, D)
        for c in ["style", "style_color"]: nn.init.zeros_(self.E[c].weight)
        for c in [c for c in self.E if c not in ("style", "style_color")]: nn.init.normal_(self.E[c].weight, 0, 0.1)
        nn.init.normal_(self.T.weight, 0, 0.1)
        self.mon = nn.Embedding(12, 4); self.hz = nn.Embedding(4, 2)
        self.heads = nn.ModuleList([nn.Sequential(nn.Dropout(DROP), nn.Linear(k + 6 + nnum, 32), nn.ReLU(), nn.Dropout(DROP), nn.Linear(32, 1)) for k in K])
    def vec(self, cat, tg, tm):
        parts = {c: self.E[c](cat[:, j]) for j, c in enumerate(HIER + ATTR)}
        parts["occ"] = (self.T(tg) * tm[..., None]).sum(1) / tm.sum(1, keepdim=True).clamp(min=1)
        return sum(parts.values()), parts
    def forward(self, cat, tg, tm, mon, hz, x):
        v, parts = self.vec(cat, tg, tm); c = torch.cat([self.mon(mon), self.hz(hz), x], 1)
        return [h(torch.cat([v[:, :k], c], 1)).squeeze(1) for k, h in zip(K, self.heads)], parts
def tensors(X, mu, sd):
    cat = torch.tensor(np.stack([X[c].astype(str).map(VOC[c]).values for c in HIER + ATTR], 1), dtype=torch.long)
    tg, tm = tag_idx(X.style_color.values)
    x = torch.tensor(((X[NUM] - mu) / sd).fillna(0).clip(-5, 5).values, dtype=torch.float32)
    return cat, torch.tensor(tg), torch.tensor(tm), torch.tensor(X.tmonth.values - 1), torch.tensor(X.h.values - 1), x
ITEMS = it.index.intersection(pd.Index(F.style_color.unique()))
IX = F.drop_duplicates("style_color").set_index("style_color").loc[ITEMS].reset_index()      # one row per item for the lookup
out, t0, WAPES = [], time.time(), []
for o in ORIGS:
    oo = pd.Period(o).ordinal; tr = F[(F.o_ord + F.h <= oo) & F.y.notna()]; te = F[F.origin == o]
    mu, sd = tr[NUM].mean(), tr[NUM].std().replace(0, 1)
    A = tensors(tr, mu, sd); y = torch.tensor(tr.rlog.values, dtype=torch.float32); Te = tensors(te, mu, sd)
    vecs, preds = [], []
    for seed in range(SEEDS):
        torch.manual_seed(seed); m = MRL(len(NUM)); opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=WD); n = len(y)
        for ep in range(EPOCHS):
            perm = torch.randperm(n)
            for b in range(0, n, 512):
                i = perm[b:b + 512]; P, parts = m(*[a[i] for a in A])
                loss = sum((p - y[i]).abs().mean() for p in P) / len(K) + sum(PS * PEN[c] * parts[c].pow(2).sum(1).mean() for c in HIER)
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            P, _ = m(*Te); preds.append(torch.stack(P, 1).numpy())
            ci = tensors(IX.assign(tmonth=1, h=1), mu, sd); v, parts = m.vec(ci[0], ci[1], ci[2])
            vecs.append({"v": v.numpy()} | {c: parts[c].numpy() for c in TABLES})
    pr = np.mean(preds, 0); yh = (te.level.values[:, None] + 1) * np.exp(pr) - 1
    tot = pd.DataFrame(yh, index=te.style_color.values).groupby(level=0).sum(); act = te.groupby("style_color").y.sum().reindex(tot.index)
    wape = {f"k={k}": float((tot[j] - act).abs().sum() / act.sum()) for j, k in enumerate(K)}
    E = pd.DataFrame({"origin": o, "style_color": ITEMS})
    for nm in ["v"] + TABLES:
        arr = np.mean([vv[nm] for vv in vecs], 0) if SEEDS == 1 else vecs[0][nm]   # seeds are not aligned: keep seed 0's space
        lab = "item" if nm == "style_color" else nm
        E = pd.concat([E, pd.DataFrame(arr, columns=[f"{lab}_{d}" for d in range(D)])], axis=1)
    out.append(E); WAPES.append(wape); print(o, "train rows", len(tr), "| head 4-month WAPE by prefix", {k: round(v, 3) for k, v in wape.items()}, f"| {time.time() - t0:.0f}s", flush=True)
pd.concat(out, ignore_index=True).to_parquet(OUT, index=False)
print("mean head WAPE by prefix:", {k: round(float(np.mean([w[k] for w in WAPES])), 3) for k in WAPES[0]})
