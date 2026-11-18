"""Forward (time-honest) per-style embeddings (D149, user request), as features for the trees.
At EVERY origin o from 2024-04 to 2026-02 a small network is trained on the item-month rows whose outcome is known by o (o' + h <= o):
  z = MLP(numeric features, product type, macro, target month, horizon) + b_style + <E_style, U[target month]>
  target z = log((y + 1) / (level + 1)), L1 loss; b_style (scalar) and E_style (8 dims) start at zero and carry an L2 penalty, so a style
  with little history stays near 0 (a random effect); U is a learned 8-dim vector per calendar month shared by all styles.
So each style learns a constant effect and a seasonal profile relative to what its features predict (D149 diagnostic: a constant
style error does not persist across windows, the same-season one does). The style vector's coordinates rotate between origins, so the
trees get rotation-invariant summaries of what the embedding does at that origin:
  sty_c   b_style                                  (constant effect, log scale)
  sty_s4  mean over the 4 target months of <E_style, U[month]>   (seasonal effect over the item's window)
  sty_n   log1p of the style's training rows       (how much the embedding has seen; NaN features if 0)
Tree rows at origin o' get the features from the model trained at o', i.e. from outcomes known before their own target: no leakage.
Also stores the network's own 4-month forecast with and without the style terms (pred4, pred4_ns).
Writes data/style_emb_feats.parquet. Usage: python eda/149_style_emb.py [seeds=2] [epochs=20] [dim=8] [pen=1]"""
import sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(8)
A = dict(a.split("=") for a in sys.argv[1:]); SEEDS = int(A.get("seeds", 2)); EPOCHS = int(A.get("epochs", 20)); DIM = int(A.get("dim", 8)); PS = float(A.get("pen", 1))
OUT = A.get("out", "data/style_emb_feats.parquet")
F = pd.read_parquet("data/features.parquet")
F["style"] = F.style_color.str.split("_").str[0]
F["rlog"] = np.log((F.y + 1) / (F.level + 1))
F["l_age"] = np.log1p(F.age_m.clip(lower=0)); F["l_live"] = np.log1p(F.days_live)
NUM = [f"lagr{k}" for k in range(1, 7)] + ["lvl", "f_r7", "f_r28", "trend91", "peak_r", "lfl_type", "lfl_macro", "lfl_occ", "own_sea", "l_age", "l_live"]
VOC = {c: {v: i + 1 for i, v in enumerate(pd.unique(F[c].astype(str)))} for c in ["style", "ptype", "macro"]}
for c in VOC: F[c + "_i"] = F[c].astype(str).map(VOC[c]).astype(int)
NS = len(VOC["style"]) + 1


class Net(nn.Module):
    def __init__(s, n):
        super().__init__()
        s.pt = nn.Embedding(len(VOC["ptype"]) + 1, 8); s.mc = nn.Embedding(len(VOC["macro"]) + 1, 4); s.tm = nn.Embedding(13, 8); s.h = nn.Embedding(5, 4)
        s.mlp = nn.Sequential(nn.Linear(2 * n + 24, 96), nn.GELU(), nn.Dropout(0.1), nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 1))
        s.b = nn.Embedding(NS, 1); s.E = nn.Embedding(NS, DIM); s.U = nn.Embedding(13, DIM)
        nn.init.zeros_(s.b.weight); nn.init.zeros_(s.E.weight); nn.init.normal_(s.U.weight, std=0.5)
    def parts(s, xn, xm, st, pt, mc, tm, h):
        base = s.mlp(torch.cat([xn, xm, s.pt(pt), s.mc(mc), s.tm(tm), s.h(h)], 1))[:, 0]
        return base, s.b(st)[:, 0], (s.E(st) * s.U(tm)).sum(1)
    def forward(s, *a): b, c, e = s.parts(*a); return b + c + e
    def penalty(s, st): return PS * (1e-3 * s.b(st).pow(2).sum(1) + 1e-3 * s.E(st).pow(2).sum(1)).mean()


def tens(df, mu, sd):
    x = df[NUM].astype(float); m = torch.tensor(x.notna().values.astype(np.float32))
    xn = torch.tensor(((x - mu) / sd).fillna(0).clip(-6, 6).values.astype(np.float32))
    L = lambda c: torch.tensor(df[c].values.astype(np.int64))
    return xn, m, L("style_i"), L("ptype_i"), L("macro_i"), L("tmonth"), L("h")


def run(tr, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = tr[NUM].astype(float).mean(), tr[NUM].astype(float).std().replace(0, 1).fillna(1)
    X = tens(tr, mu, sd); y = torch.tensor(tr.rlog.values.astype(np.float32)); net = Net(len(NUM))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4); n = len(y); bs = 512
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS * ((n + bs - 1) // bs)); net.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for b in range(0, n, bs):
            i = perm[b:b + bs]; loss = (net(*(t[i] for t in X)) - y[i]).abs().mean() + net.penalty(X[2][i])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad(): return [p.numpy() for p in net.parts(*tens(te, mu, sd))]


rows, t0 = [], time.time()
for o in [str(p) for p in pd.period_range("2024-04", "2026-02", freq="M")]:
    oo = pd.Period(o).ordinal; tr = F[(F.o_ord + F.h <= oo) & F.y.notna()]; te = F[F.origin == o].copy()
    if len(tr) < 500: continue
    P = [run(tr, te, s) for s in range(SEEDS)]
    te["base"], te["c"], te["e"] = [np.mean([p[j] for p in P], axis=0) for j in range(3)]
    cnt = tr.groupby("style").size(); te["sty_n"] = np.log1p(cnt.reindex(te["style"]).fillna(0).values)
    te["pred"] = (te.level + 1) * np.exp(te.base + te.c + te.e) - 1; te["pred_ns"] = (te.level + 1) * np.exp(te.base) - 1
    g = te.groupby(["origin", "style_color"]).agg(sty_c=("c", "mean"), sty_s4=("e", "mean"), sty_n=("sty_n", "first"), pred4=("pred", "sum"), pred4_ns=("pred_ns", "sum"), ytot=("ytot", "first")).reset_index()
    g.loc[g.sty_n == 0, ["sty_c", "sty_s4"]] = np.nan
    rows.append(g)
    ok = g.ytot.notna(); w = lambda c: (g.loc[ok, c].clip(lower=0) - g.loc[ok, "ytot"]).abs().sum() / g.loc[ok, "ytot"].sum() if ok.any() else np.nan
    print(f"{o}: train {len(tr)} | styles seen {int((g.sty_n > 0).sum())}/{len(g)} | sd sty_c {g.sty_c.std():.3f} sty_s4 {g.sty_s4.std():.3f} | "
          f"4m WAPE with style {w('pred4'):.3f} without {w('pred4_ns'):.3f} | {time.time() - t0:.0f}s", flush=True)
pd.concat(rows, ignore_index=True).to_parquet(OUT); print("saved", OUT)
