"""Neural quantile forecaster with hierarchical embeddings, backtested on the same 13 rolling origins as the LightGBM ensemble.
Item vector = macro group + product type + style (sibling colours) + item, each a learned embedding added on top of its parent
(deeper levels start at zero and get a stronger L2 penalty, so a new item falls back to its style, then its type).
Side embeddings: occasion, collection, tagger attributes. Calendar: target month and horizon embeddings, plus an
item x target-month interaction so items can learn their own seasonality. Same numeric features as LightGBM (origin-anchored).
Output: 5 quantiles (10/25/50/75/90) of log((y+1)/(level+1)) per item-month, monotone by construction, pinball loss.
Writes data/nn_backtest_preds.parquet (origin, style_color, h, q10..q90, y)."""
import sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
sys.stdout.reconfigure(encoding="utf-8")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.set_num_threads(int(dict(a.split("=") for a in sys.argv[1:]).get("threads", 8)))
ARGS = dict(a.split("=") for a in sys.argv[1:])
SEEDS = int(ARGS.get("seeds", 3)); EPOCHS = int(ARGS.get("epochs", 30)); DIM = int(ARGS.get("dim", 16)); TAG = ARGS.get("tag", "")
CAMP = int(ARGS.get("camp", 0)); ORIGS = ARGS.get("origins")
PENS = float(ARGS.get("pen", 1)); INTER = int(ARGS.get("inter", 1)); ITEM = int(ARGS.get("item", 1)); DROP = float(ARGS.get("drop", 0.1)); WT = float(ARGS.get("wt", 0)); LR = float(ARGS.get("lr", 3e-3)); WD = float(ARGS.get("wd", 1e-4))
F = pd.read_parquet("data/features.parquet")
it = pd.read_parquet("data/items.parquet").set_index("style_color")
F["style"] = F.style_color.str.split("_").str[0]
F["coll"] = it.product_collection.reindex(F.style_color).fillna("none").values
F["occ"] = it.occasion.reindex(F.style_color).fillna("none").values
F["rlog"] = np.log((F.y + 1) / (F.level + 1))
NUM = [f"lagr{k}" for k in range(1, 7)] + ["lvl", "f_r7", "f_r28", "trend91", "peak_r", "sh_ret", "sh_drop", "sh_tt", "disc_item", "disc_rel",
                                          "lfl_type", "lfl_macro", "lfl_occ", "own_sea", "color_num"]
F["l_age"] = np.log1p(F.age_m.clip(lower=0)); F["l_live"] = np.log1p(F.days_live); NUM += ["l_age", "l_live"]
if CAMP:
    # known-future site calendar: realised online discount and deep-discount days in the target month vs the level window
    d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[(~d.cancelled) & (d["class"] == "ONLINE")]; d["ny_date"] = pd.to_datetime(d.ny_date)
    s = d.groupby("ny_date").agg(q=("qty", "sum"), dw=("disc_w", "sum")); s["disc"] = s.dw / s.q
    mon = pd.DataFrame({"disc": s.dw.resample("MS").sum() / s.q.resample("MS").sum(), "deep": s.disc.ge(0.585).resample("MS").sum()})
    mon.index = mon.index.to_period("M")
    op = pd.PeriodIndex(F.origin, freq="M"); tm = op + F.h.values
    F["c_disc_t"] = mon.disc.reindex(tm).values; F["c_deep_t"] = mon.deep.reindex(tm).values / 30
    lw = lambda c: np.mean([mon[c].reindex(op - k).values for k in range(3)], axis=0)
    F["c_disc_d"] = F.c_disc_t - lw("disc"); F["c_deep_d"] = F.c_deep_t - lw("deep") / 30
    NUM += ["c_disc_t", "c_deep_t", "c_disc_d", "c_deep_d"] if CAMP == 1 else ["c_deep_t", "c_deep_d"]
RET_NET = False
ATTR = [c for c in F.columns if c.startswith("a_")]
OCC = [c for c in F.columns if c.startswith("occ_")]
NUM += OCC
CATS = ["macro", "ptype", "style", "style_color", "occ", "coll"] + ATTR
VOC = {c: {v: i + 1 for i, v in enumerate(pd.unique(F[c].astype(str)))} for c in CATS}   # 0 = unknown
for c in CATS: F[c + "_i"] = F[c].astype(str).map(VOC[c]).astype(int)
TOTAL = int(ARGS.get("total", 0)); SEED0 = int(ARGS.get("seed0", 0))
if TOTAL:
    # 4-month-total mode (as the production quantile model, archive/eda/20): one row per item x origin, target log((ytot+1)/(4 level+1)),
    # seasonal-ratio features averaged over the 4 target months; h set to 4 so the training filter o_ord + h <= origin needs the full total.
    g4 = F.groupby(["origin", "style_color"])[["lfl_type", "lfl_macro", "lfl_occ", "own_sea"]].mean()
    F = F[F.h == 1].copy(); k = pd.MultiIndex.from_arrays([F.origin, F.style_color])
    for c in ["lfl_type", "lfl_macro", "lfl_occ", "own_sea"]: F[c] = g4[c].reindex(k).values
    F["y"] = F.ytot; F["level"] = 4 * F.level; F["rlog"] = np.log((F.y + 1) / (F.level + 1)); F["h"] = 4
QS = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=DEV)
HIER = ["macro", "ptype", "style", "style_color"] if ITEM else ["macro", "ptype"]
PEN = {k: v * PENS for k, v in {"macro": 0.0, "ptype": 1e-4, "style": 1e-3, "style_color": 1e-3}.items()}
NAME = int(ARGS.get("name", 0)); TOKC = []
if NAME:
    # product-name word embeddings (mean of word vectors): words after the given name, kept if used by >= 5 items; up to 8 per item
    import re
    words = {sc: re.findall(r"[a-z]+", str(nm).lower())[1:] for sc, nm in it.product_name.items()}
    cnt = pd.Series([w for ws in words.values() for w in set(ws)]).value_counts(); TV = {w: i + 1 for i, w in enumerate(cnt[cnt >= 5].index)}
    TOKC = [f"tok_{j}" for j in range(8)]
    tok = pd.DataFrame([([TV[w] for w in words.get(sc, []) if w in TV] + [0] * 8)[:8] for sc in F.style_color.unique()], index=F.style_color.unique(), columns=TOKC)
    F[TOKC] = tok.reindex(F.style_color).values
    print("name vocabulary:", len(TV), "words; items with >= 1 word:", int((tok.tok_0 > 0).sum()), "of", len(tok), flush=True)
SEQ = int(ARGS.get("seq", 0)); SL = int(ARGS.get("sl", 26))
if SEQ:
    # last 26 weekly units before each origin's month end, as log((week x 4.35 + 1) / (level + 1)), with a live mask; a GRU encodes it
    d = pd.read_parquet("data/daily_item_chan.parquet", columns=["style_color", "ny_date", "cancelled", "qty"]); d = d[~d.cancelled]
    d["ny_date"] = pd.to_datetime(d.ny_date); D = d.groupby(["style_color", "ny_date"]).qty.sum().unstack(fill_value=0); del d
    D = D.reindex(columns=pd.date_range("2024-01-01", "2026-06-30"), fill_value=0)
    start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
    SEQX = np.zeros((len(F), SL, 2), dtype=np.float32); cd = np.cumsum(np.concatenate([np.zeros((len(D), 1)), D.values], 1), 1); di = {k: i for i, k in enumerate(D.index)}
    for o, g in F.groupby("origin"):
        e = (pd.Period(o).end_time.normalize() - pd.Timestamp("2024-01-01")).days + 1
        ends = e - 7 * np.arange(SL)[::-1]; ok = ends - 7 >= 0
        ri = np.array([di.get(sc, -1) for sc in g.style_color]); lv = np.log1p(g.level.values)
        wk = np.where(ok[None, :], cd[np.maximum(ri, 0)][:, np.clip(ends, 0, None)] - cd[np.maximum(ri, 0)][:, np.clip(ends - 7, 0, None)], 0)
        live = ((pd.Timestamp("2024-01-01") + pd.to_timedelta(ends - 7, "D")).values[None, :] >= start.reindex(g.style_color).values[:, None]) & ok[None, :] & (ri >= 0)[:, None]
        pos = F.index.get_indexer(g.index)
        SEQX[pos, :, 0] = np.where(live, np.log1p(wk * 4.35) - lv[:, None], 0); SEQX[pos, :, 1] = live
    print("weekly sequences built:", SEQX.shape, flush=True)


class Net(nn.Module):
    def __init__(s, n_num):
        super().__init__()
        s.E = nn.ModuleDict({c: nn.Embedding(len(VOC[c]) + 1, DIM) for c in CATS})
        for c in ["style", "style_color"]: nn.init.zeros_(s.E[c].weight)
        s.tm = nn.Embedding(13, DIM); s.h = nn.Embedding(5, DIM)
        if NAME: s.tok = nn.Embedding(len(TV) + 1, DIM, padding_idx=0)
        if SEQ: s.gru = nn.GRU(2, 32, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(n_num * 2 + DIM * 5 + (32 if SEQ else 0), 128), nn.GELU(), nn.Dropout(DROP), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 5))
    def forward(s, xn, xm, xc, tm, h, sq=None):
        hier = sum(s.E[c](xc[:, CATS.index(c)]) for c in HIER)
        side = sum(s.E[c](xc[:, CATS.index(c)]) for c in CATS if c not in ["macro", "ptype", "style", "style_color"])
        if NAME:
            tk = xc[:, len(CATS):]; side = side + s.tok(tk).sum(1) / (tk > 0).sum(1, keepdim=True).clamp(min=1)
        t = s.tm(tm)
        z = torch.cat([xn, xm, hier, side, t, s.h(h), hier * t * INTER] + ([s.gru(sq)[1][0]] if SEQ else []), 1)
        o = s.mlp(z)
        return torch.cat([o[:, 2:3] - torch.cumsum(nn.functional.softplus(o[:, :2]), 1).flip(1), o[:, 2:3], o[:, 2:3] + torch.cumsum(nn.functional.softplus(o[:, 3:]), 1)], 1)
    def penalty(s, xc):
        return sum(PEN[c] * s.E[c](xc[:, CATS.index(c)]).pow(2).sum(1).mean() for c in HIER if PEN[c] > 0)


def tensors(df, mu, sd):
    x = df[NUM].astype(float); m = x.notna().astype(np.float32).values
    xn = ((x - mu) / sd).fillna(0).clip(-6, 6).values.astype(np.float32)
    return tuple(t.to(DEV) for t in (torch.tensor(xn), torch.tensor(m), torch.tensor(df[[c + "_i" for c in CATS] + TOKC].values), torch.tensor(df.tmonth.values.astype(int)), torch.tensor(df.h.values.astype(int))) + ((torch.tensor(SEQX[F.index.get_indexer(df.index)]),) if SEQ else ()))


def fit_predict(tr, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = tr[NUM].astype(float).mean(), tr[NUM].astype(float).std().replace(0, 1).fillna(1)
    X = tensors(tr, mu, sd); y = torch.tensor(tr.rlog.values.astype(np.float32), device=DEV)
    w = torch.tensor(((tr.level.values + 1) ** WT / np.mean((tr.level.values + 1) ** WT)).astype(np.float32), device=DEV)
    net = Net(len(NUM)).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    n = len(y); bs = 512; steps = EPOCHS * ((n + bs - 1) // bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    net.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=DEV)
        for b in range(0, n, bs):
            ix = perm[b:b + bs]; q = net(*(t[ix] for t in X)); e = y[ix, None] - q
            loss = (torch.maximum(QS * e, (QS - 1) * e).mean(1) * w[ix]).mean() + net.penalty(X[2][ix])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad(): return (net(*tensors(te, mu, sd)).cpu().numpy(), net) if RET_NET else net(*tensors(te, mu, sd)).cpu().numpy()


if __name__ == "__main__":
    ORIGINS = ORIGS.split(",") if ORIGS else [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
    out = []
    for o in ORIGINS:
        t0 = time.time(); oo = pd.Period(o).ordinal
        tr = F[(F.o_ord + F.h <= oo) & F.y.notna()]; te = F[F.origin == o].copy()
        Q = np.mean([fit_predict(tr, te, s) for s in range(SEED0, SEED0 + SEEDS)], axis=0)
        for j, qn in enumerate(["q10", "q25", "q50", "q75", "q90"]): te[qn] = ((te.level + 1) * np.exp(Q[:, j]) - 1).clip(lower=0)
        out.append(te[["origin", "style_color", "h", "level", "q10", "q25", "q50", "q75", "q90", "y"]])
        g = te.groupby("style_color")[["q50", "y"]].sum()
        print(f"{o}: train {len(tr)} rows, {time.time() - t0:.0f}s | WAPE(P50 total) {(g.q50 - g.y).abs().sum() / g.y.sum():.3f} bias {g.q50.sum() / g.y.sum() - 1:+.3f}", flush=True)
    pd.concat(out).to_parquet(f"data/nn_backtest_preds{TAG}.parquet")
