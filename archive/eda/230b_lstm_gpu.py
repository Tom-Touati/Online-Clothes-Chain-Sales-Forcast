"""Fork "fashion" (D404; user: "train an lstm model on the gpu server that take daily data from the last 2 months and tries to compute the next
month"). Runs on the GPU server (3.94.95.98, ~/edikted_pf/venv_tab, torch + CUDA) from data/lstm_inputs.npz (eda/230a_lstm_prep.py).
Sample = one item at one cutoff day c: the 60 days c-59 .. c ->  units in the next month.
  inputs per day (9): log1p(units / s) for web, TikTok, own stores, partners and their total (s = the window's mean daily units + 0.5),
                      the average web discount, day-of-week sin / cos, a live flag (day >= first sale)
  target: z = log((y + 1) / (L + 1)), y = units in the next month (training: next 30 days; scoring: the next calendar month),
          L = units in the last 30 days x (days in the target / 30); forecast = (L + 1) exp(z_hat) - 1
  versions:  daily  the daily inputs only (as asked)
             ctx    + product type (embedding 8), the target month (sin / cos), the item's age and log level (what 60 days cannot show)
Model: LSTM (2 layers, hidden 64, dropout 0.1) -> last hidden state (+ context) -> MLP -> z_hat; Huber loss (delta 0.5), sample weight
sqrt(L + 1) (the score is volume-weighted); Adam 2e-3, 5 epochs, batch 1024; 3 seeds averaged in z.
Time-honest: for scoring month-end o, training cutoffs are every 7th day with the 30-day target ended by o's last day (c + 30 <= end of o);
an item enters at a cutoff if it is live and sold >= 5 units in the 60-day window.
Env W (default 60): the window in days. D405 (user: "try the lstm with a full year of daily history"): W=365; days before the data start
(1 Jan 2024) are zero-padded with a "data available" flag (10th input); scaling s stays the last 60 days' mean so the earlier months read
relative to the current level (last year's same-season ramp included); the training-entry rule is unchanged (>= 5 units in the last 60
days, live). Versions are named daily365 / ctx365.
D406 (user: "add all the other features to the head"): env FULL=1 adds the tree member's full input set at the cutoff month-end
(data/lstm_tab_feats.parquet, 109 features from eda/230d, standardised on the training rows, missing = 0) to the head, next to the
sequence state and the context. These exist only at month-ends, so FULL trains on month-end cutoffs only (origins o' with o' + 1 <= o,
target = the next calendar month). Versions: full<W> (sequence + context + features) and tabonly (context + features, no sequence).
Head: Linear(.., 128) -> ReLU -> Dropout 0.1 -> Linear(128, 1); 25 epochs, batch 512.
D407 (user: "also try it predicting the full 4 months"): env H4=1 predicts the next four calendar months' total, the system's own target.
Training cutoffs: weekly with the next 122 days ended by o (or month-ends o' + 4 <= o with FULL); L = the last 90 days x days / 90.
Output files get the suffix _h4.
Writes out/lstm_preds.parquet (W=60) or out/lstm_preds_w<W>.parquet (origin, style_color, version, y1_hat, z_hat, L)."""
import os
import sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
dev = "cuda"; Z = np.load("data/lstm_inputs.npz", allow_pickle=True)
W = int(os.environ.get("W", 60)); H4 = os.environ.get("H4") == "1"; HM = 4 if H4 else 1; LB = 90 if H4 else 30; NDW = 122 if H4 else 30   # horizon months, L look-back days, weekly target days
PAD = max(W - 60, 0)
X = np.concatenate([np.zeros((Z["X"].shape[0], PAD, 5), np.float32), Z["X"].astype(np.float32)], 1); N, Dd, _ = X.shape; day0 = pd.Timestamp(str(Z["day0"])) - pd.Timedelta(days=PAD); DATA0 = PAD; items = Z["items"]; ptype = torch.tensor(Z["ptype"].astype(np.int64), device=dev); first = Z["first_ix"] + PAD
EVO, EVI = Z["ev_origin"], Z["ev_item"]; ii = pd.Series(np.arange(N), index=items)
tot = X[:, :, :4].sum(2); csum = np.concatenate([np.zeros((N, 1), np.float32), np.cumsum(tot, 1)], 1)
dow = np.arange(Dd) + day0.day_of_week; DOW = np.stack([np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)], 1).astype(np.float32)
Xg = torch.tensor(X, device=dev); DOWg = torch.tensor(DOW, device=dev); first_g = torch.tensor(first, device=dev)
def window_sum(r, a, b): return csum[r, b + 1] - csum[r, a]   # units on days a..b (inclusive), vectorised over rows r
def batch_inputs(rows, cuts, ndays):
    """rows, cuts: int tensors on the GPU; returns sequence (B, 60, 9), context (B, 12) parts and L"""
    idx = cuts[:, None] + torch.arange(-W + 1, 1, device=dev)[None, :]
    seq = Xg[rows[:, None], idx]                                    # (B, 60, 5)
    tot_ = seq[:, :, :4].sum(2, keepdim=True); s = tot_[:, -60:].mean(1, keepdim=True) + 0.5   # scale = the last 60 days' level
    live = (idx >= first_g[rows][:, None]).float()[:, :, None]
    avail = (idx >= DATA0).float()[:, :, None]
    x = torch.cat([torch.log1p(seq[:, :, :4] / s), torch.log1p(tot_ / s), seq[:, :, 4:5], DOWg[idx], live, avail], 2)
    L = seq[:, -LB:, :4].sum((1, 2)) * ndays / float(LB)
    tm = ((day0 + pd.to_timedelta(cuts.cpu().numpy() + 1, "D")).month.values - 1).astype(np.float32)   # first target month
    ctx = torch.stack([torch.tensor(np.sin(2 * np.pi * tm / 12), device=dev), torch.tensor(np.cos(2 * np.pi * tm / 12), device=dev),
                       torch.log1p(torch.clamp(cuts - first_g[rows], min=0).float()) / 7, torch.log1p(s[:, 0, 0])], 1)
    return x, ctx, L
class Net(nn.Module):
    def __init__(self, use_ctx, n_tab=0, use_seq=True):
        super().__init__(); self.use_ctx, self.n_tab, self.use_seq = use_ctx, n_tab, use_seq
        self.lstm = nn.LSTM(10, 64, 2, batch_first=True, dropout=0.1) if use_seq else None
        self.emb = nn.Embedding(int(ptype.max()) + 1, 8); k = (64 if use_seq else 0) + (8 + 4 if use_ctx else 0) + n_tab
        self.head = nn.Sequential(nn.Linear(k, 64), nn.ReLU(), nn.Linear(64, 1)) if n_tab == 0 else nn.Sequential(nn.Linear(k, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(self, x, ctx, pt, tab=None):
        parts = [self.lstm(x)[0][:, -1]] if self.use_seq else []
        if self.use_ctx: parts += [self.emb(pt), ctx]
        if self.n_tab: parts += [tab]
        return self.head(torch.cat(parts, 1))[:, 0]
FULL = os.environ.get("FULL") == "1"
if FULL:
    TF = pd.read_parquet("data/lstm_tab_feats.parquet"); TCOLS = [c for c in TF.columns if c not in ("origin", "style_color")]
    TKEY = {k: i for i, k in enumerate(zip(TF.origin, TF.style_color))}; TABX = TF[TCOLS].to_numpy(np.float32)
def tab_rows(origins, rows):
    """feature-table row index for (origin, item row); -1 where missing"""
    return np.array([TKEY.get((o, items[r]), -1) for o, r in zip(origins, rows)])
def month_end_ix(o): return (pd.Period(o, "M").end_time.normalize() - day0).days
out = []; t_all = time.time()
for o in sorted(set(EVO)):
    end = month_end_ix(o); te = np.where(EVO == o)[0]; te_rows = ii.reindex(EVI[te]).values; ok = ~np.isnan(te_rows); te, te_rows = te[ok], te_rows[ok].astype(int)
    nd = sum((pd.Period(o, "M") + h).days_in_month for h in range(1, HM + 1))
    if not FULL:
        cuts = np.arange(PAD + 59, end - NDW + 1, 7); R, C = np.meshgrid(np.arange(N), cuts, indexing="ij"); R, C = R.ravel(), C.ravel(); NDtr = np.full(len(R), NDW)
    else:   # month-end cutoffs whose next calendar month ended by the end of o
        ors = [str(q) for q in pd.period_range("2024-03", o, freq="M") if q + HM <= pd.Period(o, "M")]
        cuts = np.array([month_end_ix(q) for q in ors]); R, C = np.meshgrid(np.arange(N), cuts, indexing="ij"); R, C = R.ravel(), C.ravel()
        NDtr = np.array([sum((pd.Period(ors[list(cuts).index(c)], "M") + h).days_in_month for h in range(1, HM + 1)) for c in C])
    keep = (window_sum(R, C - 59, C) >= 5) & (C >= first[R]); R, C, NDtr = R[keep], C[keep], NDtr[keep]
    y = window_sum(R, C + 1, C + NDtr); Ltr = window_sum(R, C - LB + 1, C) * NDtr / float(LB)
    if FULL:
        o_of = {c: q for q, c in zip(ors, cuts)}; tri = tab_rows([o_of[c] for c in C], R); tei = tab_rows([o] * len(te_rows), te_rows)
        mu = np.nanmean(TABX[tri[tri >= 0]], 0); sd = np.nanstd(TABX[tri[tri >= 0]], 0) + 1e-6
        prep = lambda ix: torch.tensor(np.nan_to_num(np.where(ix[:, None] >= 0, (TABX[np.maximum(ix, 0)] - mu) / sd, 0.0), nan=0.0).clip(-5, 5).astype(np.float32), device=dev)
        TABtr, TABte = prep(tri), prep(tei)
    ztr = torch.tensor(np.log((y + 1) / (Ltr + 1)), device=dev, dtype=torch.float32); wtr = torch.tensor(np.sqrt(Ltr + 1), device=dev, dtype=torch.float32); wtr = wtr / wtr.mean()
    Rg, Cg = torch.tensor(R, device=dev), torch.tensor(C, device=dev)
    rows_te = torch.tensor(te_rows, device=dev); cut_te = torch.full((len(te_rows),), end, device=dev)
    VERS = ((f"full{W}", "tabonly") if FULL else (("daily", "ctx") if W == 60 else (f"daily{W}", f"ctx{W}")))
    EPOCHS, BS = (25, 512) if FULL else (5, 1024)
    for ver in VERS:
        zs = []
        for seed in range(3):
            torch.manual_seed(seed); np.random.seed(seed); net = Net(ver.startswith(("ctx", "full", "tabonly")), n_tab=(len(TCOLS) if FULL else 0), use_seq=(ver != "tabonly")).to(dev); opt = torch.optim.Adam(net.parameters(), lr=2e-3); lossf = nn.HuberLoss(delta=0.5, reduction="none")
            for ep in range(EPOCHS):
                perm = torch.randperm(len(R), device=dev); net.train()
                for b in range(0, len(R), BS):
                    j = perm[b:b + BS]; x, ctx, _ = batch_inputs(Rg[j], Cg[j], NDW)   # (L is not used in training: z targets are precomputed)
                    loss = (lossf(net(x, ctx, ptype[Rg[j]], TABtr[j] if FULL else None), ztr[j]) * wtr[j]).mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            net.eval()
            with torch.no_grad():
                x, ctx, Lte = batch_inputs(rows_te, cut_te, nd); zs.append(net(x, ctx, ptype[rows_te], TABte if FULL else None).cpu().numpy())
        zh = np.mean(zs, 0); L = Lte.cpu().numpy()
        out.append(pd.DataFrame({"origin": o, "style_color": EVI[te], "version": ver, "z_hat": zh, "L": L, "y1_hat": np.clip((L + 1) * np.exp(zh) - 1, 0, None)}))
    print(o, "| training sequences", len(R), "| scored items", len(te), f"| {time.time() - t_all:.0f} s", flush=True)
_fn = ("out/lstm_preds_full" + str(W)) if FULL else ("out/lstm_preds" if W == 60 else f"out/lstm_preds_w{W}")
pd.concat(out).to_parquet(_fn + ("_h4" if H4 else "") + ".parquet", index=False); print("saved")
