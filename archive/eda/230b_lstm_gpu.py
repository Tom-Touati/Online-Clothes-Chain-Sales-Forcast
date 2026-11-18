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
Writes out/lstm_preds.parquet (origin, style_color, version, y1_hat, z_hat, L)."""
import sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
dev = "cuda"; Z = np.load("data/lstm_inputs.npz", allow_pickle=True)
X = Z["X"].astype(np.float32); N, Dd, _ = X.shape; day0 = pd.Timestamp(str(Z["day0"])); items = Z["items"]; ptype = torch.tensor(Z["ptype"].astype(np.int64), device=dev); first = Z["first_ix"]
EVO, EVI = Z["ev_origin"], Z["ev_item"]; ii = pd.Series(np.arange(N), index=items)
W = 60; tot = X[:, :, :4].sum(2); csum = np.concatenate([np.zeros((N, 1), np.float32), np.cumsum(tot, 1)], 1)
dow = np.arange(Dd) + day0.dayofweek; DOW = np.stack([np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)], 1).astype(np.float32)
Xg = torch.tensor(X, device=dev); DOWg = torch.tensor(DOW, device=dev); first_g = torch.tensor(first, device=dev)
def window_sum(r, a, b): return csum[r, b + 1] - csum[r, a]   # units on days a..b (inclusive), vectorised over rows r
def batch_inputs(rows, cuts, ndays):
    """rows, cuts: int tensors on the GPU; returns sequence (B, 60, 9), context (B, 12) parts and L"""
    idx = cuts[:, None] + torch.arange(-W + 1, 1, device=dev)[None, :]
    seq = Xg[rows[:, None], idx]                                    # (B, 60, 5)
    tot_ = seq[:, :, :4].sum(2, keepdim=True); s = tot_.mean(1, keepdim=True) + 0.5
    live = (idx >= first_g[rows][:, None]).float()[:, :, None]
    x = torch.cat([torch.log1p(seq[:, :, :4] / s), torch.log1p(tot_ / s), seq[:, :, 4:5], DOWg[idx], live], 2)
    L = seq[:, -30:, :4].sum((1, 2)) * ndays / 30.0
    tm = ((day0 + pd.to_timedelta(cuts.cpu().numpy() + 1, "D")).month.values - 1).astype(np.float32)
    ctx = torch.stack([torch.tensor(np.sin(2 * np.pi * tm / 12), device=dev), torch.tensor(np.cos(2 * np.pi * tm / 12), device=dev),
                       torch.log1p(torch.clamp(cuts - first_g[rows], min=0).float()) / 7, torch.log1p(s[:, 0, 0])], 1)
    return x, ctx, L
class Net(nn.Module):
    def __init__(self, use_ctx):
        super().__init__(); self.use_ctx = use_ctx; self.lstm = nn.LSTM(9, 64, 2, batch_first=True, dropout=0.1)
        self.emb = nn.Embedding(int(ptype.max()) + 1, 8); k = 64 + (8 + 4 if use_ctx else 0)
        self.head = nn.Sequential(nn.Linear(k, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x, ctx, pt):
        h = self.lstm(x)[0][:, -1]
        if self.use_ctx: h = torch.cat([h, self.emb(pt), ctx], 1)
        return self.head(h)[:, 0]
def month_end_ix(o): return (pd.Period(o, "M").end_time.normalize() - day0).days
out = []; t_all = time.time()
for o in sorted(set(EVO)):
    end = month_end_ix(o); te = np.where(EVO == o)[0]; te_rows = ii.reindex(EVI[te]).values; ok = ~np.isnan(te_rows); te, te_rows = te[ok], te_rows[ok].astype(int)
    nd = pd.Period(o, "M").__add__(1).days_in_month
    cuts = np.arange(W - 1, end - 30 + 1, 7); R, C = np.meshgrid(np.arange(N), cuts, indexing="ij"); R, C = R.ravel(), C.ravel()
    keep = (window_sum(R, C - W + 1, C) >= 5) & (C >= first[R]); R, C = R[keep], C[keep]
    y = window_sum(R, C + 1, C + 30); Ltr = window_sum(R, C - 29, C)
    ztr = torch.tensor(np.log((y + 1) / (Ltr + 1)), device=dev, dtype=torch.float32); wtr = torch.tensor(np.sqrt(Ltr + 1), device=dev, dtype=torch.float32); wtr = wtr / wtr.mean()
    Rg, Cg = torch.tensor(R, device=dev), torch.tensor(C, device=dev)
    rows_te = torch.tensor(te_rows, device=dev); cut_te = torch.full((len(te_rows),), end, device=dev)
    for ver in ("daily", "ctx"):
        zs = []
        for seed in range(3):
            torch.manual_seed(seed); np.random.seed(seed); net = Net(ver == "ctx").to(dev); opt = torch.optim.Adam(net.parameters(), lr=2e-3); lossf = nn.HuberLoss(delta=0.5, reduction="none")
            for ep in range(5):
                perm = torch.randperm(len(R), device=dev); net.train()
                for b in range(0, len(R), 1024):
                    j = perm[b:b + 1024]; x, ctx, _ = batch_inputs(Rg[j], Cg[j], 30)
                    loss = (lossf(net(x, ctx, ptype[Rg[j]]), ztr[j]) * wtr[j]).mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            net.eval()
            with torch.no_grad():
                x, ctx, Lte = batch_inputs(rows_te, cut_te, nd); zs.append(net(x, ctx, ptype[rows_te]).cpu().numpy())
        zh = np.mean(zs, 0); L = Lte.cpu().numpy()
        out.append(pd.DataFrame({"origin": o, "style_color": EVI[te], "version": ver, "z_hat": zh, "L": L, "y1_hat": np.clip((L + 1) * np.exp(zh) - 1, 0, None)}))
    print(o, "| training sequences", len(R), "| scored items", len(te), f"| {time.time() - t_all:.0f} s", flush=True)
pd.concat(out).to_parquet("out/lstm_preds.parquet", index=False); print("saved")
