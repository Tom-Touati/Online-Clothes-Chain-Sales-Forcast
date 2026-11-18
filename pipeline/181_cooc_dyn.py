"""D193 (fork): neural co-occurrence embedding over time (user request), remote GPU.
For month t and items i, j alive that month: orders containing both ~ Poisson(mu), log mu = log(R_it R_jt / N_t) + <v_i(t), v_j(t)> + c,
where R_it = online orders containing item i and N_t = online orders (the independence baseline), and
  v_i(t) = U_i + X_i W + sum_{k <= t} delta_k,i         (free part + descriptor part + smooth monthly drift, 32 dims)
Full likelihood over every alive pair (dense on the GPU); L2 penalties on U, W and (strongly) on the drift steps.
Time-honest: at each origin o (2024-03 .. 2026-06) the model sees months <= o only; it starts from the previous origin's fit.
Test, for each origin with a following month: predict month o+1's pair counts with v(o) (drift held), scored by Poisson deviance
relative to independence, on pairs of items alive in both months; against (a) a static PPMI-SVD embedding of the last 12 months
(cosine x a fitted slope, fitted on month o) and (b) the descriptor part alone (X W, what a brand-new item would get).
Writes out/fk_cooc_dyn.npz (per origin: v(o), v(o) - v(o-3), X W) and prints the test."""
import sys, time, numpy as np, torch
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
Z = np.load("data/fk_cooc_inputs.npz"); items = Z["items"]; months = list(Z["months"]); X = torch.tensor(Z["X"], device=dev)
R = torch.tensor(Z["R"], device=dev); N = torch.tensor(Z["N"], device=dev, dtype=torch.float32); T, n = R.shape; D = 32
C = torch.zeros(T, n, n, device=dev)
pm, pi, pj, pn = [torch.tensor(Z[k].astype(np.int64 if k != "pn" else np.float32), device=dev) for k in ["pm", "pi", "pj", "pn"]]
C[pm, pi, pj] = pn; C[pm, pj, pi] = pn
U = torch.nn.Parameter(0.1 * torch.randn(n, D, device=dev)); W = torch.nn.Parameter(0.01 * torch.randn(X.shape[1], D, device=dev))
dl = torch.nn.Parameter(torch.zeros(T, n, D, device=dev)); c0 = torch.nn.Parameter(torch.zeros(1, device=dev)); sc = torch.nn.Parameter(torch.ones(1, device=dev))
LU, LW, LD = 1e-3, 1e-3, 3e-2
tri = torch.triu(torch.ones(n, n, device=dev, dtype=torch.bool), 1)


def V(t): return U + X @ W + dl[:t + 1].sum(0)


def sim(A, B=None):
    An = A / (A.norm(dim=1, keepdim=True) + 1e-6); Bn = An if B is None else B / (B.norm(dim=1, keepdim=True) + 1e-6)
    return sc[0] * (An @ Bn.T) + c0


def month_nll(t, Vt):
    a = R[t] > 0; m = tri & a[:, None] & a[None, :]
    base = torch.log(R[t][:, None] * R[t][None, :] / N[t] + 1e-12)
    eta = base + sim(Vt)
    return ((torch.exp(eta) - C[t] * eta) * m).sum() / max(N[t].item(), 1)


def deviance(t, S):
    """Poisson deviance of month t counts with log mu = base + S, pairs of items alive at t and t-1; S = None means independence."""
    a = (R[t] > 0) & (R[t - 1] > 0); m = tri & a[:, None] & a[None, :]
    base = torch.log(R[t][:, None] * R[t][None, :] / N[t] + 1e-12); mu = torch.exp(base + (S if S is not None else 0))
    y = C[t]; d = 2 * (torch.where(y > 0, y * torch.log(y / mu), torch.zeros_like(y)) - (y - mu))
    return (d * m).sum().item()


def ppmi_sim(t):
    Cs = C[max(0, t - 11):t + 1].sum(0); r = Cs.sum(1); tot = r.sum(); ctx = r ** 0.75; ctx = ctx / ctx.sum()
    pmi = torch.log((Cs / tot) / (torch.outer(r / tot, ctx)) + 1e-12); pmi = torch.where(Cs > 0, torch.clamp(pmi, min=0), torch.zeros_like(pmi))
    u, s, _ = torch.svd_lowrank(pmi, q=24); E = u * s.sqrt(); E = E / (E.norm(dim=1, keepdim=True) + 1e-9); return E @ E.T


opt = torch.optim.Adam([U, W, dl, c0, sc], lr=0.01); out, res, t0 = {}, [], time.time()
for t, o in enumerate(months):
    if o < "2024-03": continue
    steps = 600 if not out else 150
    for _ in range(steps):
        loss = sum(month_nll(k, V(k)) for k in range(t + 1)) / (t + 1)
        loss = loss + LU * (U ** 2).sum() + LW * (W ** 2).sum() + LD * (dl[1:t + 1] ** 2).sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([U, W, dl, c0, sc], 5.0); opt.step()
        with torch.no_grad(): dl[t + 1:] = 0
    with torch.no_grad():
        Vo = V(t); Vb = V(max(0, t - 3)); XW = X @ W
        out[o] = Vo.cpu().numpy().astype(np.float16); out[o + "_vel"] = (Vo - Vb).cpu().numpy().astype(np.float16); out[o + "_xw"] = XW.cpu().numpy().astype(np.float16)
        if t + 1 < T:
            d0 = deviance(t + 1, None); dd = deviance(t + 1, sim(Vo)); dx = deviance(t + 1, sim(XW))
            Sp = ppmi_sim(t); a = (R[t] > 0); m = tri & a[:, None] & a[None, :]
            yb = C[t][m]; base = torch.log(R[t][:, None] * R[t][None, :] / N[t] + 1e-12)[m]; sv = Sp[m]
            ab = torch.zeros(2, device=dev, requires_grad=True); o2 = torch.optim.Adam([ab], lr=0.05)
            with torch.enable_grad():
                for _ in range(300):
                    eta = base + ab[0] * sv + ab[1]; l = (torch.exp(eta) - yb * eta).mean(); o2.zero_grad(); l.backward(); o2.step()
            ds = deviance(t + 1, ab[0].detach() * Sp + ab[1].detach())
            res.append((o, 1 - dd / d0, 1 - ds / d0, 1 - dx / d0))
            print(f"loss {loss.item():.4f} | {o} -> {months[t + 1]}: deviance explained vs independence: dynamic {1 - dd / d0:.3f} | static PPMI-SVD {1 - ds / d0:.3f} | descriptor part only {1 - dx / d0:.3f} | {time.time() - t0:.0f}s", flush=True)
np.savez_compressed("out/fk_cooc_dyn.npz", items=items, **out)
r = np.array([x[1:] for x in res]); print("mean deviance explained, next month: dynamic %.3f | static %.3f | descriptors only %.3f" % tuple(r.mean(0)))
