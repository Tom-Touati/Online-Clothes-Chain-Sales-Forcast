"""Business-decision test of the forecast: a daily buying simulation.

Rules given by the user: an order placed on day d arrives on day d+21 (3-week lead time), and two orders of the same
item must be at least 7 days apart. Each day the decision model sees the latest month-end forecast (backtest version,
so no look-ahead) and the stock it holds, and decides how many units of each item to buy.

Money (assumptions; the dataset has no prices or costs): every unit sells at price P=1 (so revenue = units sold),
costs C=0.40 to buy, unsold stock at the end of the simulation is worth S_SALV=0.20 (marked down), holding costs
25%/year of cost. Demand that finds no stock is lost. Demand = historical non-cancelled units per item per day
(itself censored by past stock-outs, so this is a lower bound on true demand).

Ideal = hindsight orders on the same days that exactly cover each order's coverage window (no lost sales, no leftover).
Loss = ideal profit - actual profit. The revenue gap (ideal revenue - actual revenue) is also reported, but it cannot be
the loss on its own: buying unlimited stock would make it zero.

Decision model (order-up-to): on an allowed day k, target S = m28 * exp(z_tau * sigma), where m28 is the forecast median
demand for days k..k+27 (this order's lead time plus the 7 days until the next order can arrive), sigma combines the
item's calibrated model spread (IQR of the 4-month quantiles, k=1.3) with Poisson noise for a 28-day window, and
q = max(0, round(S - stock on hand - stock on order)). Parameters tau (service quantile) and alpha (weight on the last
14 days of observed sales) are trained on a Mar-May 2025 window and tested on Jul-Sep 2025 and Oct-Dec 2025."""
import pandas as pd, numpy as np, sys, json
from scipy.stats import norm
sys.stdout.reconfigure(encoding="utf-8")

P, C = 1.0, 0.40
S_SALV = 0.5 * C
H = 0.25 * C / 365            # holding cost per unit per day
LEAD, GAP, COVER = 21, 7, 28
K_IQR = 1.3

d = pd.read_parquet("data/daily_item_chan.parquet", columns=["style_color", "ny_date", "cancelled", "qty"])
d = d[~d.cancelled]
d["ny_date"] = pd.to_datetime(d.ny_date)
DEM = d.groupby(["style_color", "ny_date"]).qty.sum().unstack(fill_value=0)
DEM = DEM.reindex(columns=pd.date_range(DEM.columns.min(), DEM.columns.max()), fill_value=0)
del d
BT = pd.read_parquet("data/best_backtest_preds.parquet")
QB = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"])


class Window:
    """One 3-month decision window: decisions on days 0..n_dec-1, simulation runs to the last order's coverage end."""
    def __init__(self, start, months=3):
        self.W0 = pd.Timestamp(start)
        self.Wend = self.W0 + pd.offsets.MonthEnd(months)
        self.days = pd.date_range(self.W0, self.Wend + pd.Timedelta(days=COVER - 1))
        self.T = len(self.days); self.n_dec = (self.Wend - self.W0).days + 1
        dec_months = pd.period_range(self.W0, self.Wend, freq="M")
        self.origins = [str(m - 1) for m in dec_months]
        items = set(DEM.index)
        for o in self.origins:
            items &= set(QB.loc[o].index[QB.loc[o].level > 0])
        self.items = np.array(sorted(items)); n = len(self.items)
        self.D = DEM.loc[self.items, self.days].values.astype(float)
        self.hist = DEM.loc[self.items, pd.date_range(self.W0 - pd.Timedelta(days=28), self.W0 - pd.Timedelta(days=1))].values.astype(float)
        # forecast daily rates per origin: monthly median split scaled so the 4 months sum to the P50 of the quantile model
        self.F, self.SIG = {}, {}
        dmon = self.days.to_period("M")
        for o in self.origins:
            b = BT[BT.origin == o].pivot(index="style_color", columns="h", values="pred").reindex(self.items)
            q = QB.loc[o].reindex(self.items)
            mh = b.values * (q.q50.values / b.sum(axis=1).values)[:, None]
            F = np.full((n, self.T), np.nan)
            for t, m in enumerate(dmon):
                h = (m - pd.Period(o)).n
                if 1 <= h <= 4:
                    F[:, t] = mh[:, h - 1] / m.days_in_month
            self.F[o] = np.nan_to_num(F, nan=0.0)
            self.SIG[o] = K_IQR * np.log((q.q75.values + 1) / (q.q25.values + 1)) / 1.349
        self.dec_origin = [str(pd.Period(self.days[k], "M") - 1) for k in range(self.n_dec)]
        # perfect pre-window pipeline: arrivals on days 0..LEAD-1 equal that day's demand
        self.pre = np.zeros((n, self.T)); self.pre[:, :LEAD] = self.D[:, :LEAD]


def simulate(w, policy, tau=0.8, alpha=0.0, sig_naive=0.35, fixed_orders=None):
    """Closed-loop simulation. policy in {'model','naive','oracle','ideal'}; or fixed_orders = list of (i,k,q) (open loop).
    Returns dict with orders and daily arrays."""
    n, T = w.D.shape
    arr = w.pre.copy(); onhand = np.zeros(n); sales = np.zeros((n, T)); inv_end = np.zeros((n, T))
    last = np.full(n, -10 ** 6); orders = []
    z = norm.ppf(tau)
    fo = {}
    if fixed_orders is not None:
        for (i, k, q) in fixed_orders: fo.setdefault(k, []).append((i, q))
    for k in range(T):
        onhand += arr[:, k]
        if k < w.n_dec:
            if fixed_orders is not None:
                for i, q in fo.get(k, []):
                    arr[i, k + LEAD] += q; orders.append((i, k, q, np.nan, np.nan, np.nan))
            else:
                ok = (k - last) >= GAP
                cov = COVER if k + GAP <= w.n_dec - 1 else w.T - k      # the last possible order covers to the end
                pipe = arr[:, k + 1:].sum(axis=1)
                IP = onhand + pipe
                if policy == "ideal":
                    S = w.D[:, k:k + cov].sum(axis=1)              # exact cover through the next possible arrival
                    q = np.maximum(0, S - IP); m28 = S; sig = np.zeros(n)
                    ok = ok & (k % GAP == 0)
                elif policy == "oracle":
                    m28 = w.D[:, k:k + cov].sum(axis=1); sig = np.sqrt(np.log1p(1 / np.maximum(m28, 0.5)))
                    S = m28 * np.exp(z * sig); q = np.maximum(0, np.round(S - IP))
                else:
                    recent14 = np.concatenate([w.hist, sales[:, :k]], axis=1)[:, -14:].sum(axis=1) * cov / 14   # run rate over the cover
                    if policy == "naive":
                        recent28 = np.concatenate([w.hist, sales[:, :k]], axis=1)[:, -28:].sum(axis=1)
                        m28 = recent28 * cov / 28; sm = np.full(n, sig_naive)
                    else:
                        o = w.dec_origin[k]
                        m28 = w.F[o][:, k:k + cov].sum(axis=1); sm = w.SIG[o]
                        if alpha > 0:
                            m28 = np.exp((1 - alpha) * np.log(m28 + 0.5) + alpha * np.log(recent14 + 0.5)) - 0.5
                    m28 = np.maximum(m28, 0)
                    sig = np.sqrt(sm ** 2 + np.log1p(1 / np.maximum(m28, 0.5)))
                    S = m28 * np.exp(z * sig); q = np.maximum(0, np.round(S - IP))
                place = ok & (q > 0)
                for i in np.where(place)[0]:
                    arr[i, k + LEAD] += q[i]; last[i] = k
                    orders.append((i, k, float(q[i]), float(m28[i]), float(S[i]), float(IP[i])))
        s = np.minimum(w.D[:, k], onhand); sales[:, k] = s; onhand -= s; inv_end[:, k] = onhand
    O = pd.DataFrame(orders, columns=["i", "k", "q", "m28", "S", "IP"])
    return dict(orders=O, arr=arr, sales=sales, inv=inv_end)


def money(w, r):
    rev = r["sales"].sum() * P
    buy = r["orders"].q.sum() * C + w.pre.sum() * C
    hold = r["inv"].sum() * H
    salv = r["inv"][:, -1].sum() * S_SALV
    return dict(revenue=rev, buy=buy, hold=hold, salvage=salv, profit=rev - buy - hold + salv,
                lost_units=(w.D - r["sales"]).sum(), left_units=r["inv"][:, -1].sum(), demand=w.D.sum())


def summarize(w, r, ideal):
    a, b = money(w, r), money(w, ideal)
    return dict(revenue=a["revenue"], ideal_rev=b["revenue"], rev_gap=b["revenue"] - a["revenue"],
                profit=a["profit"], ideal_profit=b["profit"], loss=b["profit"] - a["profit"],
                loss_pct=(b["profit"] - a["profit"]) / b["profit"], fill=a["revenue"] / a["demand"],
                left_units=a["left_units"], bought=r["orders"].q.sum(), n_orders=len(r["orders"]))


def gradients(w, r):
    """Exact open-loop marginal profit of one more (g_plus) and one fewer (g_minus, sign = d profit / d q from the left)
    unit on each order, holding every other order fixed."""
    D, sales, inv = w.D, r["sales"], r["inv"]; T = D.shape[1]
    lost = D - sales > 1e-9
    zero_end = inv <= 1e-9
    gp, gm = [], []
    for i, k in zip(r["orders"].i.values, r["orders"].k.values):
        a = k + LEAD
        t = np.argmax(lost[i, a:]) if lost[i, a:].any() else -1
        gp.append(P - C - H * t if t >= 0 else -C + S_SALV - H * (T - a))
        t2 = np.argmax(zero_end[i, a:] & (sales[i, a:] > 0)) if (zero_end[i, a:] & (sales[i, a:] > 0)).any() else -1
        gm.append(P - C - H * t2 if t2 >= 0 else -C + S_SALV - H * (T - a))
    return np.array(gp), np.array(gm)


def fifo_trace(w, r):
    """Trace every sold unit to the order that supplied it (first in, first out) and every lost sale to the order
    whose coverage window contained that day (the latest order to have arrived). Returns per-order and per-month tables."""
    O = r["orders"].reset_index(drop=True); D, sales = w.D, r["sales"]; T = D.shape[1]
    months = w.days.to_period("M").astype(str).values
    sold = np.zeros(len(O)); lostc = np.zeros(len(O)); hold = np.zeros(len(O)); left = np.zeros(len(O))
    first_sale = np.full(len(O), -1); last_sale = np.full(len(O), -1)
    rows = []                                   # (order id or -1 for pre-window stock, month, sold, lost)
    by_item = O.groupby("i").groups
    for i in range(D.shape[0]):
        ids = sorted(by_item.get(i, []), key=lambda j: O.k[j])
        arrivals = {}
        for j in ids: arrivals.setdefault(O.k[j] + LEAD, []).append(j)
        queue = []; cover = -1
        for t in range(T):
            if w.pre[i, t] > 0: queue.append([-1, w.pre[i, t]])
            for j in arrivals.get(t, []):
                queue.append([j, O.q[j]]); cover = j
            need = sales[i, t]
            while need > 1e-9 and queue:
                take = min(need, queue[0][1]); j = queue[0][0]
                if j >= 0:
                    sold[j] += take; last_sale[j] = t
                    if first_sale[j] < 0: first_sale[j] = t
                rows.append((j, months[t], take, 0.0)); queue[0][1] -= take; need -= take
                if queue[0][1] <= 1e-9: queue.pop(0)
            L = D[i, t] - sales[i, t]
            if L > 1e-9:
                if cover >= 0: lostc[cover] += L
                rows.append((cover, months[t], 0.0, L))
            for j, u in queue:
                if j >= 0: hold[j] += u * H
        for j, u in queue:
            if j >= 0: left[j] += u
    O = O.assign(sold=sold, lost_in_cover=lostc, hold_cost=hold, left=left, first_sale=first_sale, last_sale=last_sale)
    M = pd.DataFrame(rows, columns=["order", "month", "sold", "lost"])
    return O, M


if __name__ == "__main__":
    import time; t0 = time.time()
    TRAIN, TESTS = "2025-03-01", ["2025-07-01", "2025-10-01"]
    W = {s: Window(s) for s in [TRAIN] + TESTS}
    for s, w in W.items(): print(s, "items", len(w.items), "days", w.T, "decision days", w.n_dec, "origins", w.origins)
    IDEAL = {s: simulate(w, "ideal") for s, w in W.items()}
    for s, w in W.items():
        m = money(w, IDEAL[s]); print("ideal", s, {k: round(v) for k, v in m.items()})

    # ---- train the decision model's two parameters on the training window
    grid = []
    for tau in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        for alpha in [0.0, 0.25, 0.5]:
            r = simulate(W[TRAIN], "model", tau, alpha)
            grid.append(dict(policy="model", tau=tau, alpha=alpha, **summarize(W[TRAIN], r, IDEAL[TRAIN])))
        r = simulate(W[TRAIN], "naive", tau)
        grid.append(dict(policy="naive", tau=tau, alpha=1.0, **summarize(W[TRAIN], r, IDEAL[TRAIN])))
    G = pd.DataFrame(grid)
    print("\n== training grid (Mar-May 2025 decisions), loss = ideal profit - actual profit")
    print(G[["policy", "tau", "alpha", "loss_pct", "fill", "rev_gap", "left_units", "bought"]].round(3).to_string(index=False))
    best = G[G.policy == "model"].sort_values("loss").iloc[0]; bestn = G[G.policy == "naive"].sort_values("loss").iloc[0]
    TAU, ALPHA, TAUN = float(best.tau), float(best.alpha), float(bestn.tau)
    print(f"trained: model tau={TAU} alpha={ALPHA}; naive tau={TAUN}  ({time.time()-t0:.0f}s)")

    # ---- test windows: trained model vs naive run-rate vs perfect-forecast oracle vs untuned P50 model
    res, keep = [], {}
    for s in TESTS:
        w = W[s]
        for name, kw in [("model (trained)", dict(policy="model", tau=TAU, alpha=ALPHA)),
                         ("model P50, no safety", dict(policy="model", tau=0.5, alpha=0.0)),
                         ("naive run-rate (trained)", dict(policy="naive", tau=TAUN)),
                         ("oracle forecast", dict(policy="oracle", tau=TAU))]:
            r = simulate(w, **kw); keep[(s, name)] = r
            res.append(dict(window=s, policy=name, **summarize(w, r, IDEAL[s])))
    R = pd.DataFrame(res)
    print("\n== test windows")
    print(R[["window", "policy", "revenue", "ideal_rev", "rev_gap", "profit", "ideal_profit", "loss", "loss_pct", "fill", "left_units", "n_orders"]].round(3).to_string(index=False))
    R.to_csv("data/sim_policy_results.csv", index=False); G.to_csv("data/sim_training_grid.csv", index=False)

    # ---- trace each decision of the trained model in the season-matched window
    out = {}
    for s in TESTS:
        w = W[s]; r = keep[(s, "model (trained)")]
        gp, gm = gradients(w, r)
        O, M = fifo_trace(w, r)
        O["g_plus"], O["g_minus"] = gp, gm
        O["item"] = w.items[O.i.values]; O["date"] = w.days[O.k.values]; O["arrive"] = w.days[O.k.values + LEAD]
        # best response holding the past fixed: units this order should have been, given the stock that actually arrived before it
        nxt = O.sort_values(["i", "k"]).groupby("i").k.shift(-1)
        O["cover_end"] = (nxt.reindex(O.index).fillna(w.T).astype(int) + LEAD).clip(upper=w.T)
        carry = np.array([r["inv"][i, k + LEAD - 1] for i, k in zip(O.i, O.k)])
        need = np.array([w.D[i, k + LEAD:e].sum() for i, k, e in zip(O.i, O.k, O.cover_end)])
        O["best_q"] = np.maximum(0, need - carry); O["excess"] = O.q - O.best_q
        O["verdict"] = np.where(O.g_plus > 0, "too few", np.where(O.g_minus < 0, "too many", "right"))
        O.to_parquet(f"data/sim_decisions_{s[:7]}.parquet"); M.to_parquet(f"data/sim_month_trace_{s[:7]}.parquet")
        # gradient check: perturb 30 random orders by +-1 in an open-loop rerun
        rng = np.random.default_rng(0); base = money(w, r)["profit"]; fixed = list(zip(O.i, O.k, O.q)); errs = []
        for j in rng.choice(len(O), 30, replace=False):
            for sgn, g in [(1, O.g_plus[j]), (-1, O.g_minus[j])]:
                if sgn < 0 and O.q[j] < 1: continue
                f2 = fixed.copy(); f2[j] = (f2[j][0], f2[j][1], f2[j][2] + sgn)
                dp = money(w, simulate(w, None, fixed_orders=f2))["profit"] - base
                errs.append(abs(dp - sgn * g))
        print(f"\n== {s}: gradient check on 30 orders: max |finite difference - analytic| = {max(errs):.2e}")
        mon = M.assign(phase=np.where(M.order < 0, "pre-window stock", "window decisions")).groupby(["month", "phase"])[["sold", "lost"]].sum().unstack(fill_value=0)
        ideal_m = pd.Series(w.D.sum(axis=0), index=w.days.to_period("M").astype(str)).groupby(level=0).sum()
        print("monthly revenue (units x P=1) by supplying decisions, lost revenue, ideal revenue:")
        mm = pd.DataFrame({"revenue": M.groupby("month").sold.sum(), "lost": M.groupby("month").lost.sum(), "ideal": ideal_m})
        mm["from_pre_window_stock"] = M[M.order < 0].groupby("month").sold.sum(); mm = mm.fillna(0)
        mm["gap_pct"] = mm.lost / mm.ideal
        print(mm.round(3).to_string())
        print("verdicts:", O.verdict.value_counts().to_dict())
        print("loss attributed: lost margin in coverage", round((O.lost_in_cover * (P - C)).sum()), "| leftover write-down", round((O.left * (C - S_SALV)).sum()),
              "| holding", round(O.hold_cost.sum()))
        out[s] = dict(monthly=mm.round(1).reset_index().to_dict("records"), verdicts=O.verdict.value_counts().to_dict())
    json.dump(dict(train=dict(tau=TAU, alpha=ALPHA, tau_naive=TAUN), results=res, traces=out), open("data/sim_summary.json", "w"), indent=1, default=float)
    print(f"done {time.time()-t0:.0f}s")
