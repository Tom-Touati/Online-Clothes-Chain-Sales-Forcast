"""Buying simulation v2: fixes for the decision-side shortcomings found in D068, on the same 11 rolling 3-month windows.
(a) within-month daily shape: each month's forecast is spread over its days with a site calendar profile (day of week x
    deep-discount day lift, fitted on the 365 days before the window; the deep-discount days themselves are the campaign
    calendar, which the business knows in advance) instead of evenly;
(b) nowcast as a ratio: recent 14-day sales are compared with what the forecast expected for those same 14 days, and the
    forward forecast is scaled by that ratio^alpha (the D067 nowcast blended levels, so it chased the Black Friday peak);
(c) forecast source: LightGBM (as D067), the embedding NN (monthly P50, pipeline/57), or their 50/50 blend; spread either the
    D067 4-month IQR or the NN's own month-level IQR.
Usage: python pipeline/59_sim_v2.py [nn=nn_backtest_preds_tag]"""
import importlib.util, sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
ARGS = dict(a.split("=") for a in sys.argv[1:])
spec = importlib.util.spec_from_file_location("sim", "pipeline/47_decision_sim.py"); sim = importlib.util.module_from_spec(spec)
sys.argv = ["x"]; spec.loader.exec_module(sim)
from scipy.stats import norm
NNF = ARGS.get("nn"); NN = pd.read_parquet(f"data/{NNF}.parquet") if NNF else None
TAU, ALPHA = 0.8, 0.5
PEAK_MONTHS = (9, 10, 11, 12)          # decision months with the nowcast off when peak_off=True (D083)

d = pd.read_parquet("data/daily_item_chan.parquet", columns=["ny_date", "cancelled", "class", "qty", "disc_w"]); d = d[~d.cancelled]; d["ny_date"] = pd.to_datetime(d.ny_date)
on = d[d["class"] == "ONLINE"].groupby("ny_date")[["qty", "disc_w"]].sum(); DEEP = (on.disc_w / on.qty).ge(0.585).astype(float)
SITE = d.groupby("ny_date").qty.sum(); del d


def cal_weights(days, start):
    """log(site units) = month fixed effect + day of week + beta * deep day, fitted on the 365 days before `start`."""
    h = SITE[(SITE.index < start) & (SITE.index >= start - pd.Timedelta(days=365))]
    X = pd.get_dummies(h.index.to_period("M").astype(str)).astype(float).values
    dow = np.eye(7)[h.index.dayofweek][:, 1:]; dp = DEEP.reindex(h.index).fillna(0).values[:, None]
    A = np.hstack([X, dow, dp]); b = np.linalg.lstsq(A, np.log(h.values), rcond=None)[0]
    cd, beta = np.r_[0, b[X.shape[1]:X.shape[1] + 6]], b[-1]
    return np.exp(cd[days.dayofweek] + beta * DEEP.reindex(days).fillna(0).values), beta


def month_matrix(o, items, src):
    """monthly forecast (n x 4) for origin o; lgb = D067 construction, nn = NN monthly P50, blend = 50/50."""
    b = sim.BT[sim.BT.origin == o].pivot(index="style_color", columns="h", values="pred").reindex(items)
    q = sim.QB.loc[o].reindex(items)
    lg = b.values * (q.q50.values / b.sum(axis=1).values)[:, None]
    if src == "lgb": return np.nan_to_num(lg)
    x = NN[NN.origin == o].pivot(index="style_color", columns="h", values="q50").reindex(items).values
    x = np.where(np.isnan(x), lg, x)
    return np.nan_to_num(x if src == "nn" else 0.5 * lg + 0.5 * x)


def nn_sig(o, items):
    x25 = NN[NN.origin == o].pivot(index="style_color", columns="h", values="q25").reindex(items).values
    x75 = NN[NN.origin == o].pivot(index="style_color", columns="h", values="q75").reindex(items).values
    return np.log((x75 + 1) / (x25 + 1)) / 1.349        # n x 4, month-level log spread


def build(start, src="lgb", shape="even", sig="total", oracle=False):
    w = sim.Window(start); n = len(w.items)
    allday = pd.date_range(w.W0 - pd.Timedelta(days=28), w.days[-1])
    wt = cal_weights(allday, w.W0)[0] if shape == "cal" else np.ones(len(allday))
    mon = allday.to_period("M")
    origins = sorted(set(w.origins) | {str(pd.Period(w.W0, "M") - 2)})
    FF, SS = {}, {}
    for o in origins:
        F = np.zeros((n, len(allday))); Sg = np.zeros((n, len(allday)))
        if oracle: mh = np.stack([sim.DEM.loc[w.items, pd.date_range((pd.Period(o) + h).start_time, (pd.Period(o) + h).end_time)].sum(axis=1).values for h in range(1, 5)], 1)
        elif o in sim.QB.index.get_level_values(0): mh = month_matrix(o, w.items, src)
        else: continue
        sg = nn_sig(o, w.items) if (sig == "month" and NN is not None) else None
        for h in range(1, 5):
            m = pd.Period(o) + h; idx = np.where(mon == m)[0]
            if len(idx) == 0: continue
            full = pd.date_range(m.start_time, m.end_time); wf = cal_weights(full, w.W0)[0] if shape == "cal" else np.ones(len(full))
            F[:, idx] = mh[:, h - 1:h] * (wt[idx] / wf.sum())[None, :]
            if sg is not None: Sg[:, idx] = sg[:, h - 1:h]
        FF[o] = F; SS[o] = Sg
    w.FF, w.SS, w.off = FF, SS, 28        # day index k in the window = column k + 28 of the FF arrays
    w.sig_mode = sig
    return w


def simulate2(w, tau=TAU, alpha=ALPHA, now="level", ksig=1.3, peak_off=False):
    n, T = w.D.shape; arr = w.pre.copy(); onhand = np.zeros(n); sales = np.zeros((n, T)); inv = np.zeros((n, T)); last = np.full(n, -10 ** 6); orders = []
    z = norm.ppf(tau)
    # what the forecast expected for each past day (forecast made at the month-end before that day, i.e. h=1)
    allday = pd.date_range(w.W0 - pd.Timedelta(days=28), w.days[-1])
    po = [str(pd.Period(dd, "M") - 1) for dd in allday]
    exp_past = np.stack([w.FF[o][:, j] if o in w.FF else np.full(n, np.nan) for j, o in enumerate(po)], 1)
    for k in range(T):
        onhand += arr[:, k]
        if k < w.n_dec:
            ok = (k - last) >= sim.GAP; cov = sim.COVER if k + sim.GAP <= w.n_dec - 1 else w.T - k
            IP = onhand + arr[:, k + 1:].sum(axis=1)
            o = w.dec_origin[k]; j = k + w.off
            m28 = w.FF[o][:, j:j + cov].sum(axis=1)
            sm = (ksig * w.SS[o][:, j:j + cov].mean(axis=1)) if w.sig_mode == "month" else w.SIG[o]
            hist = np.concatenate([w.hist, sales[:, :k]], axis=1)[:, -14:].sum(axis=1)
            a_k = 0.0 if (peak_off and w.days[k].month in PEAK_MONTHS) else alpha
            if a_k > 0 and now == "level":
                m28 = np.exp((1 - a_k) * np.log(m28 + 0.5) + a_k * np.log(hist * cov / 14 + 0.5)) - 0.5
            elif a_k > 0 and now == "ratio":
                e14 = exp_past[:, j - 14:j].sum(axis=1)
                r = np.clip((hist + 2) / (np.nan_to_num(e14, nan=hist) + 2), 1 / 3, 3)
                m28 = m28 * r ** a_k
            m28 = np.maximum(m28, 0); s = np.sqrt(sm ** 2 + np.log1p(1 / np.maximum(m28, 0.5)))
            q = np.maximum(0, np.round(m28 * np.exp(z * s) - IP))
            for i in np.where(ok & (q > 0))[0]:
                arr[i, k + sim.LEAD] += q[i]; last[i] = k; orders.append((i, k, float(q[i]), float(m28[i]), np.nan, float(IP[i])))
        s_ = np.minimum(w.D[:, k], onhand); sales[:, k] = s_; onhand -= s_; inv[:, k] = onhand
    return dict(orders=pd.DataFrame(orders, columns=["i", "k", "q", "m28", "S", "IP"]), arr=arr, sales=sales, inv=inv)


if __name__ == "__main__":
    VAR = [("D067 model (lgb, even, level nowcast)", dict(src="lgb", shape="even"), dict(now="level")),
           ("lgb, no nowcast", dict(src="lgb", shape="even"), dict(alpha=0.0)),
           ("lgb, calendar shape", dict(src="lgb", shape="cal"), dict(now="level")),
           ("lgb, ratio nowcast", dict(src="lgb", shape="even"), dict(now="ratio")),
           ("lgb, calendar shape + ratio nowcast", dict(src="lgb", shape="cal"), dict(now="ratio")),
           ("monthly oracle, even", dict(src="lgb", shape="even", oracle=True), dict(alpha=0.0)),
           ("monthly oracle, calendar shape", dict(src="lgb", shape="cal", oracle=True), dict(alpha=0.0))]
    if NN is not None:
        VAR += [("blend, even, level nowcast", dict(src="blend", shape="even"), dict(now="level")),
                ("blend, calendar, level nowcast", dict(src="blend", shape="cal"), dict(now="level")),
                ("blend, calendar, no nowcast", dict(src="blend", shape="cal"), dict(alpha=0.0)),
                ("blend, calendar, level nowcast off Sep-Dec", dict(src="blend", shape="cal"), dict(now="level", peak_off=True)),
                ("lgb, even, level nowcast off Sep-Dec", dict(src="lgb", shape="even"), dict(now="level", peak_off=True)),
                ("nn, calendar + ratio", dict(src="nn", shape="cal"), dict(now="ratio")),
                ("blend, calendar + ratio", dict(src="blend", shape="cal"), dict(now="ratio")),
                ("nn, calendar + ratio, month spread", dict(src="nn", shape="cal", sig="month"), dict(now="ratio")),
                ("blend, calendar + ratio, month spread", dict(src="blend", shape="cal", sig="month"), dict(now="ratio"))]
    rows = []
    for start in pd.date_range("2025-03-01", "2026-01-01", freq="MS"):
        w0 = sim.Window(start); ideal = sim.simulate(w0, "ideal")
        for name, bk, sk in VAR:
            w = build(start, **bk); r = simulate2(w, **sk)
            rows.append(dict(start=str(start.date()), variant=name, **sim.summarize(w, r, ideal)))
        print(start.date(), "beta(deep day)", round(cal_weights(pd.date_range(start, periods=2), start)[1], 3), flush=True)
    R = pd.DataFrame(rows); R.to_csv(f"data/sim_v2_windows{('_' + NNF) if NNF else ''}.csv", index=False)
    piv = R.pivot(index="start", columns="variant", values="loss_pct")[[v[0] for v in VAR]]
    print("\n== loss (% of ideal profit) by window start"); print(piv.round(3).T.to_string())
    print("\nmean loss:"); print(piv.mean().round(4).to_string())
    base = piv.iloc[:, 0]; print("\nwindows better than D067 model:"); print((piv.lt(base, axis=0)).sum().to_string())
