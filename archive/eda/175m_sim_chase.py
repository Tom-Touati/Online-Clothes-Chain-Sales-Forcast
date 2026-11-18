"""D265 (ideas 4 and 6 of 18 Nov): buying simulation (production policy, 11 windows) of the u65ec ladder (D260: rules, the volume-weighted
D253 tree member at q65, NN, conformal TabPFN and TabPFN dyn 32 at q65; 10.36%) with
  geo       the D259 per-item correction multiplier (data/geo_ec_ns_x_preds.parquet, geo / F) applied to the ladder quantities
  chase     a breakout check in the peak months (Sep-Dec), where the production nowcast is off (D083): if the item's last 14 days sold
            >= thr x what the forecast expected for them, its forecast for the next cover period is multiplied by min(ratio, 3)^0.5
            (upward only). young = items launched < 183 days before the decision day; all = every item. D248: half of the
            breakouts show within a month (month-1 ratio >= 1.5).
Writes data/sim_chase_windows.csv."""
import sys, numpy as np, pandas as pd
from scipy.stats import norm
sys.stdout.reconfigure(encoding="utf-8")
exec(open("eda/87_sim_systems.py", encoding="utf-8").read().split("SYS = [")[0])
Q = {}
for _k, _f in {"w03ec": "data/q_backtest_nx_w03ec.parquet", "conf": "data/q_backtest_fkconf_pfn2dyn_seg_tails_x.parquet", "v2d32": "data/q_backtest_fktab_tabpfn_v2_dyn_e32_x.parquet"}.items():
    Q[_k] = pd.read_parquet(_f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).q65
_g = pd.read_parquet("data/geo_ec_ns_x_preds.parquet").set_index(["origin", "style_color"]); MULT = (_g.geo / _g.F.clip(lower=1e-9)).clip(0.2, 5)
it = pd.read_parquet("data/items.parquet").set_index("style_color")
pub = pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
START = pd.concat([pub, pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
CUR = {}
def mm(o, items, src):
    b = sim.BT[sim.BT.origin == o].pivot(index="style_color", columns="h", values="pred").reindex(items)
    sh = b.values / b.sum(axis=1).values[:, None]
    mem = lambda k: np.nan_to_num(sh * Q[k].loc[o].reindex(items).values[:, None])
    lg = mem("w03ec")
    nn = v2.NN[v2.NN.origin == o].pivot(index="style_color", columns="h", values="q50").reindex(items).values; nn = np.where(np.isnan(nn), lg, nn)
    ru = RUL[o].reindex(items).values; ru = np.where(np.isnan(ru), lg, ru)
    ex = [np.where(np.isnan(m := mem(k)), lg, m) for k in ["conf", "v2d32"]]
    out = np.maximum(0, np.mean([np.maximum(x, 0) for x in [ru, lg, nn, *ex]], axis=0))
    if CUR.get("geo"): out = out * MULT.loc[o].reindex(items).fillna(1.0).values[:, None]
    return out
v2.month_matrix = mm
def simulate3(w, tau=v2.TAU, alpha=v2.ALPHA, ksig=1.3, chase=None):
    """eda/59 simulate2 (level nowcast, off in the peak months) + the optional peak-month breakout check."""
    n, T = w.D.shape; arr = w.pre.copy(); onhand = np.zeros(n); sales = np.zeros((n, T)); inv = np.zeros((n, T)); last = np.full(n, -10 ** 6); orders = []
    z = norm.ppf(tau); allday = pd.date_range(w.W0 - pd.Timedelta(days=28), w.days[-1]); po = [str(pd.Period(dd, "M") - 1) for dd in allday]
    exp_past = np.stack([w.FF[o][:, j] if o in w.FF else np.full(n, np.nan) for j, o in enumerate(po)], 1)
    st = START.reindex(w.items).values
    for k in range(T):
        onhand += arr[:, k]
        if k < w.n_dec:
            ok = (k - last) >= sim.GAP; cov = sim.COVER if k + sim.GAP <= w.n_dec - 1 else w.T - k
            IP = onhand + arr[:, k + 1:].sum(axis=1); o = w.dec_origin[k]; j = k + w.off
            m28 = w.FF[o][:, j:j + cov].sum(axis=1); sm = w.SIG[o]
            hist = np.concatenate([w.hist, sales[:, :k]], axis=1)[:, -14:].sum(axis=1)
            peak = w.days[k].month in v2.PEAK_MONTHS; a_k = 0.0 if peak else alpha
            if a_k > 0: m28 = np.exp((1 - a_k) * np.log(m28 + 0.5) + a_k * np.log(hist * cov / 14 + 0.5)) - 0.5
            elif chase is not None:
                e14 = exp_past[:, j - 14:j].sum(axis=1); r = (hist + 2) / (np.nan_to_num(e14, nan=hist) + 2)
                flag = r >= chase["thr"]
                if chase["young"]: flag &= (pd.Timestamp(w.days[k]) - pd.to_datetime(st)).days.values < 183
                m28 = np.where(flag, m28 * np.clip(r, 1, 3) ** 0.5, m28)
            m28 = np.maximum(m28, 0); s = np.sqrt(sm ** 2 + np.log1p(1 / np.maximum(m28, 0.5)))
            q = np.maximum(0, np.round(m28 * np.exp(z * s) - IP))
            for i in np.where(ok & (q > 0))[0]:
                arr[i, k + sim.LEAD] += q[i]; last[i] = k; orders.append((i, k, float(q[i]), float(m28[i]), np.nan, float(IP[i])))
        s_ = np.minimum(w.D[:, k], onhand); sales[:, k] = s_; onhand -= s_; inv[:, k] = onhand
    return dict(orders=pd.DataFrame(orders, columns=["i", "k", "q", "m28", "S", "IP"]), arr=arr, sales=sales, inv=inv)
VARS = [("u65ec", False, None), ("u65ec_geo", True, None), ("u65ec_chase_young1.5", False, dict(thr=1.5, young=True)),
        ("u65ec_chase_young1.3", False, dict(thr=1.3, young=True)), ("u65ec_chase_all1.5", False, dict(thr=1.5, young=False))]
rows = []
for start in pd.date_range("2025-03-01", "2026-01-01", freq="MS"):
    ideal = sim.simulate(sim.Window(start), "ideal")
    for nm, geo, ch in VARS:
        CUR["geo"] = geo; w = v2.build(start, src="r_q_nn", shape="cal")
        rows.append(dict(start=str(start.date()), variant=nm, **sim.summarize(w, simulate3(w, chase=ch), ideal)))
    print(start.date(), flush=True)
R = pd.DataFrame(rows); R.to_csv("data/sim_chase_windows.csv", index=False)
piv = R.pivot(index="start", columns="variant", values="loss_pct").join(pd.read_csv("data/sim_ladder_nx_windows.csv").pivot(index="start", columns="variant", values="loss_pct"))
print(piv.round(4).to_string()); print("mean loss:", piv.mean().round(4).sort_values().to_dict())
