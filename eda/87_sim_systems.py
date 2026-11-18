"""Buying-simulation profit loss for every system (D110). eda/59_sim_v2.py machinery and 11 rolling 3-month windows (2025-03 .. 2026-01),
order every 7 days at most, 21-day lead, P=1, C=0.40, salvage 0.20, holding 25%/yr; loss = % of the perfect-foresight profit.
Only the monthly forecast changes; the safety-stock spread is the calibrated quantile-model IQR for all systems (D067), tau 0.80.
Two policies: 'production' = calendar shape + 14-day level nowcast (alpha 0.5) switched off Sep-Dec (D083/D090);
'forecast only' = calendar shape, no nowcast (isolates forecast quality).
Monthly forecasts per (origin, item, h):
  R3     = mean of the last 3 months, every month
  qmodel = LightGBM monthly split scaled to the quantile-model P50 total (the D067 construction)
  lgb_nn = 50/50 LightGBM-split and NN monthly P50 (as eda/59 'blend')
  nn     = NN monthly P50 (S13x6);  rules = the rule agents' monthly forecasts
  q_nn, r_q, r_q_nn = geometric means of the members' monthly forecasts.
Items without a rules or NN forecast at an origin fall back to the qmodel construction (counted)."""
import sys, importlib.util, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
sys.argv = ["x", "nn=nn_backtest_preds_S13x6"]
spec = importlib.util.spec_from_file_location("v2", "eda/59_sim_v2.py"); v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
sim = v2.sim
Mo = pd.read_parquet("data/monthly_item.parquet"); Mo.columns = [str(c) for c in Mo.columns]
RUL = {}
for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
    r = pd.read_csv(f"agent_eval/{o}/predictions_verified.csv", dtype={"month": str})
    r["h"] = [(pd.Period(m, "M") - pd.Period(o, "M")).n for m in r.month]; RUL[o] = r.pivot(index="style_color", columns="h", values="pred")
FALL = {"rules": [0, 0], "nn": [0, 0]}
def g(*xs): return np.exp(np.mean([np.log(np.maximum(x, 0) + 0.5) for x in xs], axis=0)) - 0.5
def month_matrix(o, items, src):
    b = sim.BT[sim.BT.origin == o].pivot(index="style_color", columns="h", values="pred").reindex(items)
    q = sim.QB.loc[o].reindex(items)
    lg = np.nan_to_num(b.values * (q.q50.values / b.sum(axis=1).values)[:, None])
    if src == "qmodel": return lg
    if src == "R3":
        cols = [str(pd.Period(o, "M") - k) for k in range(3)]
        r3 = Mo.reindex(items)[cols].fillna(0).mean(axis=1).values
        return np.repeat(r3[:, None], 4, axis=1)
    nn = v2.NN[v2.NN.origin == o].pivot(index="style_color", columns="h", values="q50").reindex(items).values
    FALL["nn"][0] += np.isnan(nn).any(axis=1).sum(); FALL["nn"][1] += len(items); nn = np.where(np.isnan(nn), lg, nn)
    ru = RUL[o].reindex(items).values
    FALL["rules"][0] += np.isnan(ru).any(axis=1).sum(); FALL["rules"][1] += len(items); ru = np.where(np.isnan(ru), lg, ru)
    return np.maximum(0, {"nn": nn, "rules": ru, "lgb_nn": 0.5 * lg + 0.5 * nn, "q_nn": g(lg, nn), "r_q": g(ru, lg), "r_q_nn": g(ru, lg, nn)}[src])
v2.month_matrix = month_matrix
SYS = ["R3", "rules", "qmodel", "nn", "lgb_nn", "q_nn", "r_q", "r_q_nn"]
POL = {"production": dict(now="level", peak_off=True), "forecast only": dict(alpha=0.0)}
rows = []
for start in pd.date_range("2025-03-01", "2026-01-01", freq="MS"):
    ideal = sim.simulate(sim.Window(start), "ideal")
    for s in SYS:
        w = v2.build(start, src=s, shape="cal")
        for pn, pk in POL.items():
            rows.append(dict(start=str(start.date()), system=s, policy=pn, **sim.summarize(w, v2.simulate2(w, **pk), ideal)))
    print(start.date(), "done", flush=True)
R = pd.DataFrame(rows); R.to_csv("data/sim_systems_windows.csv", index=False)
print("fallback to qmodel construction (item-origin rows):", {k: f"{a} of {b}" for k, (a, b) in FALL.items()})
for pn in POL:
    piv = R[R.policy == pn].pivot(index="start", columns="system", values="loss_pct")[SYS]
    print(f"\n== policy: {pn} | loss (% of perfect-foresight profit) by window"); print(piv.round(3).to_string())
    print("mean loss:", piv.mean().round(4).to_dict())
    print("windows better than qmodel:", piv.lt(piv.qmodel, axis=0).sum().to_dict())
    print("windows better than R3:", piv.lt(piv.R3, axis=0).sum().to_dict())
    f = R[R.policy == pn].groupby("system")[["fill", "left_units"]].mean().reindex(SYS) if "left_units" in R else R[R.policy == pn].groupby("system")[["fill"]].mean().reindex(SYS)
    print("mean fill rate / leftover units:"); print(f.round(3).T.to_string())
