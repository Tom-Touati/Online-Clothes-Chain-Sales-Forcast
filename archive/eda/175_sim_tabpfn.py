"""D169 (fork request, D204): buying simulation of the fork's TabPFN stacks, production policy, 11 windows (as eda/142/168/171).
Monthly forecast = equal geometric mean of the members' monthly matrices: rules (monthly), the D198 quantile model
(data/q_backtest_fkdyn_dyn.parquet) and the TabPFN members (data/q_backtest_fktab_tabpfn_{v2,v35f}.parquet), each a 4-month P50
spread over months by the LightGBM monthly shape (as the quantile construction), and the NN (monthly P50). No Black Friday fix
(the fork found it no longer helps once TabPFN is in). Variants: p2 (+ v2), p35f (+ 3.5 fast), p2_35f (+ both).
Writes data/sim_tabpfn_stack_windows.csv."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
exec(open("eda/87_sim_systems.py", encoding="utf-8").read().split("SYS = [")[0])
Q = {k: pd.read_parquet(f).set_index(["origin", "style_color"]).q50 for k, f in
     {"dyn": "data/q_backtest_fkdyn_dyn.parquet", "v2": "data/q_backtest_fktab_tabpfn_v2.parquet", "v35f": "data/q_backtest_fktab_tabpfn_v35f.parquet"}.items()}
CUR = {"extra": []}
def mm(o, items, src):
    b = sim.BT[sim.BT.origin == o].pivot(index="style_color", columns="h", values="pred").reindex(items)
    sh = b.values / b.sum(axis=1).values[:, None]
    mem = lambda k: np.nan_to_num(sh * Q[k].loc[o].reindex(items).values[:, None])
    lg = mem("dyn")
    nn = v2.NN[v2.NN.origin == o].pivot(index="style_color", columns="h", values="q50").reindex(items).values; nn = np.where(np.isnan(nn), lg, nn)
    ru = RUL[o].reindex(items).values; ru = np.where(np.isnan(ru), lg, ru)
    ex = [np.where(np.isnan(m := mem(k)), lg, m) for k in CUR["extra"]]
    return np.maximum(0, g(ru, lg, nn, *ex))
v2.month_matrix = mm
rows = []
for start in pd.date_range("2025-03-01", "2026-01-01", freq="MS"):
    ideal = sim.simulate(sim.Window(start), "ideal")
    for nm, ex in [("p2", ["v2"]), ("p35f", ["v35f"]), ("p2_35f", ["v2", "v35f"])]:
        CUR["extra"] = ex; w = v2.build(start, src="r_q_nn", shape="cal")
        rows.append(dict(start=str(start.date()), variant=nm, **sim.summarize(w, v2.simulate2(w, now="level", peak_off=True), ideal)))
    print(start.date(), flush=True)
R = pd.DataFrame(rows); R.to_csv("data/sim_tabpfn_stack_windows.csv", index=False)
piv = R.pivot(index="start", columns="variant", values="loss_pct").join(pd.read_csv("data/sim_dyn_stack_windows.csv").pivot(index="start", columns="variant", values="loss_pct")).join(pd.read_csv("data/sim_wx_stack_windows.csv").pivot(index="start", columns="variant", values="loss_pct"))
print(piv.round(4).to_string()); print("mean loss:", piv.mean().round(4).to_dict())
