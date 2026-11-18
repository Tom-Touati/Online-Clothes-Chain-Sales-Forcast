"""D153: buying simulation with the new quantile model (data/q_backtest_nf_cpllm.parquet) in place of the lean one; built from eda/142 (D141b): three-way monthly forecast built on the lean + occasion-timing quantile model
(the quantile construction scaled to its P50, data/q_backtest_lean.parquet) with the D137 November ratios (data/bf_fix_ratios.parquet),
against the plain three-way and the lean three-way alone. Production policy, 11 windows, spread unchanged."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
src = open("eda/87_sim_systems.py", encoding="utf-8").read().split("SYS = [")[0]
exec(src)
LQ50 = pd.read_parquet("data/q_backtest_nf_cpllm.parquet").set_index(["origin", "style_color"]).q50
RAT = pd.read_parquet("data/bf_fix_ratios.parquet").set_index(["origin", "style_color", "h"]).r_bf
MODE = {"lean": False, "bf": False}
def mm(o, items, src):
    b = sim.BT[sim.BT.origin == o].pivot(index="style_color", columns="h", values="pred").reindex(items)
    q50 = (LQ50.loc[o].reindex(items).values if MODE["lean"] else sim.QB.loc[o].reindex(items).q50.values)
    lg = np.nan_to_num(b.values * (q50 / b.sum(axis=1).values)[:, None])
    nn = v2.NN[v2.NN.origin == o].pivot(index="style_color", columns="h", values="q50").reindex(items).values; nn = np.where(np.isnan(nn), lg, nn)
    ru = RUL[o].reindex(items).values; ru = np.where(np.isnan(ru), lg, ru)
    M = np.maximum(0, g(ru, lg, nn))
    if MODE["bf"]:
        for h in range(1, 5): M[:, h - 1] *= RAT.reindex(pd.MultiIndex.from_arrays([[o] * len(items), items, [h] * len(items)])).fillna(1.0).values
    return M
v2.month_matrix = mm
rows = []
for start in pd.date_range("2025-03-01", "2026-01-01", freq="MS"):
    ideal = sim.simulate(sim.Window(start), "ideal")
    for nm, lean, bf in [("new three-way", True, False), ("new three-way + BF fix", True, True)]:
        MODE.update(lean=lean, bf=bf); w = v2.build(start, src="r_q_nn", shape="cal")
        rows.append(dict(start=str(start.date()), variant=nm, **sim.summarize(w, v2.simulate2(w, now="level", peak_off=True), ideal)))
    print(start.date(), flush=True)
R = pd.DataFrame(rows); R.to_csv("data/sim_new_stack_windows.csv", index=False)
piv = R.pivot(index="start", columns="variant", values="loss_pct").join(pd.read_csv("data/sim_best_stack_windows.csv").pivot(index="start", columns="variant", values="loss_pct"))
piv = piv[["three-way", "lean three-way", "lean three-way + BF fix", "new three-way", "new three-way + BF fix"]]
print(piv.round(4).to_string()); print("mean loss:", piv.mean().round(4).to_dict())
print("windows better than three-way:", {c: int((piv[c] < piv["three-way"]).sum()) for c in piv.columns[1:]})
print(R.groupby("variant")[["fill", "left_units"]].mean().round(3).to_string())
