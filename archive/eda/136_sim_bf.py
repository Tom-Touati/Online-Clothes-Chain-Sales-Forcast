"""Buying simulation with the Black Friday fix (D137c): the three-way monthly forecast (eda/87 'r_q_nn') with November scaled by
(a) the last-year level calibration and (b) level + item-lift allocation (ratios from eda/132, data/bf_fix_ratios.parquet; months other
than November and origins without the fix keep ratio 1). Production buying policy, 11 windows, calibrated spread unchanged."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
src = open("eda/87_sim_systems.py", encoding="utf-8").read().split("SYS = [")[0]
exec(src)
RAT = pd.read_parquet("data/bf_fix_ratios.parquet").set_index(["origin", "style_color", "h"])
base_mm = month_matrix; MODE = {"v": None}
def mm(o, items, src):
    M = base_mm(o, items, "r_q_nn")
    if MODE["v"] is None: return M
    for h in range(1, 5):
        r = RAT[MODE["v"]].reindex(pd.MultiIndex.from_arrays([[o] * len(items), items, [h] * len(items)])).fillna(1.0).values
        M[:, h - 1] = M[:, h - 1] * r
    return M
v2.month_matrix = mm
rows = []
for start in pd.date_range("2025-03-01", "2026-01-01", freq="MS"):
    ideal = sim.simulate(sim.Window(start), "ideal")
    for v in [None, "r_level", "r_bf"]:
        MODE["v"] = v; w = v2.build(start, src="r_q_nn", shape="cal")
        rows.append(dict(start=str(start.date()), variant=v or "three-way", **sim.summarize(w, v2.simulate2(w, now="level", peak_off=True), ideal)))
    print(start.date(), flush=True)
R = pd.DataFrame(rows); R.to_csv("data/sim_bf_windows.csv", index=False)
piv = R.pivot(index="start", columns="variant", values="loss_pct")[["three-way", "r_level", "r_bf"]]
print(piv.round(4).to_string()); print("mean loss:", piv.mean().round(4).to_dict())
print("windows better than three-way:", {c: int((piv[c] < piv["three-way"]).sum()) for c in ["r_level", "r_bf"]})
print(R.groupby("variant")[["fill", "left_units"]].mean().round(3).to_string())
