"""Buying simulation with the D117 distribution variants as the safety-stock spread (production policy, 11 windows, as eda/87).
Centre = the three-way monthly combination (eda/87 'r_q_nn'). The spread for each item and origin is sigma = (P75 - P25 log offset) / 1.349
of the variant, times 1.3 / 1.25 so that D0 reproduces the simulation's own calibrated spread (K_IQR = 1.3 on the raw quantile IQR).
Items without a variant spread keep the default. Writes data/sim_dist_windows.csv."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
src = open("eda/87_sim_systems.py", encoding="utf-8").read().split("SYS = [")[0]
exec(src)
Z = np.load("data/dist_items.npz", allow_pickle=True)
key = pd.MultiIndex.from_arrays([Z["origin"], Z["style_color"]])
SIGV = {v: pd.Series((Z[v][:, 3] - Z[v][:, 1]) / 1.349 * 1.3 / 1.25, index=key) for v in ["D0", "D2", "D3"]}
rows = []
for start in pd.date_range("2025-03-01", "2026-01-01", freq="MS"):
    ideal = sim.simulate(sim.Window(start), "ideal")
    for v in ["default", "D0", "D2", "D3"]:
        w = v2.build(start, src="r_q_nn", shape="cal")
        if v != "default":
            for o in w.SIG:
                s = SIGV[v].reindex(pd.MultiIndex.from_arrays([[o] * len(w.items), w.items])).values
                w.SIG[o] = np.where(np.isnan(s), w.SIG[o], s)
        rows.append(dict(start=str(start.date()), variant=v, **sim.summarize(w, v2.simulate2(w, now="level", peak_off=True), ideal)))
    print(start.date(), flush=True)
R = pd.DataFrame(rows); R.to_csv("data/sim_dist_windows.csv", index=False)
piv = R.pivot(index="start", columns="variant", values="loss_pct")[["default", "D0", "D2", "D3"]]
print(piv.round(4).to_string()); print("mean loss:", piv.mean().round(4).to_dict())
print("windows D2 better than D0:", int((piv.D2 < piv.D0).sum()), "| D3 better than D0:", int((piv.D3 < piv.D0).sum()))
print(R.groupby("variant")[["fill", "left_units"]].mean().round(3).to_string())
