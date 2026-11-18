"""Buying-simulation loss (production policy, 11 windows, as eda/87/92) for the D148 residual-distribution members added to the
three-way as a fourth member (4w) or replacing the quantile member (3w). The member's monthly split is the LightGBM monthly shape scaled to its sample-median total (as the
quantile construction); the safety-stock spread stays the stored production IQR. Variants from argv (default: all rd files).
Writes data/sim_rd_windows.csv (variant = column name used by eda/145, e.g. log_4w)."""
import sys, numpy as np, pandas as pd
VARS = sys.argv[1:] or ["add", "log", "logt"]; sys.argv = sys.argv[:1]
sys.stdout.reconfigure(encoding="utf-8")
exec(open("eda/87_sim_systems.py", encoding="utf-8").read().split("SYS = [")[0])
VQ = {v: pd.read_parquet(f"data/q_backtest_rd_{v}.parquet").set_index(["origin", "style_color"]).q50 for v in VARS}
CUR = {"v": None, "mode": "4w"}
def month_matrix_v(o, items, src):
    b = sim.BT[sim.BT.origin == o].pivot(index="style_color", columns="h", values="pred").reindex(items)
    sh = b.values / b.sum(axis=1).values[:, None]
    lg = np.nan_to_num(sh * sim.QB.loc[o].reindex(items).q50.values[:, None])
    rd = VQ[CUR["v"]].loc[o].reindex(items).values; rd = np.where(np.isnan(rd), lg.sum(axis=1), rd)
    lr = np.nan_to_num(sh * rd[:, None])
    nn = v2.NN[v2.NN.origin == o].pivot(index="style_color", columns="h", values="q50").reindex(items).values; nn = np.where(np.isnan(nn), lg, nn)
    ru = RUL[o].reindex(items).values; ru = np.where(np.isnan(ru), lg, ru)
    return np.maximum(0, {"3w": g(ru, lr, nn), "4w": g(ru, lg, nn, lr), "base": g(ru, lg, nn)}[CUR["mode"]])
v2.month_matrix = month_matrix_v
rows = []
for start in pd.date_range("2025-03-01", "2026-01-01", freq="MS"):
    ideal = sim.simulate(sim.Window(start), "ideal")
    CUR.update(v=VARS[0], mode="base"); w = v2.build(start, src="r_q_nn", shape="cal")
    rows.append(dict(start=str(start.date()), variant="base_rerun", **sim.summarize(w, v2.simulate2(w, now="level", peak_off=True), ideal)))
    for v in VARS:
        for mode in ["4w", "3w"]:
            CUR.update(v=v, mode=mode); w = v2.build(start, src="r_q_nn", shape="cal")
            rows.append(dict(start=str(start.date()), variant=f"{v}_{mode}", **sim.summarize(w, v2.simulate2(w, now="level", peak_off=True), ideal)))
    print(start.date(), flush=True)
R = pd.DataFrame(rows); R.to_csv("data/sim_rd_windows.csv", index=False)
S0 = pd.read_csv("data/sim_systems_windows.csv"); S0 = S0[S0.policy == "production"]
base = S0[S0.system == "r_q_nn"].set_index("start").loss_pct
piv = R.pivot(index="start", columns="variant", values="loss_pct"); piv["three-way"] = base.reindex(piv.index).values
print(piv.round(4).to_string()); print("mean loss:", piv.mean().round(4).to_dict())
print("windows better than three-way:", {c: int((piv[c] < piv["three-way"]).sum()) for c in piv.columns if c != "three-way"})
