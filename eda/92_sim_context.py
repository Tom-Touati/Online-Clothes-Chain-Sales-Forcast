"""Buying-simulation loss for the D112 context-feature variants (production policy, 11 windows, as eda/87). Only the forecast centre
changes: the quantile-model construction (LightGBM monthly split scaled to the variant's P50) and the three-way combination built on it;
the safety-stock spread stays the stored production model's IQR, so losses compare directly with R3 and R3 + rules in
data/sim_systems_windows.csv. Writes data/sim_ctx_windows.csv."""
import sys, glob, numpy as np, pandas as pd
PFX = sys.argv[1] if len(sys.argv) > 1 else "ctx"          # ctx = D112 context features, mrl = D113 Matryoshka embeddings
sys.stdout.reconfigure(encoding="utf-8")
src = open("eda/87_sim_systems.py", encoding="utf-8").read().split("SYS = [")[0]
exec(src)
base_mm = month_matrix
VQ = {p.replace("\\", "/").split(PFX + "_")[-1][:-8]: pd.read_parquet(p).set_index(["origin", "style_color"]).q50 for p in sorted(glob.glob(f"data/q_backtest_{PFX}_*.parquet"))}
CUR = {"v": None}
def month_matrix_v(o, items, src):
    b = sim.BT[sim.BT.origin == o].pivot(index="style_color", columns="h", values="pred").reindex(items)
    q50 = VQ[CUR["v"]].loc[o].reindex(items).values
    lg = np.nan_to_num(b.values * (q50 / b.sum(axis=1).values)[:, None])
    if src == "qmodel": return lg
    nn = v2.NN[v2.NN.origin == o].pivot(index="style_color", columns="h", values="q50").reindex(items).values; nn = np.where(np.isnan(nn), lg, nn)
    ru = RUL[o].reindex(items).values; ru = np.where(np.isnan(ru), lg, ru)
    return np.maximum(0, g(ru, lg, nn))
v2.month_matrix = month_matrix_v
rows = []
for start in pd.date_range("2025-03-01", "2026-01-01", freq="MS"):
    ideal = sim.simulate(sim.Window(start), "ideal")
    for v in VQ:
        CUR["v"] = v
        for s in ["qmodel", "r_q_nn"]:
            w = v2.build(start, src=s, shape="cal")
            rows.append(dict(start=str(start.date()), variant=v, system=s, **sim.summarize(w, v2.simulate2(w, now="level", peak_off=True), ideal)))
    print(start.date(), flush=True)
R = pd.DataFrame(rows); R.to_csv(f"data/sim_{PFX}_windows.csv", index=False)
print(R.pivot_table(index=["system", "variant"], values="loss_pct", aggfunc="mean").round(4).to_string())
