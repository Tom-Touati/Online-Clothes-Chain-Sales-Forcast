"""D431: the seasonal naive in the buying simulation, for the metrics table's buying-loss row (user: "add a table that shows the improvement of
each metric over R3, seasonal naive, and agent rule based"). Same machinery, policy and windows as pipeline/231_sim_nx11.py; the monthly forecast is
the seasonal naive of the scoreboard (D294): R3's monthly level x the product type's like-for-like seasonal ratio per target month
(data/features.parquet: m3 x exp(lfl_type), per horizon month). Appends its rows (system 'sea_naive') to data/sim_nx11_windows.csv.
Below-normal priority, 3 threads. stdout -> pipeline/sim_naive_output.txt"""
import os, sys, ctypes
for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]: os.environ[v] = "3"
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception: pass
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
exec(open("pipeline/87_sim_systems.py", encoding="utf-8").read().split("SYS = [")[0])
base87 = month_matrix
fe = pd.read_parquet("data/features.parquet", columns=["origin", "style_color", "h", "m3", "lfl_type"]); fe["st"] = fe.m3 * np.exp(fe.lfl_type.fillna(0))
SEA = {o: g.pivot(index="style_color", columns="h", values="st") for o, g in fe.groupby("origin")}
FB = [0, 0]
def mm(o, items, src):
    if src != "sea_naive": return base87(o, items, src)
    lg = base87(o, items, "qmodel"); x = SEA[o].reindex(items).reindex(columns=[1, 2, 3, 4]).values if o in SEA else np.full((len(items), 4), np.nan)
    miss = ~np.isfinite(x).all(axis=1); FB[0] += int(miss.sum()); FB[1] += len(items)
    return np.maximum(0, np.where(miss[:, None], lg, x))
v2.month_matrix = mm
rows = []
for start in pd.date_range("2025-03-01", "2026-01-01", freq="MS"):
    ideal = sim.simulate(sim.Window(start), "ideal"); w = v2.build(start, src="sea_naive", shape="cal")
    rows.append(dict(start=str(start.date()), system="sea_naive", policy="production", **sim.summarize(w, v2.simulate2(w, now="level", peak_off=True), ideal)))
    print(start.date(), "done", flush=True)
R = pd.DataFrame(rows); old = pd.read_csv("data/sim_nx11_windows.csv"); old = old[old.system != "sea_naive"]
pd.concat([old, R]).to_csv("data/sim_nx11_windows.csv", index=False)
print("fallback rows (qmodel construction; includes the pre-window origin):", FB)
print(R[["start", "loss_pct", "fill"]].round(4).to_string(index=False)); print("mean loss:", round(float(R.loss_pct.mean()), 4), "| fill:", round(float(R.fill.mean()), 4))
