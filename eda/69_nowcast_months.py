"""Sensitivity of the seasonal nowcast switch (D083): which decision months have the nowcast off. Blend forecast (LightGBM + 6-seed NN)
with the calendar shape, 11 rolling windows, eda/59 functions. Also tau 0.75/0.80/0.85 for the best month set."""
import sys, numpy as np, pandas as pd
sys.argv = ["x", "nn=nn_backtest_preds_B13x6"]
import importlib.util; spec = importlib.util.spec_from_file_location("v2", "eda/59_sim_v2.py"); v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
sim = v2.sim
def run(w, months, tau=0.8):
    src = v2.simulate2.__code__  # reuse simulate2 with a patched month set
    return v2.simulate2(w, tau=tau, now="level", peak_off=True) if months is None else None
SETS = {"none (alpha 0.5 all year)": (), "Sep-Dec": (9, 10, 11, 12), "Oct-Dec": (10, 11, 12), "Sep-Nov": (9, 10, 11), "Aug-Dec": (8, 9, 10, 11, 12), "Nov-Dec": (11, 12), "all year (alpha 0)": tuple(range(1, 13))}
rows = []
for start in pd.date_range("2025-03-01", "2026-01-01", freq="MS"):
    w0 = sim.Window(start); ideal = sim.simulate(w0, "ideal"); w = v2.build(start, src="blend", shape="cal")
    for name, ms in SETS.items():
        for tau in ([0.75, 0.8, 0.85] if name == "Sep-Dec" else [0.8]):
            orig = v2.simulate2.__globals__.get("PEAK_MONTHS")
            v2.PEAK_MONTHS = ms
            r = v2.simulate2(w, tau=tau, now="level", peak_off=True)
            rows.append(dict(start=str(start.date()), months=name, tau=tau, **sim.summarize(w, r, ideal)))
    print(start.date(), flush=True)
R = pd.DataFrame(rows); R.to_csv("data/sim_nowcast_months.csv", index=False)
P = R.pivot_table(index="start", columns=["months", "tau"], values="loss_pct")
print(P.round(3).T.to_string()); print("\nmean loss:"); print(P.mean().round(4).to_string())
