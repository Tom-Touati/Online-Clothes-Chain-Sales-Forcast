"""Rolling 3-month windows of the buying simulation (eda/47_decision_sim.py) with the parameters trained on Mar-May 2025,
plus a decomposition of the model's loss: 'monthly oracle' uses the true monthly totals spread evenly over each month's
days (same spread and service level as the model), so model - monthly oracle = cost of monthly forecast error and
monthly oracle - daily oracle = cost of not knowing the timing within the month (campaign days, Black Friday)."""
import importlib.util, sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
spec = importlib.util.spec_from_file_location("sim", "eda/47_decision_sim.py"); sim = importlib.util.module_from_spec(spec)
sys.argv = ["x"]; spec.loader.exec_module(sim)
TAU, ALPHA, TAUN = 0.8, 0.5, 0.85          # trained on Mar-May 2025 (D067)


def monthly_oracle(w):
    """Replace each origin's daily forecast by the true month total / days in month (all months the window touches)."""
    mon = w.days.to_period("M")
    tot = {}
    for m in mon.unique():
        cols = pd.date_range(m.start_time, m.end_time)
        tot[m] = sim.DEM.loc[w.items, cols].sum(axis=1).values / m.days_in_month
    F = np.stack([tot[m] for m in mon], axis=1)
    for o in w.origins: w.F[o] = F.copy()
    return w


rows = []
for start in pd.date_range("2025-03-01", "2026-01-01", freq="MS"):
    w = sim.Window(start); ideal = sim.simulate(w, "ideal")
    for name, kw in [("model", dict(policy="model", tau=TAU, alpha=ALPHA)), ("naive", dict(policy="naive", tau=TAUN)),
                     ("model, no nowcast", dict(policy="model", tau=TAU, alpha=0.0))]:
        rows.append(dict(start=str(start.date()), policy=name, **sim.summarize(w, sim.simulate(w, **kw), ideal)))
    wo = monthly_oracle(sim.Window(start))
    rows.append(dict(start=str(start.date()), policy="monthly oracle", **sim.summarize(wo, sim.simulate(wo, "model", tau=TAU, alpha=0.0), ideal)))
    rows.append(dict(start=str(start.date()), policy="daily oracle", **sim.summarize(w, sim.simulate(w, "oracle", tau=TAU), ideal)))
    print(start.date(), "items", len(w.items), flush=True)
R = pd.DataFrame(rows)
R.to_csv("data/sim_rolling_windows.csv", index=False)
piv = R.pivot(index="start", columns="policy", values="loss_pct")[["model", "model, no nowcast", "naive", "monthly oracle", "daily oracle"]]
print("\n== loss as % of ideal profit, by window start (decisions over 3 months)")
print(piv.round(3).to_string())
print("\nmean:", piv.mean().round(3).to_dict())
print("model better than naive in", int((piv.model < piv.naive).sum()), "of", len(piv), "windows")
f = R.pivot(index="start", columns="policy", values="fill")[["model", "naive", "monthly oracle"]]
print("\n== fill rate"); print(f.round(3).to_string())
