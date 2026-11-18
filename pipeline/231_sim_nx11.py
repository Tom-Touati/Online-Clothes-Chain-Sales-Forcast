"""D431: the buying simulation of the submitted system (nx11 = nx9's numbers; user, 18 Nov: "run the buying simulation and add it to the
evalutaion", "and to the submission"). The last simulated version was the D293 system (nx_ecsig / nx_ecnsnn era, about 10.8%); the six-member
system with the routed launch tree, the recommender at a quarter weight, the look-alike correction and the campaign uplift was never simulated.
Machinery and policy unchanged (pipeline/87_sim_systems.py on pipeline/59_sim_v2.py / pipeline/47_decision_sim.py, D067 / D083 / D090): 11 rolling 3-month
windows 2025-03 .. 2026-01, orders at most every 7 days, 21-day lead, safety stock from the calibrated quantile-model IQR, tau 0.80, the
14-day level nowcast (alpha 0.5) off for decisions in September-December ('production'); loss = % of the perfect-foresight profit. Each day
the decision uses only the latest month-end forecast. Only the monthly forecast changes between systems:
  R3, rules, r_q_nn       pipeline/87's constructions (rerun here as a reproduction check against data/sim_systems_windows.csv)
  nx11_nocamp             the submitted backtest total (data/geo_nx5r_fash_x_preds.parquet, column geo: six-member weighted mean x the
                          corrections incl. the look-alike term) split by month like the workbook: geometric mean of the rules' and the
                          neural network's monthly shares (nn_backtest_preds_S13ecx6), renormalised
  nx11                    the same x the campaign uplift per month as in the workbook: exp(0.22 x x_h), x_h = change in deep-discount days
                          against the same month a year earlier / days in the month x the item's edikted.com share (data/campaign_feats.parquet)
Items or origins without a submitted backtest forecast fall back to pipeline/87's qmodel construction. The printed counts add two systems each and
include the pre-window origin (W0-2), which only the ratio nowcast reads: in this run all 509 fallback rows sit there, the same rows for every
system, and none at a decision origin, so they do not affect the production results (audit, D431). R3 and rules reproduce pipeline/87's stored runs
exactly; r_q_nn is rerun with the submitted NN (S13ecx6 instead of S13x6), not a reproduction. The campaign uplift's backtest uses the realised
campaign calendar as the plan (an upper bound on what the plan adds).
Runs at below-normal priority with 3 threads (the user's compute rule). Writes data/sim_nx11_windows.csv; stdout -> pipeline/sim_nx11_output.txt"""
import os, sys, ctypes
for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]: os.environ[v] = "3"
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)   # BELOW_NORMAL_PRIORITY_CLASS
except Exception: pass
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
exec(open("pipeline/87_sim_systems.py", encoding="utf-8").read().split("SYS = [")[0])   # v2, sim, Mo, RUL, month_matrix (pipeline/87's), FALL
v2.NN = pd.read_parquet("data/nn_backtest_preds_S13ecx6.parquet")                 # the submitted system's NN backtest (D272)
base87 = month_matrix
G = pd.read_parquet("data/geo_nx5r_fash_x_preds.parquet").set_index(["origin", "style_color"]).geo
CF = pd.read_parquet("data/campaign_feats.parquet").set_index(["origin", "style_color"])[["x_h1", "x_h2", "x_h3", "x_h4"]]
LIFT = 0.22
FB = {"nx11": [0, 0]}
def nx11(o, items, camp):
    lg = base87(o, items, "qmodel")
    tot = G.loc[o].reindex(items).values if o in G.index.get_level_values(0) else np.full(len(items), np.nan)
    ru = RUL[o].reindex(items).values if o in RUL else np.full((len(items), 4), np.nan)
    nn = v2.NN[v2.NN.origin == o].pivot(index="style_color", columns="h", values="q50").reindex(items).values
    sh = lambda m: np.clip(m / m.sum(axis=1, keepdims=True), 1e-4, None)
    with np.errstate(invalid="ignore", divide="ignore"):   # audit (D431): a rules row that is zero in all four months gives NaN, so a flat split, as in the workbook
        prof = np.exp((np.log(sh(np.maximum(ru, 0))) + np.log(sh(np.maximum(nn, 0) + 1e-9))) / 2); prof = prof / prof.sum(axis=1, keepdims=True)
    prof = np.where(np.isfinite(prof), prof, 0.25)
    out = tot[:, None] * prof
    if camp:
        x = CF.loc[o].reindex(items).values if o in CF.index.get_level_values(0) else np.zeros((len(items), 4))
        out = out * np.exp(LIFT * np.nan_to_num(x))
    miss = ~np.isfinite(out).all(axis=1); FB["nx11"][0] += int(miss.sum()); FB["nx11"][1] += len(items)
    return np.maximum(0, np.where(miss[:, None], lg, out))
def mm(o, items, src):
    if src == "nx11_nocamp": return nx11(o, items, False)
    if src == "nx11": return nx11(o, items, True)
    return base87(o, items, src)
v2.month_matrix = mm
SYS = ["R3", "rules", "r_q_nn", "nx11_nocamp", "nx11"]
rows = []
for start in pd.date_range("2025-03-01", "2026-01-01", freq="MS"):
    ideal = sim.simulate(sim.Window(start), "ideal")
    for s in SYS:
        w = v2.build(start, src=s, shape="cal")
        rows.append(dict(start=str(start.date()), system=s, policy="production", **sim.summarize(w, v2.simulate2(w, now="level", peak_off=True), ideal)))
    print(start.date(), "done", flush=True)
R = pd.DataFrame(rows); R.to_csv("data/sim_nx11_windows.csv", index=False)
print("fallback to the qmodel construction (item-origin rows):", {k: f"{a} of {b}" for k, (a, b) in (FALL | FB).items()})
piv = R.pivot(index="start", columns="system", values="loss_pct")[SYS]
print("\n== production policy | loss (% of perfect-foresight profit) by window"); print(piv.round(4).to_string())
print("mean loss:", piv.mean().round(4).to_dict())
for ref in ["R3", "rules", "r_q_nn", "nx11_nocamp"]: print(f"windows where nx11 beats {ref}:", int((piv.nx11 < piv[ref]).sum()), "of", len(piv))
cols = [c for c in ["fill", "left_units"] if c in R]; print("mean", cols, ":"); print(R.groupby("system")[cols].mean().reindex(SYS).round(4).T.to_string())
if os.path.exists("data/sim_systems_windows.csv"):                                   # reproduction check of pipeline/87's stored runs
    old = pd.read_csv("data/sim_systems_windows.csv"); old = old[old.policy == "production"].pivot(index="start", columns="system", values="loss_pct")
    for s in ["R3", "rules", "r_q_nn"]:
        if s in old: print(f"reproduction {s}: max |new - stored| over windows = {float((piv[s] - old[s].reindex(piv.index)).abs().max()):.6f}")
