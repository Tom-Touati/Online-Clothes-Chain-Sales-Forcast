"""Score the D148 residual-distribution members (data/q_backtest_rd_*.parquet, eda/144) on the eda/86 items and origins.
Standard R3-relative table (eda/score_lib.py; R3 + rules row; pinball with the stored quantile model's spread around each centre),
for each variant: alone (sample median, sample mean), replacing the quantile model in the three-way, and ADDED as a fourth member
(geometric mean of rules, quantile, NN and the variant). Also each variant's own pinball and coverage from its sampled quantiles,
against the stored quantile model's own quantiles. Paired item bootstrap against the three-way. Sim column from data/sim_rd_windows.csv if present."""
import sys, glob, contextlib, io, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
src = open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src)
import importlib.util
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color])
G = lambda *xs: np.exp(np.mean([np.log(np.maximum(x, 0) + 0.5) for x in xs], axis=0)) - 0.5
V = {p.replace("\\", "/").split("rd_")[-1][:-8]: pd.read_parquet(p).set_index(["origin", "style_color"]).reindex(key) for p in sorted(glob.glob("data/q_backtest_rd_*.parquet"))}
QS = ["q10", "q25", "q50", "q75", "q90"]; TAU = [0.1, 0.25, 0.5, 0.75, 0.9]
cols, names = ["r_q_nn"], {"r_q_nn": "rules+quantile+NN (reference)", "qmodel": "quantile model alone"}
cols.insert(0, "qmodel")
for v, q in V.items():
    assert q.q50.notna().all(), v
    T[f"{v}_med"] = q.q50.values; T[f"{v}_mean"] = q["mean"].values
    T[f"{v}_3w"] = G(T.rules.values, T[f"{v}_med"].values, T.nn.values)
    T[f"{v}_4w"] = G(T.rules.values, T.qmodel.values, T.nn.values, T[f"{v}_med"].values)
    T[f"{v}_4wm"] = G(T.rules.values, T.qmodel.values, T.nn.values, T[f"{v}_mean"].values)
    cols += [f"{v}_med", f"{v}_mean", f"{v}_3w", f"{v}_4w", f"{v}_4wm"]
    names |= {f"{v}_med": f"resid-{v} alone (sample median)", f"{v}_mean": f"resid-{v} alone (sample mean)",
              f"{v}_3w": f"rules+NN+resid-{v} (replaces quantile)", f"{v}_4w": f"rules+quantile+NN+resid-{v} (added)",
              f"{v}_4wm": f"rules+quantile+NN+resid-{v} mean (added)"}
try:
    S = pd.read_csv("data/sim_rd_windows.csv"); S0 = pd.read_csv("data/sim_systems_windows.csv"); S0 = S0[S0.policy == "production"].groupby("system").loss_pct.mean()
    SIM = (S.groupby("variant").loss_pct.mean() / S0["R3"]).to_dict() | {"r_q_nn": S0["r_q_nn"] / S0["R3"], "qmodel": S0["qmodel"] / S0["R3"], "rules": S0["rules"] / S0["R3"]}
except FileNotFoundError: SIM = None
QST = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"]).reindex(key)


def native(F, Q):
    i = F.index.values; y = F.y.values; tot = 0
    for c, tau in zip(QS, TAU): u = y - Q[c].values[i]; tot += np.maximum(tau * u, (tau - 1) * u).sum()
    return tot / (len(QS) * y.sum()), ((y >= Q.q10.values[i]) & (y <= Q.q90.values[i])).mean(), ((y >= Q.q25.values[i]) & (y <= Q.q75.values[i])).mean()


REC_ = [o for o in ORIGINS if o >= "2025-06"]
SUB = {"nine recent origins, all items": T.origin.isin(REC_), "nine recent, the 100 targets": T.origin.isin(REC_) & T.target, "all 13 origins": np.ones(len(T), bool)}
for s, m in SUB.items():
    F = T[m]; print(f"\n######## {s}")
    print(sl.table(F, cols, D, names, sim=SIM if s.startswith("nine recent origins") else None).round(4).to_string())
    p0 = native(F, QST); print(f"   own quantiles: stored quantile model pinball {p0[0]:.4f} | P10-P90 cov {p0[1]:.3f} | P25-P75 cov {p0[2]:.3f}")
    for v, q in V.items():
        p = native(F, q); print(f"   own quantiles: resid-{v:5s} pinball {p[0]:.4f} ({p[0] / p0[0]:.3f}x) | P10-P90 cov {p[1]:.3f} | P25-P75 cov {p[2]:.3f}")
F = T[SUB["nine recent origins, all items"]]
print("\n== paired bootstrap, recent origins: WAPE(system) - WAPE(rules+quantile+NN), 95% interval")
for c in cols[2:]: print(f"   {names[c]:48s}", np.round(sl.boot(F, c, "r_q_nn"), 4))
