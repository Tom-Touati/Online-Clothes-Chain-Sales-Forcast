"""D148 follow-up: the log-residual member against the best quantile model (lean + occasion timing, data/q_backtest_lean.parquet, D135).
Recent origins, WAPE / R3 and paired bootstrap against the lean three-way (rules x lean quantile x NN)."""
import sys, contextlib, io, importlib.util, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color])
G = lambda *xs: np.exp(np.mean([np.log(np.maximum(x, 0) + 0.5) for x in xs], axis=0)) - 0.5
T["lean"] = pd.read_parquet("data/q_backtest_lean.parquet").set_index(["origin", "style_color"]).reindex(key).q50.values
T["rd"] = pd.read_parquet("data/q_backtest_rd_log.parquet").set_index(["origin", "style_color"]).reindex(key).q50.values
T["lean3"] = G(T.rules, T.lean, T.nn); T["lean4"] = G(T.rules, T.lean, T.nn, T.rd); T["rd3"] = G(T.rules, T.rd, T.nn)
T["lean_rd5050"] = G(T.rules, G(T.lean, T.rd), T.nn)
N = {"r_q_nn": "rules+quantile+NN (production quantile)", "lean3": "rules+lean quantile+NN", "lean4": "rules+lean quantile+NN+resid-log (added)",
     "rd3": "rules+NN+resid-log (replaces quantile)", "lean_rd5050": "rules+NN+(lean quantile x resid-log) (shares the quantile slot)"}
REC_ = [o for o in ORIGINS if o >= "2025-06"]
for s, m in {"nine recent origins": T.origin.isin(REC_), "recent, 100 targets": T.origin.isin(REC_) & T.target, "all 13 origins": np.ones(len(T), bool)}.items():
    print(f"\n######## {s}"); print(sl.table(T[m], list(N), D, N).round(4).to_string())
F = T[T.origin.isin(REC_)]
print("\n== paired bootstrap, recent: WAPE(system) - WAPE(rules+lean quantile+NN)")
for c in ["lean4", "rd3", "lean_rd5050", "r_q_nn"]: print(f"   {N[c]:62s}", np.round(sl.boot(F, c, "lean3"), 4))
