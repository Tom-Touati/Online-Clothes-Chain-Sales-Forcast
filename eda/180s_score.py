"""D253 scorer: data/q_backtest_nx_<variant>.parquet medians (a) alone, against the base refit and the stored D198 member (qdyn),
and (b) inside the final system p2c32a (arithmetic mean of rules, D198 quantile, NN, conformal TabPFN, TabPFN dyn 32) in place of
qdyn. Standard R3-relative table on the nine recent origins, all 13, the Black Friday origins (Jul-Oct 2025) and the 100 targets;
bootstrap of each against its reference (recent). Also: seen vs unseen styles for base and styleout.
Usage: python eda/180s_score.py [variants...] (default: every nx file)."""
import sys, glob, contextlib, io, importlib.util, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
VS = sys.argv[1:] or [f.replace("\\", "/").split("q_backtest_nx_")[1][:-8] for f in sorted(glob.glob("data/q_backtest_nx_*.parquet"))]
sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color])
qv = lambda f, c="q50": pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(key)[c].values
AM = lambda *xs: np.nanmean(np.column_stack([np.clip(x, 0, None) for x in xs]), axis=1)
qd = qv("data/q_backtest_fkdyn_dyn.parquet"); cf = qv("data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet"); t2 = qv("data/q_backtest_fktab_tabpfn_v2_dyn_e32.parquet")
T["qdyn"] = qd; T["p2c32a"] = AM(T.rules.values, qd, T.nn.values, cf, t2); COLS_A = ["qdyn"]; COLS_B = ["p2c32a"]
for v in VS:
    x = qv(f"data/q_backtest_nx_{v}.parquet"); x = np.where(np.isfinite(x), x, qd); T[v] = x; T["E:" + v] = AM(T.rules.values, x, T.nn.values, cf, t2)
    COLS_A.append(v); COLS_B.append("E:" + v)
REC_ = [o for o in ORIGINS if o >= "2025-06"]; BF = T.origin.isin(["2025-07", "2025-08", "2025-09", "2025-10"])
for nm, m in [("nine recent origins", T.origin.isin(REC_)), ("all 13 origins", np.ones(len(T), bool)), ("Black Friday origins", BF), ("recent, 100 targets", T.origin.isin(REC_) & T.target)]:
    print(f"\n######## {nm}: (a) member alone"); print(sl.table(T[m], COLS_A, D).round(4).to_string())
    print(f"######## {nm}: (b) in the final system in place of qdyn"); print(sl.table(T[m], COLS_B, D).round(4).to_string())
F_ = T[T.origin.isin(REC_)]; ref = "base" if "base" in VS else "qdyn"
print(f"\n== bootstrap, recent: member vs {ref} | in the system vs p2c32a")
for v in VS:
    if v == ref: continue
    print(f"  {v:12s}", np.round(sl.boot(F_, v, ref), 4), "|", np.round(sl.boot(F_, "E:" + v, "p2c32a"), 4))
FB = T[BF]; print("\n== bootstrap, Black Friday origins, in the system vs p2c32a:"); [print(f"  {v:12s}", np.round(sl.boot(FB, 'E:' + v, 'p2c32a'), 4)) for v in VS if v != ref]
for v in [x for x in ("base", "styleout") if x in VS]:
    S = pd.read_parquet(f"data/q_backtest_nx_{v}.parquet").drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(key)
    R = T.assign(seen=S.seen_style.values)[T.origin.isin(REC_)]
    for s, g in R.groupby("seen"): print(f"  {v:9s} {'seen' if s else 'unseen'} styles: n {len(g):5d}, WAPE / R3 {(g[v] - g.y).abs().sum() / (g.R3 - g.y).abs().sum():.4f}, bias {g[v].sum() / g.y.sum() - 1:+.3f}")
