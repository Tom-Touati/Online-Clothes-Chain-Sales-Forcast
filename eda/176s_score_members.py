"""D182 (overnight M5 goal): score candidate members whose files may not cover every item-origin (neural models data/q_backtest_nf_*,
weekly-origin models data/q_backtest_wk_*, augmentation data/q_backtest_aug_*, Trends / content data/q_backtest_gt_*, ...).
Usage: python eda/176s_score_members.py <glob pattern under data/> [...]. Missing item-origins are filled with the D159 quantile
model (coverage printed). Rows: each member alone; p2 (rules, D198 quantile, NN, TabPFN v2); p2 + member as a fifth member;
p2 with the member replacing the D198 quantile model. Standard R3-relative table; bootstrap of each against p2 (recent)."""
import sys, glob, contextlib, io, importlib.util, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
PAT = sys.argv[1:]; sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); G = lambda *xs: np.exp(np.mean([np.log(np.maximum(x, 0) + 0.5) for x in xs], axis=0)) - 0.5
qv = lambda f: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(key).q50.values
d159 = qv("data/q_backtest_wx_cpllm_we.parquet"); qd = qv("data/q_backtest_fkdyn_dyn.parquet"); t2 = qv("data/q_backtest_fktab_tabpfn_v2.parquet")
T["p2"] = G(T.rules.values, qd, T.nn.values, t2); COLS = ["p2"]; NM = {"p2": "p2 (rules, D198 q, NN, TabPFN v2)"}
for pat in PAT:
    for f in sorted(glob.glob(f"data/{pat}")):
        tag = f.replace("\\", "/").split("/")[-1].replace("q_backtest_", "").replace(".parquet", ""); v = qv(f)
        cov = np.isfinite(v).mean(); v = np.where(np.isfinite(v), v, d159); T[tag] = v
        T[tag + "+p2"] = G(T.rules.values, qd, T.nn.values, t2, v); T[tag + "@q"] = G(T.rules.values, v, T.nn.values, t2)
        COLS += [tag, tag + "+p2", tag + "@q"]; NM |= {tag: f"{tag} alone (coverage {cov:.3f})", tag + "+p2": f"p2 + {tag} (5 members)", tag + "@q": f"p2 with {tag} instead of D198 q"}
REC_ = [o for o in ORIGINS if o >= "2025-06"]
for nm, m in [("nine recent origins", T.origin.isin(REC_)), ("all 13 origins", np.ones(len(T), bool)), ("recent, 100 targets", T.origin.isin(REC_) & T.target)]:
    print(f"\n######## {nm}"); print(sl.table(T[m], COLS, D, NM).round(4).to_string())
F_ = T[T.origin.isin(REC_)]; print("\n== bootstrap vs p2, recent"); [print(f"  {NM[c]:55s}", np.round(sl.boot(F_, c, "p2"), 4)) for c in COLS[1:]]
