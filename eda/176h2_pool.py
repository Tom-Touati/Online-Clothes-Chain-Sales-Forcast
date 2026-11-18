"""D186 (user: anything hierarchical): partial pooling of p2's item forecasts across the hierarchy. Each item's forecast ratio
z_i = log((p2_i + 1) / (4 x level_i + 1)) is shrunk toward its style's level-weighted mean ratio (siblings), or its product type's:
z_i' = (1 - lam) z_i + lam z_group, lam in 0.15 / 0.3 / 0.5; and a size-adaptive version lam_i = k / (k + level_i) (small colours pooled
more), k in 5 / 20. Group means use only the origin's own forecasts (no outcome), so this is time-honest by construction.
Standard table and bootstrap against p2."""
import sys, contextlib, io, importlib.util, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); G = lambda *xs: np.exp(np.mean([np.log(np.maximum(x, 0) + 0.5) for x in xs], axis=0)) - 0.5
qv = lambda f: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(key).q50.values
T["p2"] = G(T.rules.values, qv("data/q_backtest_fkdyn_dyn.parquet"), T.nn.values, qv("data/q_backtest_fktab_tabpfn_v2.parquet"))
T["style"] = T.style_color.str.split("_").str[0]; T["ptype"] = pd.read_parquet("data/items.parquet").set_index("style_color").product_type.reindex(T.style_color).values
L4 = 4 * T.level.clip(lower=0) + 1; T["zi"] = np.log((T.p2 + 1) / L4); T["w"] = T.level.clip(lower=0) + 1
for g in ["style", "ptype"]: T[f"z_{g}"] = (T.zi * T.w).groupby([T.origin, T[g]]).transform("sum") / T.w.groupby([T.origin, T[g]]).transform("sum")
COLS = ["p2"]
for g in ["style", "ptype"]:
    for lam in (0.15, 0.3, 0.5): c = f"pool_{g}_{int(lam * 100)}"; T[c] = L4 * np.exp((1 - lam) * T.zi + lam * T[f"z_{g}"]) - 1; COLS.append(c)
    for k in (5, 20): lam = k / (k + T.level.clip(lower=0)); c = f"pool_{g}_k{k}"; T[c] = L4 * np.exp((1 - lam) * T.zi + lam * T[f"z_{g}"]) - 1; COLS.append(c)
REC_ = [o for o in ORIGINS if o >= "2025-06"]
for nm, m in [("nine recent origins", T.origin.isin(REC_)), ("all 13 origins", np.ones(len(T), bool)), ("recent, 100 targets", T.origin.isin(REC_) & T.target)]:
    print(f"\n######## {nm}"); print(sl.table(T[m], COLS, D).round(4).to_string())
F_ = T[T.origin.isin(REC_)]; print("\n== bootstrap vs p2, recent"); [print(f"  {c:16s}", np.round(sl.boot(F_, c, "p2"), 4)) for c in COLS[1:]]
