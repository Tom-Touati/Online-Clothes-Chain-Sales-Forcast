"""D362 part 2: the tuned Huber tree member (data/hpo_tree_best.json from eda/191_hpo_tree.py) refitted for judging and deployment:
  all 13 backtest origins (normal training)                  -> data/q_backtest_hpo_tree.parquet
  the same with last year's window removed (o - 12 +/- 1)     -> data/q_backtest_hpo_tree_noly.parquet
  the 2026-06 origin (training on every closed window)        -> data/q_backtest_hpo_tree_2026.parquet
Time-honest training o' + 4 <= o (asserted). Usage: python eda/193_hpo_tree_refit.py [threads=8]"""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
KW = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); TH = int(KW.get("threads", 8))
sys.argv = ["eda/180q_models.py"]; ns = {"__name__": "hpo"}
exec(compile(open("eda/180q_models.py", encoding="utf-8").read(), "eda/180q_models.py", "exec"), ns)
T, sets, P0 = ns["T"], ns["sets"], dict(ns["P"]); B_, F_ = sets("ec_ns")
J = json.load(open("data/hpo_tree_best.json")); P = dict(P0); P.update(J["params"]); P["n_jobs"] = TH; wb, delta = J["w_base"], J["huber_delta"]
def fit(o, noly):
    oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()]; te = T[T.origin == o]
    if noly: tr = tr[~tr.o_ord.between(oo - 13, oo - 11)]
    assert (tr.o_ord + 4 <= oo).all() and len(te)
    z = wb * lgb.LGBMRegressor(objective="huber", alpha=delta, **P).fit(tr[B_], tr.z).predict(te[B_]) + (1 - wb) * lgb.LGBMRegressor(objective="huber", alpha=delta, **P).fit(tr[F_], tr.z).predict(te[F_])
    return pd.DataFrame({"origin": o, "style_color": te.style_color.values, "q50": np.clip((4 * te.level.values + 1) * np.exp(z) - 1, 0, None)})
ORIG = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
for name, origins, noly in [("", ORIG, False), ("_noly", ORIG, True), ("_2026", ["2026-06"], False)]:
    pd.concat([fit(o, noly) for o in origins]).to_parquet(f"data/q_backtest_hpo_tree{name}.parquet", index=False); print("saved", name or "normal", flush=True)
