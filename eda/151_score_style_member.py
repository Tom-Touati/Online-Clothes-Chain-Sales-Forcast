"""D149: the forward style-embedding network (eda/149) scored as an ensemble member, on the eda/86 items and origins.
Its 4-month forecast with the style terms (semb) and without them (semb_ns, same network, style terms zeroed); alone, replacing
the quantile model, added as a fourth member, and sharing the quantile slot; against the production and the lean three-way.
Standard R3-relative table (eda/score_lib.py) and paired bootstrap against the lean three-way."""
import sys, contextlib, io, importlib.util, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color])
G = lambda *xs: np.exp(np.mean([np.log(np.maximum(x, 0) + 0.5) for x in xs], axis=0)) - 0.5
SE = pd.read_parquet("data/style_emb_feats.parquet").set_index(["origin", "style_color"]).reindex(key)
T["semb"] = SE.pred4.clip(lower=0).values; T["semb_ns"] = SE.pred4_ns.clip(lower=0).values
print("items without a network forecast:", int(T.semb.isna().sum()))
T["semb"] = T.semb.fillna(T.qmodel); T["semb_ns"] = T.semb_ns.fillna(T.qmodel)
T["lean"] = pd.read_parquet("data/q_backtest_lean.parquet").set_index(["origin", "style_color"]).reindex(key).q50.values
T["lean3"] = G(T.rules, T.lean, T.nn)
for c in ["semb", "semb_ns"]:
    T[c + "_3w"] = G(T.rules, T[c], T.nn); T[c + "_4w"] = G(T.rules, T.lean, T.nn, T[c]); T[c + "_sh"] = G(T.rules, G(T.lean, T[c]), T.nn)
N = {"qmodel": "production quantile alone", "lean": "lean quantile alone", "semb": "style-embedding net alone", "semb_ns": "same net, style terms zeroed",
     "r_q_nn": "rules+quantile+NN (production)", "lean3": "rules+lean quantile+NN",
     "semb_3w": "rules+style net+NN (replaces quantile)", "semb_4w": "rules+lean+NN+style net (added)", "semb_sh": "rules+(lean x style net)+NN (shares slot)",
     "semb_ns_4w": "rules+lean+NN+net without style (added)", "semb_ns_sh": "rules+(lean x net without style)+NN (shares slot)"}
REC_ = [o for o in ORIGINS if o >= "2025-06"]
for s, m in {"nine recent origins": T.origin.isin(REC_), "recent, 100 targets": T.origin.isin(REC_) & T.target, "all 13 origins": np.ones(len(T), bool)}.items():
    print(f"\n######## {s}"); print(sl.table(T[m], list(N), D, N).round(4).to_string())
F = T[T.origin.isin(REC_)]
print("\n== paired bootstrap, recent: WAPE(system) - WAPE(rules+lean quantile+NN)")
for c in ["semb_3w", "semb_4w", "semb_sh", "semb_ns_4w", "semb_ns_sh"]: print(f"   {N[c]:52s}", np.round(sl.boot(F, c, "lean3"), 4))
print("   style net alone - lean alone                        ", np.round(sl.boot(F, "semb", "lean"), 4))
print("   style net alone - same net without style terms      ", np.round(sl.boot(F, "semb", "semb_ns"), 4))
