"""D293: bootstrap of the D307 tree-member variants in the corrected system against the submitted system (E:ecnsnn_x_geo, D273), recent and all 13 origins."""
import sys, contextlib, io, importlib.util, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); REC_ = [o for o in ORIGINS if o >= "2025-06"]
T["sub"] = pd.read_parquet("data/geo_ecnsnn_x_preds.parquet").set_index(["origin", "style_color"]).geo.reindex(key).values
for t in ["sig", "lm", "siglm"]:
    T[t] = pd.read_parquet(f"data/geo_ecnsnn{t}_x_preds.parquet").set_index(["origin", "style_color"]).geo.reindex(key).values
    h = lambda c, m: ((np.abs(T[c] - T.y) <= 0.2 * T.y) & (T.y > 0))[m].mean()
    m = T.origin.isin(REC_)
    print(f"{t:6s} vs submitted: recent {np.round(sl.boot(T[m], t, 'sub'), 4)} | all 13 {np.round(sl.boot(T, t, 'sub'), 4)} | within 20% recent {h(t, m):.3f} vs {h('sub', m):.3f}")
