"""D354: the retrained recommender (Fork 2 D339, rc_deepfm with the minimum-history fallback) as a sixth member, corrections re-fitted
(data/geo_ecnsnnrcd2_x_preds.parquet), against the submitted system (data/geo_ecnsnn_x_preds.parquet): bootstrap recent / all 13, the four
early origins, within +/-20% and the item-equal GMRAE."""
import sys, contextlib, io, importlib.util, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color])
for c, f in [("sub", "data/geo_ecnsnn_x_preds.parquet"), ("rc", "data/geo_ecnsnnrcd2_x_preds.parquet")]: T[c] = pd.read_parquet(f).set_index(["origin", "style_color"]).geo.reindex(key).values
REC = T.origin >= "2025-06"; E = T.origin <= "2025-05"
for c in ["sub", "rc"]:
    e, e3 = (T[c] - T.y).abs(), (T.R3 - T.y).abs()
    print(c, "early 4:", round(e[E].sum() / e3[E].sum(), 4), "| within 20% recent:", round(float(((e <= 0.2 * T.y) & (T.y > 0))[REC].mean()), 4), "| GMRAE recent:", round(float(np.exp(np.mean(np.log((e[REC] + 1) / (e3[REC] + 1))))), 4))
print("bootstrap rc vs submitted: recent", np.round(sl.boot(T[REC], "rc", "sub"), 4), "| all 13", np.round(sl.boot(T, "rc", "sub"), 4))
