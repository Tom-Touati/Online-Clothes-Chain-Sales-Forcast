"""D257: the D248 geometric-mean correction (segment re-allocation D245 x breakout model D247, each origin total kept) re-fitted on a new
base system (data/dec_base_<tag>.parquet; corrections from DEC=_<tag> runs of eda/179x2_bias.py and eda/179b_breakout.py).
Scored against p2c32a and p2c32a_geo (D248). Usage: python eda/180g_geo.py <tag>; writes data/geo_<tag>_preds.parquet."""
import sys, contextlib, io, importlib.util, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220); tag = sys.argv[1]; sys.argv = sys.argv[:1]
def geo(suf):
    B = pd.read_parquet(f"data/dec_base{suf}.parquet")[["origin", "style_color", "F"]]
    S = pd.read_parquet(f"data/bias_seg_preds{suf}.parquet")[["origin", "style_color", "vol_age_disc_0.5n"]]; K = pd.read_parquet(f"data/breakout_preds{suf}.parquet")[["origin", "style_color", "res_all_0.5n"]]
    B = B.merge(S, on=["origin", "style_color"]).merge(K, on=["origin", "style_color"]); B["g"] = np.sqrt(B["vol_age_disc_0.5n"].clip(lower=0) * B["res_all_0.5n"].clip(lower=0))
    B["geo"] = B.g * B.groupby("origin").F.transform("sum") / B.groupby("origin").g.transform("sum"); return B
G0, G1 = geo(""), geo(f"_{tag}")
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); ix = lambda D, c: D.set_index(["origin", "style_color"])[c].reindex(key).values
T["p2c32a"] = ix(G0, "F"); T["p2c32a_geo"] = ix(G0, "geo"); T[f"E:{tag}"] = ix(G1, "F"); T[f"E:{tag}_seg"] = ix(G1, "vol_age_disc_0.5n"); T[f"E:{tag}_brk"] = ix(G1, "res_all_0.5n"); T[f"E:{tag}_geo"] = ix(G1, "geo")
COLS = ["p2c32a", "p2c32a_geo", f"E:{tag}", f"E:{tag}_seg", f"E:{tag}_brk", f"E:{tag}_geo"]; REC_ = [o for o in ORIGINS if o >= "2025-06"]
for nm, m in [("nine recent origins", T.origin.isin(REC_)), ("all 13 origins", np.ones(len(T), bool)), ("Black Friday origins", T.origin.isin(["2025-07", "2025-08", "2025-09", "2025-10"])), ("recent, 100 targets", T.origin.isin(REC_) & T.target)]:
    print(f"\n######## {nm}"); print(sl.table(T[m], COLS, D).round(4).to_string())
F_ = T[T.origin.isin(REC_)]
for c in COLS[1:]: print(f"  {c:16s} vs p2c32a", np.round(sl.boot(F_, c, "p2c32a"), 4), "| vs p2c32a_geo", np.round(sl.boot(F_, c, "p2c32a_geo"), 4))
G1.to_parquet(f"data/geo_{tag}_preds.parquet", index=False)
