"""D257: base table for re-fitting the D245 / D247 corrections on a new system: data/dec_base.parquet with the D198 member (qdyn) replaced
by data/q_backtest_nx_<tag>.parquet's median, F = arithmetic mean of the five members, spread / e / ae recomputed.
Usage: python eda/180d_newbase.py <tag>  -> data/dec_base_<tag>.parquet (then DEC=_<tag> for eda/179x2_bias.py and eda/179b_breakout.py)."""
import sys, numpy as np, pandas as pd
tag = sys.argv[1]; B = pd.read_parquet("data/dec_base.parquet"); MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]
X = pd.read_parquet(f"data/q_backtest_nx_{tag}.parquet").drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).q50
B["qdyn"] = X.reindex(pd.MultiIndex.from_frame(B[["origin", "style_color"]])).fillna(pd.Series(B.qdyn.values, index=pd.MultiIndex.from_frame(B[["origin", "style_color"]]))).values
B["F"] = B[MEM].clip(lower=0).mean(1); B["spread"] = np.log(B[MEM].clip(lower=1).max(1) / B[MEM].clip(lower=1).min(1))
B["e"] = np.log((B.y + 1) / (B.F + 1)); B["ae"] = (B.F - B.y).abs(); REC = B.origin >= "2025-06"
print(tag, "recent WAPE / R3 of F:", round(B[REC].ae.sum() / (B[REC].R3 - B[REC].y).abs().sum(), 4)); B.to_parquet(f"data/dec_base_{tag}.parquet", index=False)
