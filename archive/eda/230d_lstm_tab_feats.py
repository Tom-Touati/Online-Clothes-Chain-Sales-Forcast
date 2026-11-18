"""Fork "fashion" (D406; user: "add all the other features to the head"): the tree member's full input set at every month-end, for the
LSTM's head (eda/230b_lstm_gpu.py FULL=1). The table T and the feature lists come from the tree script's preamble (eda/187_losses.py:
the union of its BASE and FULL sets, i.e. every input of the submitted tree member); categoricals as integer codes. Writes
data/lstm_tab_feats.parquet (origin, style_color + the features)."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
src = open("eda/187_losses.py", encoding="utf-8").read().split("for v in VARS:")[0]; sys.argv = ["x"]; exec(compile(src, "187", "exec"))
cols = list(dict.fromkeys(B_ + F_)); D = T[["origin", "style_color"] + cols].copy()
for c in cols:
    if not np.issubdtype(D[c].dtype, np.number): D[c] = pd.Categorical(D[c].astype(str)).codes.astype(np.float32)
D[cols] = D[cols].astype(np.float32); D.to_parquet("data/lstm_tab_feats.parquet", index=False)
print("rows", len(D), "| features", len(cols), "| origins", D.origin.min(), "..", D.origin.max(), "| missing share", round(float(D[cols].isna().mean().mean()), 3))
