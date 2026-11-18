"""Fork 2: which new-item tree to route inside nx5 (before corrections; D389h found the gain on the log-L1 tree, not on Huber).
nx5 members with the tree column replaced for items under 6 months by each variant; WAPE / R3 recent / held out / all 13 / early, new items."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
B = pd.read_parquet("data/dec_base_ecnsnnhubrq26.parquet"); B = B[B.y.notna()].reset_index(drop=True); K = pd.MultiIndex.from_frame(B[["origin", "style_color"]])
rd = lambda f: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).q50.reindex(K).values
new = B.age_m.fillna(99).values < 6; O = B.origin.values; y, R3 = B.y.values, B.R3.values
MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32", "rc"]; W = np.array([1, 1, 1, 1, 1, 0.25])
S = {"rec": O >= "2025-06", "hold": O >= "2025-10", "all": np.ones(len(B), bool), "early": O <= "2025-05"}
w = lambda f, m: np.abs(f - y)[m].sum() / np.abs(R3 - y)[m].sum()
for nm, f in [("nx5 (Huber)", None), ("huber_new5", "huber_new5"), ("huber_new3", "huber_new3"), ("l1log_new5", "l1log_new5"), ("l1log (ec_ns)", "l1log"), ("mean huber_new5 + l1log_new5", "mix")]:
    X = B[MEM].clip(lower=0).values.copy()
    if f == "mix": t = (rd("data/q_backtest_ls_huber_new5.parquet") + rd("data/q_backtest_ls_l1log_new5.parquet")) / 2
    elif f: t = rd(f"data/q_backtest_ls_{f}.parquet")
    if f: X[:, 1] = np.where(new, t, X[:, 1])
    F = (X * W).sum(1) / W.sum()
    print(f"{nm:30s}", " ".join(f"{k} {w(F, m):.4f}" for k, m in S.items()), f"| new rec {w(F, S['rec'] & new):.4f} new hold {w(F, S['hold'] & new):.4f}")
