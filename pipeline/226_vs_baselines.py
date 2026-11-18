"""D389v (user: "say the model improved significantly over R3, naive Seasonality and hand rules together"): the submitted system's backtest
(data/geo_nx5r_x_preds.parquet, time-honest) against the three simple baselines and their combination.
  R3        last three months x 4
  seasonal  R3's monthly level x the product type's like-for-like seasonal ratio per target month (D294's seasonal naive)
  rules     the hand-written planning rules (the rule agents' forecast at each origin)
  together  the plain average of the three
WAPE of each, the system's WAPE as a ratio to each, the share of items the system beats each on, with 95% intervals of the ratio from a
bootstrap over styles (400 draws) and over whole origins (2,000 draws). Recent nine, all 13 and the two focus windows."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
import os; SYS = os.environ.get("SYS", "nx5r")   # D422: SYS=nx5r_fash scores the system with the look-alike correction
G = pd.read_parquet(f"data/geo_{SYS}_x_preds.parquet")[["origin", "style_color", "geo", "y", "R3"]]; G["style"] = G.style_color.str.split("_").str[0]
B = pd.read_parquet("data/dec_base_nx5r26.parquet")[["origin", "style_color", "rules"]]; G = G.merge(B, on=["origin", "style_color"])
fe = pd.read_parquet("data/features.parquet"); fe["st"] = fe.m3 * np.exp(fe.lfl_type.fillna(0)); G = G.merge(fe.groupby(["origin", "style_color"]).st.sum().rename("sea").reset_index(), on=["origin", "style_color"], how="left")
G["sea"] = G.sea.fillna(G.R3); G["together"] = (G.R3 + G.sea + G.rules) / 3
y = G.y.values; f = G.geo.values; O = G.origin.values; rng = np.random.default_rng(0)
def read(m, lab):
    print(f"\n== {lab} (item-origins {m.sum()}): system WAPE {np.abs(f - y)[m].sum() / y[m].sum():.3f}")
    for nm, c in [("R3", "R3"), ("seasonal naive", "sea"), ("hand rules", "rules"), ("the three together", "together")]:
        b = G[c].values; a_, b_ = np.abs(f - y), np.abs(b - y)
        d = pd.DataFrame(dict(s=G["style"].values[m], o=O[m], a=a_[m], b=b_[m]))
        gs = d.groupby("s")[["a", "b"]].sum().values; i = rng.integers(0, len(gs), (400, len(gs))); rs = gs[i].sum(1); rs = rs[:, 0] / rs[:, 1]
        go = d.groupby("o")[["a", "b"]].sum().values; j = rng.integers(0, len(go), (2000, len(go))); ro = go[j].sum(1); ro = ro[:, 0] / ro[:, 1]
        print(f"   vs {nm:19s} baseline WAPE {b_[m].sum() / y[m].sum():.3f} | system / baseline {a_[m].sum() / b_[m].sum():.3f} | by style [{np.percentile(rs, 2.5):.3f}, {np.percentile(rs, 97.5):.3f}]"
              + (f" | by origin [{np.percentile(ro, 2.5):.3f}, {np.percentile(ro, 97.5):.3f}]" if len(go) > 1 else "") + f" | items won {np.mean(a_[m] < b_[m]):.0%} | origins won {int((go[:, 0] < go[:, 1]).sum())} of {len(go)}")
read(O >= "2025-06", "recent nine"); read(O > "0", "all 13"); read(O == "2025-06", "Jul-Oct 2025"); read(O == "2026-02", "Mar-Jun 2026")
