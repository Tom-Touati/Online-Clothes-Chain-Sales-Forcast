"""Fork 2: the D248 geometric-mean correction (segment re-allocation x breakout model, each origin's total kept) on the routed nx5 base,
as eda/180g_geo.py's geo(); validated by reproducing nx5's stored data/geo_ecnsnnhubrq_x_2026.parquet from its own inputs. Writes
data/geo_nx5r_x_2026.parquet (2026-06 rows) and data/geo_nx5r_x_preds.parquet (backtest rows), and scores the backtest."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
def geo(base, seg, brk):
    B = pd.read_parquet(base)[["origin", "style_color", "F", "y", "R3", "age_m", "target"]]
    S = pd.read_parquet(seg)[["origin", "style_color", "vol_age_disc_0.5n"]]; K = pd.read_parquet(brk)[["origin", "style_color", "res_all_0.5n"]]
    B = B.merge(S, on=["origin", "style_color"]).merge(K, on=["origin", "style_color"]); B["g"] = np.sqrt(B["vol_age_disc_0.5n"].clip(lower=0) * B["res_all_0.5n"].clip(lower=0))
    B["geo"] = B.g * B.groupby("origin").F.transform("sum") / B.groupby("origin").g.transform("sum"); return B
A = geo("data/dec_base_ecnsnnhubrq26.parquet", "data/bias_seg_preds_ecnsnnhubrq26.parquet", "data/breakout_preds_ecnsnnhubrq26_x.parquet")
ref = pd.read_parquet("data/geo_ecnsnnhubrq_x_2026.parquet").set_index("style_color")
a26 = A[A.origin == "2026-06"].set_index("style_color"); print("reproduces nx5's 2026 geo:", float(np.nanmax(np.abs(a26.geo.reindex(ref.index) / ref.geo - 1))))
G = geo("data/dec_base_nx5r26.parquet", "data/bias_seg_preds_nx5r26.parquet", "data/breakout_preds_nx5r26_x.parquet")
bt = G[G.y.notna()]; w = lambda d, c, m: (d[c] - d.y).abs()[m].sum() / (d.R3 - d.y).abs()[m].sum()
Ab = A[A.y.notna()]
for nm, d in [("nx5 (D361)", Ab), ("nx5 routed", bt)]:
    O = d.origin; print(f"{nm:12s} corrected: recent {w(d, 'geo', O >= '2025-06'):.4f} held out {w(d, 'geo', O >= '2025-10'):.4f} all 13 {w(d, 'geo', O > '0'):.4f} early {w(d, 'geo', O <= '2025-05'):.4f}"
          f" | new items recent {w(d, 'geo', (O >= '2025-06') & (d.age_m.fillna(99) < 6)):.4f} | targets recent {w(d, 'geo', (O >= '2025-06') & d.target.astype(bool)):.4f}")
bt.to_parquet("data/geo_nx5r_x_preds.parquet", index=False)
g26 = G[G.origin == "2026-06"][["style_color", "F", "geo"]]; g26.to_parquet("data/geo_nx5r_x_2026.parquet", index=False)
print("multipliers 10/50/90%:", np.round(np.percentile((g26.geo / g26.F).dropna(), [10, 50, 90]), 3))
