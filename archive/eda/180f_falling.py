"""D259: falling-fast items (trend91 < -0.4; D244 WAPE / R3 0.57, unchanged by D253-D257). Final new system E:ec_ns_geo, recent origins.
Split of their error into under-forecast (rebound: actual above forecast) and over-forecast (kept falling), and Spearman of the log residual
with candidate signals known at the origin (D252 blocks, D252b promo, seasonality, discount, member spread). Writes eda/falling_output.txt."""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 200)
B = pd.read_parquet("data/dec_base_ec_ns.parquet"); B = B[B.origin >= "2025-06"]
G = pd.read_parquet("data/geo_ec_ns_preds.parquet")[["origin", "style_color", "geo"]]; B = B.merge(G, on=["origin", "style_color"])
for f in ["data/nf2_feats.parquet", "data/nf3_feats.parquet"]: B = B.merge(pd.read_parquet(f).drop(columns=["age_m"], errors="ignore"), on=["origin", "style_color"], how="left")
Fe = pd.read_parquet("data/features.parquet"); Fe = Fe[Fe.h == 1][["origin", "style_color", "lagr1", "lagr2", "lagr3", "f_r7", "peak_r", "disc_rel"]].drop_duplicates(["origin", "style_color"])
B = B.merge(Fe, on=["origin", "style_color"], how="left"); B["e2"] = np.log((B.y + 1) / (B.geo + 1)); B["ae2"] = (B.geo - B.y).abs()
FF = B[B.trend91 < -0.4]; E = B.ae2.sum()
print(f"falling fast: {len(FF)} item-origins, {FF.y.sum() / B.y.sum():.1%} of units, {FF.ae2.sum() / E:.1%} of error; WAPE / R3 {FF.ae2.sum() / (FF.R3 - FF.y).abs().sum():.4f}")
u = FF.y > FF.geo; print(f"  under-forecast (rebounded): {u.mean():.1%} of items, {FF.ae2[u].sum() / FF.ae2.sum():.1%} of their error | over (kept falling): {FF.ae2[~u].sum() / FF.ae2.sum():.1%}")
print("  by age:", FF.groupby(pd.cut(FF.age_m, [-1, 3, 6, 12, 999]), observed=True).apply(lambda d: f"n {len(d)}, err share {d.ae2.sum() / FF.ae2.sum():.2f}, bias {d.geo.sum() / d.y.sum() - 1:+.2f}").to_dict())
print("  by Black Friday window:", FF.groupby("bf_window").apply(lambda d: f"n {len(d)}, err share {d.ae2.sum() / FF.ae2.sum():.2f}, bias {d.geo.sum() / d.y.sum() - 1:+.2f}").to_dict())
S = ["lagr1", "lagr2", "lagr3", "f_r7", "f_r28", "peak_r", "own_sea", "disc_item", "disc_rel", "spread", "age_m", "curve_fwd", "curve_dev", "pk_life_days", "pk_365", "pk_365_days", "pk_anniv_dist",
     "tr_365", "tr_365_days", "pk_28", "anniv_dist", "tr_ptype_mom", "tr_ptype_shr", "item_vs_type", "promo_lift", "bfcm_ly", "wk_end_rel", "sh_tt"]
W = FF[(FF.y >= 20) | (FF.geo >= 20)]
r = {c: spearmanr(W[c], W.e2, nan_policy="omit")[0] for c in S if c in W}
print(f"\n== Spearman with the residual log((y + 1) / (forecast + 1)), falling fast, n {len(W)}")
print("  " + " | ".join(f"{k} {v:+.3f}" for k, v in sorted(r.items(), key=lambda kv: -abs(kv[1]))))
