"""D258: which D244 error segments did the new system clear? Recent origins; WAPE / R3 and bias per segment for
p2c32a (F), p2c32a_geo (D248), E:ec_ns (D253 member in the system) and E:ec_ns_geo (D257). Writes eda/seg_compare_output.txt (stdout)."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
B = pd.read_parquet("data/dec_base.parquet"); B = B[B.origin >= "2025-06"].copy()
g0 = pd.read_parquet("data/geo_ec_ns_preds.parquet").rename(columns={"F": "ecns", "geo": "ecns_geo"})[["origin", "style_color", "ecns", "ecns_geo"]]
S = pd.read_parquet("data/bias_seg_preds.parquet")[["origin", "style_color", "vol_age_disc_0.5n"]]; K = pd.read_parquet("data/breakout_preds.parquet")[["origin", "style_color", "res_all_0.5n"]]
B = B.merge(g0, on=["origin", "style_color"]).merge(S, on=["origin", "style_color"]).merge(K, on=["origin", "style_color"])
B["geo"] = np.sqrt(B["vol_age_disc_0.5n"] * B["res_all_0.5n"]); B["geo"] *= B.groupby("origin").F.transform("sum") / B.groupby("origin").geo.transform("sum")
SY = {"F": "p2c32a", "geo": "p2c32a_geo", "ecns": "E:ec_ns", "ecns_geo": "E:ec_ns_geo"}
B["age_band"] = pd.cut(B.age_m, [-1, 3, 6, 12, 999], labels=["<3m", "3-6m", "6-12m", "12m+"])
B["vol_band"] = B.groupby("origin").R3.transform(lambda v: pd.qcut(np.log1p(v), 5, labels=False, duplicates="drop")).map(lambda k: f"v{int(k) + 1}")
B["phase"] = pd.cut(B.trend91, [-99, -0.4, -0.1, 0.1, 0.4, 99], labels=["falling fast", "falling", "steady", "rising", "rising fast"])
B["bf"] = np.where(B.bf_window == 1, "Black Friday window", "other windows"); B["kind"] = B.macro
rows = []
for seg in ["age_band", "vol_band", "phase", "kind", "bf"]:
    for v, d in B.groupby(seg, observed=True):
        r = {"segment": seg, "value": v, "n": len(d), "unit_share": round(d.y.sum() / B.y.sum(), 3)}
        for c, nm in SY.items(): r[nm] = round((d[c] - d.y).abs().sum() / (d.R3 - d.y).abs().sum(), 4); r[nm + " bias"] = round(d[c].sum() / d.y.sum() - 1, 3)
        rows.append(r)
R = pd.DataFrame(rows); R["change"] = R["E:ec_ns_geo"] - R["p2c32a"]
print(R[["segment", "value", "n", "unit_share", "p2c32a", "p2c32a_geo", "E:ec_ns", "E:ec_ns_geo", "change", "p2c32a bias", "E:ec_ns_geo bias"]].to_string(index=False))
