"""D202 (fork): the explanation layer applied to the Jul-Oct 2026 deliverable. At the June 2026 origin the D159 quantile model's median
(trained on every outcome known by then; SHAP from eda/188 ... 2026-06 → data/fk_shap_2026-06.parquet) is split into the categoriser's
categories (data/fk_categories.json, primary map). For each of the 100 targets: R3, the model's median, its deviation from R3 and each
category's contribution (log units; exp - 1 as %), the three largest drivers in words. Portfolio view: the R3-weighted mean contribution
of each category over the targets. Writes data/fk_explain_2026_targets.csv and eda/explain_2026_output.txt (stdout)."""
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
S = pd.read_parquet("data/fk_shap_2026-06.parquet"); CAT = json.load(open("data/fk_categories.json")); FM = CAT["feature_map"]
NAME = {c["id"]: c["name"] for c in CAT["categories"]}; EXP = [c["id"] for c in CAT["categories"] if c["kind"] == "explained"]
it = pd.read_parquet("data/items.parquet").set_index("style_color")
S = S[S.target.astype(bool)].copy(); F = [c[5:] for c in S.columns if c.startswith("shap_")]
S["F"] = (4 * S.level + 1) * np.exp(S.z_hat) - 1; S["d_hat"] = np.log((S.F + 1) / (S.R3 + 1)); S["offset"] = np.log((4 * S.level + 1) / (S.R3 + 1))
C = pd.DataFrame(0.0, index=S.index, columns=EXP)
for f in F: C[FM[f]["primary"]] += S["shap_" + f].values
C[CAT["special_terms"]["base"]] += S.base.values; C[CAT["special_terms"]["level_offset"]] += S.offset.values
out = S[["style_color", "R3", "F", "d_hat"]].copy(); out.insert(1, "product", it.product_name.reindex(S.style_color).values)
for c in EXP: out[c] = C[c].round(3)
def drivers(r):
    s = C.loc[r.name].sort_values(key=np.abs, ascending=False).head(3)
    return "; ".join(f"{NAME[k]} {np.expm1(v):+.0%}" for k, v in s.items())
out["top_drivers"] = out.apply(drivers, axis=1)
out = out.sort_values("R3", ascending=False); out.to_csv("data/fk_explain_2026_targets.csv", index=False)
w = S.R3.values / S.R3.sum()
print(f"100 targets, June 2026 origin: R3 total {S.R3.sum():,.0f}, model median total {S.F.sum():,.0f} ({S.F.sum() / S.R3.sum() - 1:+.1%}); check: categories sum to the deviation, max error {np.abs(C.sum(1) - S.d_hat).max():.2e}")
print("\n== R3-weighted mean contribution by category (log units; % effect)")
pv = pd.Series({NAME[c]: (C[c].values * w).sum() for c in EXP}).sort_values(key=np.abs, ascending=False)
for k, v in pv.items(): print(f"  {k:52s} {v:+.3f}  ({np.expm1(v):+.1%})")
print("\n== the ten largest targets"); print(out.head(10)[["style_color", "product", "R3", "F", "top_drivers"]].to_string(index=False))
