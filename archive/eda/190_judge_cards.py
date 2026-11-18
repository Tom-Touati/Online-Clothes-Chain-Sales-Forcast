"""D194 (fork): item cards for the LLM judges (explainability step c).
Sample: 120 item-origins from the nine recent origins, 60 of the 100 targets and 60 other items, stratified by the size of the actual
deviation from R3 (quartiles) and spread over origins (fixed seed). Each card: item name, type, colour, LLM descriptor highlights, age,
origin, R3, the model's median forecast F, the actual, the deviations log((F+1)/(R3+1)) and log((actual+1)/(R3+1)), the base value and
level offset, and the 12 largest SHAP contributions with the feature's value. Category sums from the categoriser's primary map
(data/fk_categories.json) are computed for every item-origin for the later steps, but NOT shown to the judges.
Writes data/fk_judge_cards.md, data/fk_judge_sample.csv and data/fk_shap_categories.parquet (all item-origins x category sums)."""
import sys, json, glob, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
S = pd.read_parquet("data/fk_shap.parquet"); T = pd.read_parquet("data/fk_explain_table.parquet"); CAT = json.load(open("data/fk_categories.json"))
it = pd.read_parquet("data/items.parquet").set_index("style_color")
L = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("data/llm/scores_part*.csv"))]).drop_duplicates("style_color").set_index("style_color")
F = [c[5:] for c in S.columns if c.startswith("shap_")]; FM = CAT["feature_map"]
S["F"] = (4 * S.level + 1) * np.exp(S.z_hat) - 1; S["d_hat"] = np.log((S.F + 1) / (S.R3 + 1)); S["offset"] = np.log((4 * S.level + 1) / (S.R3 + 1))
ids = [c["id"] for c in CAT["categories"]]
C = pd.DataFrame(0.0, index=S.index, columns=ids)
for f in F: C[FM[f]["primary"]] += S["shap_" + f].values
C[CAT["special_terms"]["base"]] += S.base.values; C[CAT["special_terms"]["level_offset"]] += S.offset.values
C["residual"] = S.dev_actual - S.d_hat
pd.concat([S[["origin", "style_color", "target", "R3", "F", "ytot", "dev_actual", "d_hat"]], C], axis=1).to_parquet("data/fk_shap_categories.parquet", index=False)
rec = S[S.origin >= "2025-06"].copy(); rec["q"] = pd.qcut(rec.dev_actual.abs(), 4, labels=False); rng = np.random.default_rng(7)
pick = []
for tg in [True, False]:
    for q in range(4):
        pool = rec[(rec.target.astype(bool) == tg) & (rec.q == q)]; pick.append(pool.sample(15, random_state=int(rng.integers(1e6))))
smp = pd.concat(pick).sample(frac=1, random_state=1).reset_index(); smp["card"] = [f"C{k:03d}" for k in range(len(smp))]
TV = T.set_index(["origin", "style_color"])
lines = ["# Item cards\n", "Units are log ratios: 0.1 is about +10%. Deviation = log((x + 1) / (R3 + 1)); R3 = the item's last 3 months x 4.",
         "SHAP values add up to the model deviation together with the base value and the level offset.\n"]
for _, r in smp.iterrows():
    sc = r.style_color; info = it.loc[sc] if sc in it.index else None; l = L.loc[sc] if sc in L.index else None
    hi = ", ".join(f"{k} {int(v)}" for k, v in l.sort_values(ascending=False).head(4).items()) if l is not None else ""
    sh = pd.Series({f: r["shap_" + f] for f in F}); top = sh.reindex(sh.abs().sort_values(ascending=False).index).head(12)
    fv = TV.loc[(r.origin, sc)]
    lines.append(f"## {r.card}: {info.product_name if info is not None else sc} [{info.product_type if info is not None else ''}, {sc.split('_', 1)[1]}]")
    lines.append(f"- origin {r.origin} (forecast covers the next 4 months); item age {fv.age_m:.0f} months; LLM highlights: {hi}")
    lines.append(f"- R3 {r.R3:.0f} | model {r.F:.0f} (deviation {r.d_hat:+.2f}) | actual {r.ytot:.0f} (deviation {r.dev_actual:+.2f}) | residual {r.dev_actual - r.d_hat:+.2f}")
    lines.append(f"- base value {r.base:+.2f}; level offset {r.offset:+.2f}; other features sum {sh.sum() - top.sum():+.2f}")
    lines.append("- top SHAP: " + "; ".join(f"{f}={fv[f] if not isinstance(fv[f], float) else round(fv[f], 3)} -> {v:+.3f}" for f, v in top.items()) + "\n")
open("data/fk_judge_cards.md", "w", encoding="utf-8").write("\n".join(lines))
smp[["card", "origin", "style_color", "target", "R3", "F", "ytot", "dev_actual", "d_hat"]].to_csv("data/fk_judge_sample.csv", index=False)
print(len(smp), "cards;", smp.target.astype(bool).sum(), "targets")
