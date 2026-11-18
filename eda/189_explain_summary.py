"""D191 (fork): (a) deviations from R3 and (b) their SHAP decomposition, summarised; context file for the LLM categoriser (c).
Deviation of the model's median from R3: d_hat = log((F + 1) / (R3 + 1)), F = (4 x level + 1) exp(z_hat) - 1. It splits exactly into
  level offset  log((4 x level + 1) / (R3 + 1))   (the recent rate per live day vs the plain 3-month mean: young items, part-month)
  base value    the trees' average deviation at that origin
  SHAP terms    per feature, summed into groups (data/fk_explain_meta.json)
Measures (nine recent origins unless stated): share of total |SHAP| by group; R^2 of the actual deviation dev_actual on d_hat and on each
group's sum alone; the same for the 100 targets. Writes eda/explain_summary_output.txt (stdout), data/fk_shap_groups.parquet and
data/fk_categoriser_context.md (feature dictionary, importances, monthly patterns, top-gain trees)."""
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
S = pd.read_parquet("data/fk_shap.parquet"); META = json.load(open("data/fk_explain_meta.json")); GROUP = META["GROUP"]; TREES = json.load(open("data/fk_top_trees.json"))
F = [c[5:] for c in S.columns if c.startswith("shap_")]
S["F"] = (4 * S.level + 1) * np.exp(S.z_hat) - 1; S["d_hat"] = np.log((S.F + 1) / (S.R3 + 1)); S["offset"] = np.log((4 * S.level + 1) / (S.R3 + 1))
groups = sorted(set(GROUP[f] for f in F))
for g in groups: S["G_" + g] = S[["shap_" + f for f in F if GROUP[f] == g]].sum(1)
S["G_level offset (young items)"] = S.offset; S["G_base"] = S.base
GC = [c for c in S.columns if c.startswith("G_")]
S[["origin", "style_color", "target", "R3", "F", "ytot", "dev_actual", "d_hat", "z_hat", "z"] + GC].to_parquet("data/fk_shap_groups.parquet", index=False)
rec = S[S.origin >= "2025-06"]
def r2(y, x): m = np.isfinite(x) & np.isfinite(y); return 1 - np.var(y[m] - np.polyval(np.polyfit(x[m], y[m], 1), x[m])) / np.var(y[m])
out = []
print("== (a) deviations from R3, nine recent origins")
for nm, D in [("all items", rec), ("100 targets", rec[rec.target.astype(bool)])]:
    print(f"  {nm}: actual deviation sd {D.dev_actual.std():.3f}, model deviation sd {D.d_hat.std():.3f}, corr {np.corrcoef(D.dev_actual, D.d_hat)[0, 1]:.3f}, "
          f"R^2 of actual on model {r2(D.dev_actual.values, D.d_hat.values):.3f}; sign agreement {(np.sign(D.dev_actual) == np.sign(D.d_hat)).mean():.3f}")
print("\n== (b) SHAP by group, nine recent origins: share of total |contribution|, mean signed, R^2 of the actual deviation on the group alone")
tot = sum(rec[c].abs().mean() for c in GC if c != "G_base")
for c in sorted(GC, key=lambda c: -rec[c].abs().mean()):
    print(f"  {c[2:]:38s} share {rec[c].abs().mean() / tot:6.1%} | mean {rec[c].mean():+.3f} | sd {rec[c].std():.3f} | R^2 alone {r2(rec.dev_actual.values, rec[c].values):.3f}")
imp = pd.Series({f: rec["shap_" + f].abs().mean() for f in F}).sort_values(ascending=False)
print("\n== top 30 features by mean |SHAP|"); print(imp.head(30).round(4).to_string())
mon = rec.assign(om=rec.origin.str[-2:]).groupby("om")[[c for c in GC if c != "G_base"]].mean().round(3); mon.columns = [c[2:] for c in mon.columns]
print("\n== mean signed group contribution by origin month (recent)"); print(mon.T.to_string())
DESC = {"lagr": "lagr1..6: log ratio of each of the last 6 months' units to the recent level (momentum)", "lvl": "log recent level", "f_r7": "last 7 days' rate vs level",
        "f_r28": "last 28 days' rate vs level", "trend91": "slope over 91 days", "peak_r": "recent peak vs level", "own_sea4": "item's own last-year ratio for the 4 target months",
        "lfl_type4": "product type's like-for-like seasonal ratio", "lfl_macro4": "macro group's seasonal ratio", "lfl_occ4": "occasion group's seasonal ratio", "tmonth": "first target month",
        "occ_next": "days to the item's next occasion", "occ_last": "days since its last occasion", "occ_n4": "occasions in the 4 target months",
        "cp_nb_sea4": "co-purchase neighbours' last-year ramp into the target months", "cp_nb_lvl": "co-purchase neighbours' sales level",
        "t_org": "temperature felt in the origin window (item's regional mix)", "ta_org": "origin-window temperature anomaly", "t_tgt_norm": "normal temperature of the target months",
        "dT_norm": "seasonal temperature swing origin -> target", "dT_real_from_org": "target normal minus origin felt temperature", "beta_T": "item's temperature preference across states",
        "bT_dT": "temperature preference x seasonal swing", "bT_ta": "temperature preference x origin anomaly", "x_warm_dT": "LLM warmth x seasonal swing",
        "ev_": "ev_<event>_t / _o: days of a US event window in the target / origin months", "x_": "x_<event>: LLM event-fit score x event days in the target months",
        "warmth": "LLM score 0-1 (also coverage, formality, fit_tightness, statement, staple, summer, winter, back_to_school, party, halloween, festival, layering, versatility)",
        "ptype": "product type", "a_": "tagger attributes (fabric, sleeve, length, silhouette, print, occasion tier)"}
with open("data/fk_categoriser_context.md", "w", encoding="utf-8") as fh:
    fh.write("# SHAP context for defining explanation categories\n\nModel: LightGBM median of log((4-month units + 1) / (4 x recent level + 1)), i.e. each item's deviation from recency (R3 = last 3 months x 4). "
             "SHAP values are in log units (0.1 = about +10%). 13 monthly origins Feb 2025 - Feb 2026; summaries below use the nine recent ones (Jun 2025 - Feb 2026). A fashion e-commerce brand (Gen-Z women's apparel, US).\n\n")
    fh.write("## Feature dictionary (feature: group: meaning)\n\n")
    for f in F:
        d = next((v for k, v in DESC.items() if f == k or (k.endswith("_") and f.startswith(k)) or (k == "lagr" and f.startswith("lagr"))), "LLM item score 0-10" if GROUP[f].startswith("fashion") else "")
        fh.write(f"- `{f}`: {GROUP[f]}: {d}\n")
    fh.write("\n## Groups: share of total |SHAP|, mean signed contribution, R^2 of the actual deviation on the group alone\n\n")
    for c in sorted(GC, key=lambda c: -rec[c].abs().mean()):
        fh.write(f"- {c[2:]}: share {rec[c].abs().mean() / tot:.1%}, mean {rec[c].mean():+.3f}, R^2 {r2(rec.dev_actual.values, rec[c].values):.3f}\n")
    fh.write("\n## Top 40 features by mean |SHAP|\n\n" + "\n".join(f"- `{k}` ({GROUP[k]}): {v:.4f}" for k, v in imp.head(40).items()))
    fh.write("\n\n## Mean signed group contribution by origin month\n\n```\n" + mon.T.to_string() + "\n```\n")
    fh.write(f"\n## How well the model's deviation matches the truth\n\nCorrelation of actual vs model deviation from R3: {np.corrcoef(rec.dev_actual, rec.d_hat)[0, 1]:.3f}; residual sd {np.std(rec.dev_actual - rec.d_hat):.3f} log units. "
             "The residual (actual minus model) is what the features do not explain.\n")
    fh.write("\n## Highest-gain trees (origins 2025-06, 2025-10, 2026-02; top 3 levels)\n\n")
    for o in ["2025-06", "2025-10", "2026-02"]:
        for t in TREES[o][:3]: fh.write(f"### {o}, tree {t['tree']} (total gain {t['total_gain']:.0f})\n```\n{t['rules']}\n```\n")
print("\nwrote data/fk_categoriser_context.md")
