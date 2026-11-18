"""D191 (fork): SHAP of the deviation from recency. Runs on the remote machine (or locally) from data/fk_explain_table.parquet.
At each of the 13 backtest origins (Feb 2025 .. Feb 2026), time-honest like the quantile model: the median (alpha 0.5) LightGBM
models of the D159 quantile model are refitted on both feature sets (BASE, FULL) with rows whose 4-month outcome is known; TreeSHAP
(pred_contrib) on the test rows decomposes the predicted deviation z_hat = log((forecast + 1) / (4 x level + 1)) into per-feature
contributions; the two sets are averaged (a feature missing from a set counts 0).
Also the five highest-gain trees of each FULL model, as readable rules (top three levels), for the LLM judges.
Writes data/fk_shap.parquet (origin, style_color, base, z_hat, shap_<feature>...) and data/fk_top_trees.json."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
T = pd.read_parquet("data/fk_explain_table.parquet"); META = json.load(open("data/fk_explain_meta.json"))
BASE, FULL, P = META["BASE"], META["FULL"], META["PARAMS"]; P["n_jobs"] = int(sys.argv[1]) if len(sys.argv) > 1 else 4
for c in META["CAT"]: T[c] = T[c].astype("category")
FEATS = sorted(set(BASE) | set(FULL)); out, trees, t0 = [], {}, time.time()


def rules(tree, depth=3):
    def walk(n, d):
        if "split_feature" not in n or d >= depth: return [] if "split_feature" not in n else ["  " * d + "..."]
        f = FULL[n["split_feature"]]; th = n["threshold"]; op = n.get("decision_type", "<=")
        return ["  " * d + f"if {f} {op} {th} (gain {n['split_gain']:.1f}):"] + walk(n["left_child"], d + 1) + ["  " * d + "else:"] + walk(n["right_child"], d + 1)
    return "\n".join(walk(tree["tree_structure"], 0))


for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
    oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()]; te = T[T.origin == o]
    S = np.zeros((len(te), len(FEATS))); base = np.zeros(len(te)); zh = np.zeros(len(te))
    for cols in (BASE, FULL):
        m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **P).fit(tr[cols], tr.z)
        c = m.predict(te[cols], pred_contrib=True); idx = [FEATS.index(f) for f in cols]
        S[:, idx] += c[:, :-1] / 2; base += c[:, -1] / 2; zh += m.predict(te[cols]) / 2
        if cols is FULL:
            info = m.booster_.dump_model()["tree_info"]
            def tot(n): return 0 if "split_gain" not in n else n["split_gain"] + tot(n["left_child"]) + tot(n["right_child"])
            g = np.array([tot(t["tree_structure"]) for t in info]); top = np.argsort(-g)[:5]
            trees[o] = [{"tree": int(i), "total_gain": float(g[i]), "rules": rules(info[i])} for i in top]
            imp = pd.Series(m.booster_.feature_importance("gain"), index=FULL).sort_values(ascending=False)
            trees[o + "_gain_importance"] = {k: float(v) for k, v in imp.head(25).items()}
    R = te[["origin", "style_color", "z", "R3", "dev_actual", "level", "ytot", "target"]].copy(); R["base"] = base; R["z_hat"] = zh
    R = pd.concat([R.reset_index(drop=True), pd.DataFrame(S, columns=["shap_" + f for f in FEATS])], axis=1); out.append(R)
    print(o, "train", len(tr), "test", len(te), "| max |sum(shap)+base - z_hat|", round(float(np.abs(S.sum(1) + base - zh).max()), 6), f"| {time.time() - t0:.0f}s", flush=True)
pd.concat(out, ignore_index=True).to_parquet("data/fk_shap.parquet", index=False); json.dump(trees, open("data/fk_top_trees.json", "w"), indent=1); print("saved")
