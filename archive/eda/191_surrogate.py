"""D195 (fork): explainability step (d), simple surrogates of the lift over R3.
Target: the actual deviation from R3, dev_actual = log((actual + 1) / (R3 + 1)); forecast = (R3 + 1) exp(prediction) - 1.
Time-honest: SHAP exists for the 13 backtest origins (models fitted at each origin, eda/188); a surrogate scored at origin o is fitted
on the earlier backtest origins whose outcome is known (o' + 4 <= o), so it is scored on the nine recent origins.
Surrogates:
  cat_lin     linear regression on the 11 explained category sums (categoriser map, data/fk_shap_categories.parquet)
  cat_tree    depth-3 regression tree on the same
  cat_unit    the category sums added with weight 1 (= the full model's own deviation; reference for the map)
  shap_lin    linear on the 15 features with the largest mean |SHAP| in the training origins (their SHAP values)
  raw_lin     linear on those 15 features' raw values (median-imputed)
  raw_tree    depth-3 tree on those raw values
Lift = 1 - WAPE / WAPE(R3); recovered = lift(surrogate) / lift(full model median). Also R^2 of dev_actual, all items and 100 targets.
Writes eda/surrogate_output.txt (stdout) and data/fk_surrogate_preds.parquet."""
import sys, json, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor, export_text
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 200)
C = pd.read_parquet("data/fk_shap_categories.parquet"); S = pd.read_parquet("data/fk_shap.parquet"); T = pd.read_parquet("data/fk_explain_table.parquet")
CAT = json.load(open("data/fk_categories.json")); EXP = [c["id"] for c in CAT["categories"] if c["kind"] == "explained"]
F = [c[5:] for c in S.columns if c.startswith("shap_")]
D = C.merge(S[["origin", "style_color"] + ["shap_" + f for f in F]], on=["origin", "style_color"]).merge(T[["origin", "style_color"] + F], on=["origin", "style_color"])
num = [f for f in F if str(D[f].dtype) not in ("category", "object")]
ORIG = sorted(D.origin.unique()); preds = {k: np.full(len(D), np.nan) for k in ["cat_lin", "cat_tree", "cat_unit", "shap_lin", "raw_lin", "raw_tree"]}
trees_txt = {}
for o in ORIG:
    oo = pd.Period(o).ordinal; tr = D[[pd.Period(x).ordinal + 4 <= oo for x in D.origin]]; te = D.origin == o
    if len(tr) < 500: continue
    y = tr.dev_actual.values
    m = Ridge(alpha=1.0).fit(tr[EXP], y); preds["cat_lin"][te] = m.predict(D.loc[te, EXP])
    t = DecisionTreeRegressor(max_depth=3, min_samples_leaf=100).fit(tr[EXP], y); preds["cat_tree"][te] = t.predict(D.loc[te, EXP])
    preds["cat_unit"][te] = D.loc[te, EXP].sum(1)
    top = pd.Series({f: tr["shap_" + f].abs().mean() for f in F}).sort_values(ascending=False).head(15).index.tolist()
    m = Ridge(alpha=1.0).fit(tr[["shap_" + f for f in top]], y); preds["shap_lin"][te] = m.predict(D.loc[te, ["shap_" + f for f in top]])
    rt = [f for f in top if f in num]; med = tr[rt].median()
    m = Ridge(alpha=1.0).fit(tr[rt].fillna(med), y); preds["raw_lin"][te] = m.predict(D.loc[te, rt].fillna(med))
    t2 = DecisionTreeRegressor(max_depth=3, min_samples_leaf=100).fit(tr[rt].fillna(med), y); preds["raw_tree"][te] = t2.predict(D.loc[te, rt].fillna(med))
    if o == ORIG[-1]:
        trees_txt["cat_tree"] = export_text(t, feature_names=EXP, decimals=2); trees_txt["raw_tree"] = export_text(t2, feature_names=rt, decimals=2)
        trees_txt["cat_lin_coef"] = dict(zip(EXP, np.round(Ridge(alpha=1.0).fit(tr[EXP], y).coef_, 3)))
        trees_txt["top15"] = top
for k, v in preds.items(): D["p_" + k] = v
rec = D[D.origin >= "2025-06"].copy()
def wape(Z, fc): return (fc - Z.ytot).abs().sum() / Z.ytot.sum()
def r2(y, x): return 1 - np.mean((y - x) ** 2) / np.var(y)
for nm, Z in [("nine recent origins, all items", rec), ("nine recent, 100 targets", rec[rec.target.astype(bool)])]:
    w3 = wape(Z, Z.R3); wf = wape(Z, Z.F); lf = 1 - wf / w3
    print(f"\n######## {nm}: {len(Z)} item-origins | R3 WAPE {w3:.3f}")
    print(f"  {'full model (median of the D159 quantile model)':52s} WAPE/R3 {wf / w3:.4f} | lift {lf:.3f} | recovered 100.0% | R^2 {r2(Z.dev_actual.values, Z.d_hat.values):.3f}")
    for k in preds:
        fc = np.clip((Z.R3 + 1) * np.exp(Z["p_" + k]) - 1, 0, None); w = wape(Z, fc)
        print(f"  {k:52s} WAPE/R3 {w / w3:.4f} | lift {1 - w / w3:.3f} | recovered {(1 - w / w3) / lf:6.1%} | R^2 {r2(Z.dev_actual.values, Z['p_' + k].values):.3f}")
print("\n== last origin's surrogates"); print("cat_lin coefficients (log units per unit of category SHAP):", trees_txt["cat_lin_coef"])
print("top-15 SHAP features:", trees_txt["top15"]); print("\ncat_tree:\n" + trees_txt["cat_tree"]); print("raw_tree:\n" + trees_txt["raw_tree"])
D[["origin", "style_color", "target", "R3", "ytot", "F", "dev_actual", "d_hat"] + ["p_" + k for k in preds]].to_parquet("data/fk_surrogate_preds.parquet", index=False)
