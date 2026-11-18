"""D200 (fork): explainability steps (c) and (d) on the LLM judges' scores (data/fk_judge_{A,B,C}.csv; A = Opus, B = Sonnet, C = Haiku)
for the 120 sampled item-origins (data/fk_judge_sample.csv).
  1. Agreement between judges: per explained category, the correlation of the signed scores for each pair of judges; for the residual,
     the share of cards where two judges name the same primary cause, and Fleiss' kappa.
  2. Judges vs SHAP: correlation of the judges' mean score with the category's SHAP sum (categoriser map) for each category.
  3. Surrogate (d): 5-fold cross-validated ridge regression of the actual deviation from R3 on the judges' mean explained scores; also on
     the SHAP category sums of the same cards, and the full model. Lift recovered = (1 - WAPE / WAPE_R3) / the full model's.
  4. The residual: which unexplained causes the judges name, by size and sign of the residual, and for young items.
Writes eda/judges_output.txt (stdout) and data/fk_judge_consensus.csv."""
import sys, json, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
CAT = json.load(open("data/fk_categories.json")); EXP = [c["id"] for c in CAT["categories"] if c["kind"] == "explained"]; UNX = [c["id"] for c in CAT["categories"] if c["kind"] == "unexplained"]
J = {k: pd.read_csv(f"data/fk_judge_{k}.csv").set_index("card") for k in "ABC"}
smp = pd.read_csv("data/fk_judge_sample.csv").set_index("card")
SC = pd.read_parquet("data/fk_shap_categories.parquet").set_index(["origin", "style_color"])
SC = SC.reindex(pd.MultiIndex.from_arrays([smp.origin, smp.style_color])); SC.index = smp.index
T = pd.read_parquet("data/fk_explain_table.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([smp.origin, smp.style_color])); smp["age_m"] = T.age_m.values
print("== 1. agreement between judges (A Opus, B Sonnet, C Haiku): correlation of signed scores per explained category")
rows = []
for c in EXP:
    a, b, cc = (J[k][c].reindex(smp.index).astype(float) for k in "ABC")
    rr = lambda x, y: np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else np.nan
    m = (a + b + cc) / 3
    rows.append({"category": c, "A-B": rr(a, b), "A-C": rr(a, cc), "B-C": rr(b, cc), "judges mean vs SHAP sum": rr(m, SC[c]), "mean |score|": m.abs().mean()})
A1 = pd.DataFrame(rows).set_index("category"); print(A1.round(3).to_string()); print("mean pairwise r:", round(A1[["A-B", "A-C", "B-C"]].mean().mean(), 3), "| mean judges-vs-SHAP r:", round(A1["judges mean vs SHAP sum"].mean(), 3))
rp = pd.DataFrame({k: J[k].residual_primary.reindex(smp.index) for k in "ABC"})
print("\n== residual primary cause: pairwise agreement A-B %.2f, A-C %.2f, B-C %.2f" % tuple((rp[x] == rp[y]).mean() for x, y in [("A", "B"), ("A", "C"), ("B", "C")]))
cats = sorted(set(rp.values.ravel())); cnt = np.stack([(rp == c).sum(1).values for c in cats], 1); nr = 3
Pi = ((cnt * (cnt - 1)).sum(1) / (nr * (nr - 1))).mean(); pj = cnt.sum(0) / cnt.sum(); kappa = (Pi - (pj ** 2).sum()) / (1 - (pj ** 2).sum())
print(f"Fleiss kappa {kappa:.3f}"); print(rp.apply(lambda s: s.value_counts()).fillna(0).astype(int).to_string())
print("\n== 3. surrogate of the lift over R3 on the 120 cards (5-fold cross-validated ridge on log deviation)")
Mj = sum(J[k][EXP].reindex(smp.index).astype(float) for k in "ABC") / 3
y = smp.dev_actual.values; R3 = smp.R3.values; act = smp.ytot.values
def cv(X):
    p = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X): p[te] = Ridge(alpha=1.0).fit(X[tr], y[tr]).predict(X[te])
    return p
def wape(fc): return np.abs(fc - act).sum() / act.sum()
w3 = wape(R3); wf = wape(smp.F.values); lf = 1 - wf / w3
print(f"  R3 WAPE {w3:.3f} | full model (D159 quantile median) WAPE/R3 {wf / w3:.3f}, lift {lf:.3f}")
for nm, X in [("judges' mean explained scores (11)", Mj.values), ("SHAP category sums (11), same cards", SC[EXP].values), ("judges + SHAP sums", np.hstack([Mj.values, SC[EXP].values])),
              ("judges' unexplained plausibility (5)", (sum(J[k][UNX].reindex(smp.index).astype(float) for k in "ABC") / 3).values)]:
    p = cv(X); fc = np.clip((R3 + 1) * np.exp(p) - 1, 0, None); w = wape(fc)
    print(f"  {nm:40s} WAPE/R3 {w / w3:.3f} | lift {1 - w / w3:.3f} | recovered {(1 - w / w3) / lf:6.1%} | R^2 {1 - np.mean((y - p) ** 2) / np.var(y):.3f}")
print("\n== 4. residual = actual minus model deviation: what the judges call it (majority vote)")
maj = rp.apply(lambda r: r.value_counts().idxmax() if r.value_counts().max() >= 2 else "no majority", axis=1)
res = smp.dev_actual - smp.d_hat
D = pd.DataFrame({"cause": maj, "residual": res, "young": smp.age_m < 6})
print(D.groupby("cause").agg(cards=("residual", "size"), mean_residual=("residual", "mean"), mean_abs=("residual", lambda x: x.abs().mean()), young_share=("young", "mean")).round(3).sort_values("cards", ascending=False).to_string())
print(f"\nresidual |r| >= 0.5: {int((res.abs() >= 0.5).sum())} cards; their majority causes:"); print(D[res.abs() >= 0.5].cause.value_counts().to_string())
out = smp.copy(); out["residual"] = res; out["cause_majority"] = maj
for c in EXP: out["judge_" + c] = Mj[c]
out["explanation_A"] = J["A"].explanation.reindex(smp.index); out["explanation_B"] = J["B"].explanation.reindex(smp.index); out["explanation_C"] = J["C"].explanation.reindex(smp.index)
out.to_csv("data/fk_judge_consensus.csv")
