"""Combination test on identical items (D105): rules, production quantile median and embedding NN (S13x6), geometric blends.
Uses the scored frame of eda/77 (same items, origins and actuals). Writes data/combo_eval_output.txt via stdout redirect."""
import numpy as np, pandas as pd, contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("eda/77_rules_eval.py", encoding="utf-8").read().split("# paired item bootstrap")[0])
    exec("# paired item bootstrap" + open("eda/77_rules_eval.py", encoding="utf-8").read().split("# paired item bootstrap")[1].split("print(")[0])
g = lambda *cs, w=None: np.exp(sum((1 / len(cs) if w is None else wi) * np.log(T[c].clip(lower=0.5)) for c, wi in zip(cs, w or [None] * len(cs))))
T["q_nn"] = g("qmodel", "nn")                      # the NN re-centring of the candidate workbook (D095)
T["r_q"] = g("rules", "qmodel")
T["r_nn"] = g("rules", "nn")
T["r_q_nn"] = g("rules", "qmodel", "nn")
T["r_qnn"] = g("rules", "q_nn")                    # half rules, half the candidate's centre
C = ["rules", "qmodel", "nn", "lgb_nn", "q_nn", "r_q", "r_nn", "r_q_nn", "r_qnn"]
TR = T[T.origin.isin(REC)]
for nm, F in [("all 13 origins", T), ("nine recent origins", TR)]:
    print(f"== {nm}: pooled WAPE / bias / within 20%")
    print(pd.DataFrame({c: met(F, c) for c in C}).T[["wape", "bias", "w20"]].round(3).to_string())
    piv = pd.DataFrame({c: F.groupby("origin").apply(lambda x: met(x, c).wape) for c in C})
    print("origins where r_q_nn beats:", {c: int((piv.r_q_nn < piv[c]).sum()) for c in ["r_q", "q_nn", "qmodel"]}, "of", len(piv))
    for a, c in [("r_q_nn", "r_q"), ("r_q_nn", "q_nn"), ("r_q", "q_nn"), ("r_qnn", "r_q")]:
        print(f"  {a} - {c}:", np.round(boot(a, c, 1000, F), 3))
print("== recent origins, targets only")
X = TR[TR.target]; print(pd.DataFrame({c: met(X, c) for c in C}).T[["items", "wape", "bias"]].round(3).to_string())
print("== recent origins by weight on rules (rest split q/nn equally)")
for w in [0.2, 0.33, 0.4, 0.5, 0.6]:
    T["_w"] = g("rules", "qmodel", "nn", w=[w, (1 - w) / 2, (1 - w) / 2]); F = T[T.origin.isin(REC)]
    print(f"  w_rules {w:.2f}: recent {met(F, '_w').wape:.3f} | all {met(T, '_w').wape:.3f}")
