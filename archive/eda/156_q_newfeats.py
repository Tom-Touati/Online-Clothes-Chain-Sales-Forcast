"""D150/D151/D152: new feature groups added to both feature sets of the lean + occasion-timing quantile model (eda/20b, D135).
All features are time-honest (built from data up to each row's own origin). Variants (argv; default all):
  cp     co-purchase neighbour features (eda/153 cp): nb_sea4, nb_lvl, nb_mom1, nb_mom3, nb_yoy
  cpsea  co-purchase seasonal ramp and level only: nb_sea4, nb_lvl
  attr   the same five from attribute-only neighbours (eda/153 attr), as a control
  reg    regional mix and momentum (eda/154)
  llm    the 14 LLM item scores (D151, static)
  cpllm  cpsea + llm
Writes data/q_backtest_nf_<variant>.parquet; scored by eda/91_score_context.py nf data/q_backtest_lean.parquet."""
import sys, glob, numpy as np, pandas as pd, warnings, importlib.util, time
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
key = ["origin", "style_color"]
def bf(v, cols): X = pd.read_parquet(f"data/basket_feats_{v}.parquet")[key + cols]; return X.rename(columns={c: f"{v}_{c}" for c in cols}), [f"{v}_{c}" for c in cols]
NB = ["nb_sea4", "nb_lvl", "nb_mom1", "nb_mom3", "nb_yoy"]
RG = pd.read_parquet("data/region_feats.parquet"); RGC = [c for c in RG.columns if c not in key]
def llm():
    L = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("data/llm/scores_part*.csv"))]).drop_duplicates("style_color")
    return L, [c for c in L.columns if c != "style_color"]
VAR = {"cp": lambda: [bf("cp", NB)], "cpsea": lambda: [bf("cp", ["nb_sea4", "nb_lvl"])], "attr": lambda: [bf("attr", NB)], "reg": lambda: [(RG, RGC)],
       "llm": lambda: [llm()], "cpllm": lambda: [bf("cp", ["nb_sea4", "nb_lvl"]), llm()]}
RUN = sys.argv[1:] or list(VAR)
spec = importlib.util.spec_from_file_location("qm", "eda/20b_quantiles_lean.py"); qm = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(qm); sys.argv = _a
qm.P["n_jobs"] = 8; B0, F0, T0 = list(qm.BASE), list(qm.FULL), qm.T1
for var in RUN:
    T1, NEW = T0, []
    for X, cols in VAR[var]():
        T1 = T1.merge(X, on=key if "origin" in X.columns else ["style_color"], how="left"); NEW += cols
    qm.BASE = B0 + NEW; qm.FULL = F0 + NEW; out, t0 = [], time.time()
    for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
        oo = pd.Period(o).ordinal; tr = T1[(T1.o_ord + 4 <= oo) & T1.ytot.notna()]; te = T1[T1.origin == o]
        zq = qm.fit_q(tr, te); R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
        for j, q in enumerate(qm.QS): R[f"q{int(q*100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
        out.append(R)
    pd.concat(out).to_parquet(f"data/q_backtest_nf_{var}.parquet"); print(var, len(NEW), "features,", round(time.time() - t0), "s", flush=True)
