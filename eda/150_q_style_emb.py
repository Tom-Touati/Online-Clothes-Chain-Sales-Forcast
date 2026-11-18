"""D149: the forward style-embedding summaries (data/style_emb_feats.parquet, eda/149) added to both feature sets of the lean +
occasion-timing quantile model (eda/20b, D135). Each tree row (origin o', item) carries the summaries from the embedding trained at o'.
Variants: cs = sty_c + sty_s4 + sty_n; s = sty_s4 + sty_n (seasonal part only).
Writes data/q_backtest_semb_{cs,s}.parquet; scored by eda/91_score_context.py semb data/q_backtest_lean.parquet."""
import sys, numpy as np, pandas as pd, warnings, importlib.util, time
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
SF = pd.read_parquet("data/style_emb_feats.parquet")[["origin", "style_color", "sty_c", "sty_s4", "sty_n"]]
spec = importlib.util.spec_from_file_location("qm", "eda/20b_quantiles_lean.py"); qm = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(qm); sys.argv = _a
qm.P["n_jobs"] = 8; T1 = qm.T1.merge(SF, on=["origin", "style_color"], how="left"); B0, F0 = list(qm.BASE), list(qm.FULL)
print("tree rows with style features:", round(T1.sty_c.notna().mean(), 3))
for var, NEW in [("cs", ["sty_c", "sty_s4", "sty_n"]), ("s", ["sty_s4", "sty_n"])]:
    qm.BASE = B0 + NEW; qm.FULL = F0 + NEW; out, t0 = [], time.time()
    for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
        oo = pd.Period(o).ordinal; tr = T1[(T1.o_ord + 4 <= oo) & T1.ytot.notna()]; te = T1[T1.origin == o]
        zq = qm.fit_q(tr, te); R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
        for j, q in enumerate(qm.QS): R[f"q{int(q*100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
        out.append(R)
    pd.concat(out).to_parquet(f"data/q_backtest_semb_{var}.parquet"); print(var, round(time.time() - t0), "s", flush=True)
