"""Production quantile model (eda/20, 4-month totals, 13 origins) with the D113 Matryoshka embeddings added to both feature sets.
At each origin the trees use the embedding model trained at that origin (data/mrl_emb.parquet), for training and test rows alike;
the test rows' outcomes never reached the embedding model, so the backtest stays honest.
Variants: v4 / v8 / v16 = first 4 / 8 / 16 dims of the summed item vector; all4 = every table's first 4 dims (hierarchy levels,
attributes, collection, colour, occasion); all16 = every table's full 16 dims plus the summed vector (all the NN embeddings).
Writes data/q_backtest_mrl_<var>.parquet. Usage: python eda/94_q_mrl.py v4 v8 v16 all4 all16"""
import sys, numpy as np, pandas as pd, warnings, importlib.util, time
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
ARGS = dict(a.split("=") for a in sys.argv[1:] if "=" in a); VARS = [a for a in sys.argv[1:] if "=" not in a] or ["v4", "v8", "v16", "all4", "all16"]
EMB = ARGS.get("emb", "data/mrl_emb.parquet"); TAG = ARGS.get("tag", "mrl")
ORIGS = ARGS["origins"].split(",") if "origins" in ARGS else [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
spec = importlib.util.spec_from_file_location("q20", "eda/20_quantiles.py"); q20 = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(q20); sys.argv = _a
q20.P["n_jobs"] = 8
EM = pd.read_parquet(EMB)
TAB = sorted({c.rsplit("_", 1)[0] for c in EM.columns if c not in ("origin", "style_color")} - {"v"})
GROUP = [t for t in TAB if t not in ("style", "item")]          # group-level tables: no style or item vectors (they memorise single items)
def add_group_sum(EM):
    for d in range(16): EM[f"g_{d}"] = sum(EM[f"{t}_{d}"] for t in GROUP)
    return EM
EM = add_group_sum(EM)
def cols(var):
    if var.startswith("v"): return [f"v_{d}" for d in range(int(var[1:]))]
    if var.startswith("gall"): return [f"{t}_{d}" for t in GROUP for d in range(int(var[4:]))]
    if var.startswith("g"): return [f"g_{d}" for d in range(int(var[1:]))]
    k = int(var[3:]); return [f"{t}_{d}" for t in TAB for d in range(k)] + ([f"v_{d}" for d in range(16)] if k == 16 else [])
B0, F0 = list(q20.BASE), list(q20.FULL)
for var in VARS:
    C = cols(var); q20.BASE, q20.FULL = B0 + C, F0 + C; rows, t0 = [], time.time()
    for o in ORIGS:
        E = EM[EM.origin == o].drop(columns="origin")[["style_color"] + C]
        oo = pd.Period(o).ordinal
        tr = q20.T1[(q20.T1.o_ord + 4 <= oo) & q20.T1.ytot.notna()].merge(E, on="style_color", how="left")
        te = q20.T1[q20.T1.origin == o].merge(E, on="style_color", how="left")
        zq = q20.fit_q(tr, te); R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
        for j, q in enumerate(q20.QS): R[f"q{int(q*100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
        rows.append(R)
    pd.concat(rows).to_parquet(f"data/q_backtest_{TAG}_{var}.parquet"); print(var, len(C), "features added", round(time.time() - t0), "s", flush=True)
