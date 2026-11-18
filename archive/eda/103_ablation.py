"""Information-dimension probes (D119): drop one feature group at a time from the production quantile model (eda/20, both feature
sets, 13 origins) and score the loss. Groups as in the waterfall (eda/101): seasonality, recent momentum, lifecycle, product &
attributes, occasion, channel mix, discount, size. Also 'only history' (lags + level only) and 'no history' (everything but the lags/pace).
Writes data/q_backtest_abl_<group>.parquet; scored by eda/91_score_context.py abl."""
import sys, numpy as np, pandas as pd, warnings, importlib.util, time
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
VARS = sys.argv[1:]
spec = importlib.util.spec_from_file_location("q20", "eda/20_quantiles.py"); q20 = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(q20); sys.argv = _a
q20.P["n_jobs"] = 8
G = {"seasonality": lambda c: c in ("own_sea4", "lfl_type4", "lfl_macro4", "lfl_occ4", "tmonth"),
     "momentum": lambda c: c.startswith("lagr") or c in ("f_r7", "f_r28", "trend91", "peak_r"),
     "lifecycle": lambda c: c in ("age_m", "days_live", "color_num"),
     "product": lambda c: c == "ptype" or c.startswith("a_"), "occasion": lambda c: c.startswith("occ_"),
     "channel": lambda c: c in ("sh_ret", "sh_drop", "sh_tt"), "discount": lambda c: c in ("disc_item", "disc_rel"), "size": lambda c: c == "lvl"}
CF = pd.read_parquet("data/context_feats.parquet")[["origin", "style_color", "occ_next", "occ_last", "occ_n4"]]
q20.T1 = q20.T1.merge(CF, on=["origin", "style_color"], how="left"); OCCT = ["occ_next", "occ_last", "occ_n4"]          # D112 occasion timing
B0, F0 = list(q20.BASE), list(q20.FULL)
for var in VARS:
    if var == "onlyhistory": keep = lambda c: G["momentum"](c) or G["size"](c)
    elif var == "nohistory": keep = lambda c: not G["momentum"](c)
    elif var == "lean": keep = lambda c: not (G["lifecycle"](c) or G["channel"](c) or G["discount"](c) or G["occasion"](c))   # D134: drop the groups worth ~0
    elif var != "leanocct": keep = lambda c, g=G[var]: not g(c)
    if var == "leanocct": keep = lambda c: not (G["lifecycle"](c) or G["channel"](c) or G["discount"](c) or G["occasion"](c))
    q20.BASE, q20.FULL = [c for c in B0 if keep(c)], [c for c in F0 if keep(c)]
    if var == "leanocct": q20.BASE, q20.FULL = q20.BASE + OCCT, q20.FULL + OCCT
    rows, t0 = [], time.time()
    for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
        oo = pd.Period(o).ordinal; tr = q20.T1[(q20.T1.o_ord + 4 <= oo) & q20.T1.ytot.notna()]; te = q20.T1[q20.T1.origin == o]
        zq = q20.fit_q(tr, te); R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
        for j, q in enumerate(q20.QS): R[f"q{int(q*100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
        rows.append(R)
    pd.concat(rows).to_parquet(f"data/q_backtest_abl_{var}.parquet"); print(var, len(q20.BASE), len(q20.FULL), round(time.time() - t0), "s", flush=True)
