"""D159: can temperature + special events replace seasonality? And do weather, events and the multi-view clusters add to the best model?
Quantile-model variants on the lean + occasion-timing model (eda/20b, D135); the seasonality group (D134) is own_sea4, lfl_type4,
lfl_macro4, lfl_occ4, tmonth. Occasion timing (days to the item's occasion) stays in every variant: it is itself an event feature.
  nosea         lean minus seasonality
  nosea_we      lean minus seasonality + weather known at the origin + events + LLM scores and event interactions
  nosea_weor    nosea_we + ORACLE realised target-window temperature (an upper bound on what weather could add; not a forecast)
  we            lean + weather + events + LLM scores
  cpllm_we      D153 model (lean + cart ramp + LLM) + weather + events
  cpllm_cl      D153 model + multi-view cluster features
  cpllm_wecl    D153 model + weather + events + clusters
  nosea_t / nosea_e / cpllm_t / cpllm_e   the same split into temperature only and events only (D159b)
  dpllm_we / cpdp_we   neighbour features from the learned descriptor + observed cart/region vectors (eda/166, D160b) instead of / beside the cart ones
Writes data/q_backtest_wx_<variant>.parquet; scored by eda/91_score_context.py wx data/q_backtest_lean.parquet (and against cpllm)."""
import sys, glob, numpy as np, pandas as pd, warnings, importlib.util, time
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
key = ["origin", "style_color"]
WF = pd.read_parquet("data/weather_event_feats.parquet")
ORACLE = ["t_tgt_real", "ta_tgt"]; WE = [c for c in WF.columns if c not in key + ORACLE]
CF = pd.read_parquet("data/cluster_feats.parquet"); CL = [c for c in CF.columns if c not in key]
BK = pd.read_parquet("data/basket_feats_cp.parquet")[key + ["nb_sea4", "nb_lvl"]].rename(columns={"nb_sea4": "cp_nb_sea4", "nb_lvl": "cp_nb_lvl"}); CP = ["cp_nb_sea4", "cp_nb_lvl"]
DPF = pd.read_parquet("data/descr_behaviour_feats.parquet")[key + ["dp_nb_sea4", "dp_nb_lvl"]]; DP = ["dp_nb_sea4", "dp_nb_lvl"]
L = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("data/llm/scores_part*.csv"))]).drop_duplicates("style_color"); LL = [c for c in L.columns if c != "style_color"]
TEMP = [c for c in WE if c.startswith(("t_", "ta_", "dT_", "beta_T", "bT_")) or c == "x_warm_dT"]; EVT = [c for c in WE if c not in TEMP]
SEA = ("own_sea4", "lfl_type4", "lfl_macro4", "lfl_occ4", "tmonth")
VAR = {"nosea": (True, []), "nosea_we": (True, WE + LL), "nosea_weor": (True, WE + LL + ORACLE), "we": (False, WE + LL),
       "cpllm_we": (False, CP + LL + WE), "cpllm_cl": (False, CP + LL + CL), "cpllm_wecl": (False, CP + LL + WE + CL),
       "nosea_t": (True, TEMP + LL), "nosea_e": (True, EVT + LL), "cpllm_t": (False, CP + LL + TEMP), "cpllm_e": (False, CP + LL + EVT),
       "dpllm_we": (False, DP + LL + WE), "cpdp_we": (False, CP + DP + LL + WE)}
RUN = sys.argv[1:] or list(VAR)
spec = importlib.util.spec_from_file_location("qm", "eda/20b_quantiles_lean.py"); qm = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(qm); sys.argv = _a
qm.P["n_jobs"] = 8; B0, F0 = list(qm.BASE), list(qm.FULL)
T0 = qm.T1.merge(WF, on=key, how="left").merge(CF, on=key, how="left").merge(BK, on=key, how="left").merge(DPF, on=key, how="left").merge(L, on="style_color", how="left")
for var in RUN:
    drop, NEW = VAR[var]; keep = (lambda c: c not in SEA) if drop else (lambda c: True)
    qm.BASE = [c for c in B0 if keep(c)] + NEW; qm.FULL = [c for c in F0 if keep(c)] + NEW; out, t0 = [], time.time()
    for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
        oo = pd.Period(o).ordinal; tr = T0[(T0.o_ord + 4 <= oo) & T0.ytot.notna()]; te = T0[T0.origin == o]
        zq = qm.fit_q(tr, te); R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
        for j, q in enumerate(qm.QS): R[f"q{int(q*100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
        out.append(R)
    pd.concat(out).to_parquet(f"data/q_backtest_wx_{var}.parquet"); print(var, len(qm.FULL), "features (full set),", round(time.time() - t0), "s", flush=True)
