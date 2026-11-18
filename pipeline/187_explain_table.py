"""D191 (fork): self-contained table for the explainability pipeline and the remote runs.
One row per item x origin (2024-03 .. 2026-06) with every feature of the current best quantile model (D159 cpllm_we: lean + occasion
timing + cart neighbours' seasonal ramp + LLM scores + temperature + events), the model's target z = log((ytot + 1) / (4 x level + 1)),
R3 and the deviation targets:
  dev_actual = log((ytot + 1) / (R3 + 1))           how far the truth moved from R3
  z           the model's own target (deviation from the live-adjusted recent level 4 x level)
Also the 100-target flag and the feature lists of both feature sets (BASE, FULL) and a feature -> group map for SHAP reading.
Writes data/fk_explain_table.parquet and data/fk_explain_meta.json."""
import sys, glob, json, importlib.util, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
key = ["origin", "style_color"]
spec = importlib.util.spec_from_file_location("qm", "pipeline/20b_quantiles_lean.py"); qm = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(qm); sys.argv = _a
WF = pd.read_parquet("data/weather_event_feats.parquet"); ORACLE = ["t_tgt_real", "ta_tgt"]; WE = [c for c in WF.columns if c not in key + ORACLE]
BK = pd.read_parquet("data/basket_feats_cp.parquet")[key + ["nb_sea4", "nb_lvl"]].rename(columns={"nb_sea4": "cp_nb_sea4", "nb_lvl": "cp_nb_lvl"}); CP = ["cp_nb_sea4", "cp_nb_lvl"]
L = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("data/llm/scores_part*.csv"))]).drop_duplicates("style_color"); LL = [c for c in L.columns if c != "style_color"]
T = qm.T1.merge(WF.drop(columns=ORACLE), on=key, how="left").merge(BK, on=key, how="left").merge(L, on="style_color", how="left")
BASE = list(qm.BASE) + CP + LL + WE; FULL = list(qm.FULL) + CP + LL + WE
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
Pm = lambda o, k: str(pd.Period(o, "M") + k)
T["R3"] = [4 * M.loc[s, [Pm(o, -k) for k in range(3)]].reindex().fillna(0).mean() if s in M.index else np.nan for o, s in zip(T.origin, T.style_color)]
T["dev_actual"] = np.log((T.ytot + 1) / (T.R3 + 1))
tg = pd.read_parquet("data/fixes_items.parquet")[["style_color", "target"]].drop_duplicates("style_color"); T = T.merge(tg, on="style_color", how="left"); T["target"] = T.target.fillna(False)
cat = [c for c in set(BASE + FULL) if str(T[c].dtype) == "category"]
GROUP = {}
for c in set(BASE + FULL):
    if c.startswith("lagr") or c in ("f_r7", "f_r28", "trend91", "peak_r"): g = "recent momentum"
    elif c in ("lvl", "level"): g = "sales level"
    elif c in ("own_sea4", "lfl_type4", "lfl_macro4", "lfl_occ4", "tmonth"): g = "seasonality (history)"
    elif c.startswith(("t_", "ta_", "dT_", "beta_T", "bT_")) or c == "x_warm_dT": g = "temperature / weather"
    elif c.startswith("ev_") or c.startswith("x_") or c.startswith("occ_"): g = "special events / occasions"
    elif c.startswith("cp_"): g = "co-purchase cluster seasonality"
    elif c in LL: g = "fashion descriptors (LLM scores)"
    elif c.startswith("a_") or c in ("ptype", "macro"): g = "product type and attributes"
    elif c.startswith("sh_") or c.startswith("disc"): g = "channel / discount"
    else: g = "other"
    GROUP[c] = g
keep = sorted(set(key + ["o_ord", "level", "ytot", "z", "R3", "dev_actual", "target", "age_m"] + BASE + FULL))
T[keep].to_parquet("data/fk_explain_table.parquet", index=False)
json.dump({"BASE": BASE, "FULL": FULL, "CAT": cat, "GROUP": GROUP, "PARAMS": {k: v for k, v in qm.P.items()}}, open("data/fk_explain_meta.json", "w"), indent=1)
print(T[keep].shape, "| BASE", len(BASE), "FULL", len(FULL), "| categorical", cat)
print(pd.Series(GROUP).value_counts().to_string())
