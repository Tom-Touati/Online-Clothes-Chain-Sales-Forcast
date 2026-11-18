"""D204 (fork): TabPFN on the remote GPU (venv_tab, tabpfn 8.2), same task as eda/196 (TabDPT): target z = log((4-month total + 1) /
(4 x level + 1)), D159 FULL features (categoricals as integer codes), time-honest context = training rows whose window ended by the
origin (the most recent `cap` rows; default: all). Quantiles read from TabPFN's predictive distribution (output_type="quantiles").
Versions: v2 (Prior Labs License 1.1, Apache-2.0 + attribution, no sign-in; venv_tab, tabpfn 8.2); v3.5 and v3.5-fast (tabpfn 10.0 in venv_pfn;
non-commercial, non-production licence,
accepted by the user through Prior Labs' sign-in; the API key sits in ~/.cache/tabpfn/auth_token on the remote).
Usage: python eda/196b_tabpfn.py v2 [v3.5 v3.5-fast] [cap=10000] [est=8] [feat=dyn|dyn_all|dyn_fa|dyn_ch|dyn_nn]
Writes out/q_backtest_tabpfn_<version>[_c<cap>].parquet (q_backtest layout)."""
import sys, time, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
KW = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); VERS = [a for a in sys.argv[1:] if "=" not in a] or ["v2"]
CAP = int(KW.get("cap", 0)); EST = int(KW.get("est", 8))
T = pd.read_parquet("data/fk_explain_table.parquet"); META = json.load(open("data/fk_explain_meta.json")); COLS = META["FULL"]
for c in META["CAT"]: T[c] = T[c].astype("category").cat.codes.astype(float)
FEAT = KW.get("feat", "")
if FEAT == "sel":   # D206 F4: the early-origin pruned feature list (eda/193c), which already includes the dyn features
    T = T.merge(pd.read_parquet("data/fk_dyn_feats.parquet")[["origin", "style_color", "dy_nb_sea4", "dy_nb_lvl", "dy_nb_mom3", "dy_nb_yoy"]], on=["origin", "style_color"], how="left")
    COLS = json.load(open("data/fk_feat_select.json"))["FULL"]; FEAT_TAG = "sel"; FEAT = ""
else: FEAT_TAG = FEAT   # D205: extra feature blocks, e.g. feat=dyn or feat=dyn_all (dyn = eda/182 neighbours; fa/ch/nn = eda/182b blocks)
if FEAT:
    DY = ["dy_nb_sea4", "dy_nb_lvl", "dy_nb_mom3", "dy_nb_yoy"]; T = T.merge(pd.read_parquet("data/fk_dyn_feats.parquet")[["origin", "style_color"] + DY], on=["origin", "style_color"], how="left"); COLS = COLS + DY
    FMb = json.load(open("data/fk_fash_meta.json")); EX = {"dyn": [], "dyn_fa": FMb["FA"], "dyn_ch": FMb["CH"], "dyn_nn": FMb["NN"], "dyn_all": FMb["FA"] + FMb["CH"] + FMb["NN"],
                                                          "dyn_lean": FMb.get("LEAN", []), "dyn_core": FMb.get("CORE", []), "dyn_fl": FMb.get("FL", [])}[FEAT]
    for path, cols in [("data/fk_fash_feats.parquet", [c for c in EX if not c.startswith("fl_")]), ("data/fk_fash2_feats.parquet", [c for c in EX if c.startswith("fl_")])]:
        if cols: T = T.merge(pd.read_parquet(path)[["origin", "style_color"] + cols], on=["origin", "style_color"], how="left"); COLS = COLS + cols
QS = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
ORIG = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
if "orig" in KW: ORIG = KW["orig"].split(",")   # D209: e.g. orig=2026-06 for the deliverable (training rows: outcome known by the origin)
for v in VERS:
    rows, t0 = [], time.time(); tag = v.replace(".", "_") + ("_o" + KW["orig"].replace(",", "_") if "orig" in KW else "") + (f"_c{CAP}" if CAP else "") + (f"_{FEAT_TAG}" if FEAT_TAG else "") + (f"_e{EST}" if EST != 8 else "")
    try:
        for o in ORIG:
            oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()].sort_values("o_ord")
            if CAP: tr = tr.tail(CAP)
            te = T[T.origin == o]
            m = TabPFNRegressor.create_default_for_version(ModelVersion(v), n_estimators=EST, device="cuda", ignore_pretraining_limits=True, random_state=0)
            m.fit(tr[COLS].values.astype("float32"), tr.z.values.astype("float32"))
            zq = np.sort(np.column_stack(m.predict(te[COLS].values.astype("float32"), output_type="quantiles", quantiles=QS)), axis=1)
            R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
            for j, q in enumerate(QS): R[f"q{int(q * 100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
            rows.append(R); w = (R.q50 - R.ytot).abs().sum() / R.ytot.sum() if R.ytot.notna().any() else float("nan")
            print("tabpfn", v, o, "context", len(tr), "test", len(te), f"P50 WAPE {w:.3f}", f"{time.time() - t0:.0f}s", flush=True)
        pd.concat(rows).to_parquet(f"out/q_backtest_tabpfn_{tag}.parquet"); print("saved", tag, flush=True)
    except Exception as e: print("tabpfn", v, "FAILED:", repr(e)[:600], flush=True)
