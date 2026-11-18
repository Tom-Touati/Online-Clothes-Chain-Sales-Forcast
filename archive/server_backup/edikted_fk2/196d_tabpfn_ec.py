"""D270 (fork; user request via Main, 17 Nov): TabPFN v2 members rebuilt on Main's D253 features. Same task and time-honest rule as
eda/196b (target z, context = rows whose 4-month outcome is known by the origin, all of them; quantiles from TabPFN's predictive
distribution). Feature sets (data/fk_tab_ec_meta.json, built from fk_explain_table + fk_dyn_feats + nf2_feats):
  ec    D159 FULL + DY (dynamic co-purchase neighbours) + EV2 (15 more event windows + calendar counts, c_pay dropped) + CURVE (launch curves)
  ecns  the same minus STATIC (tagger attributes, product type, the 14 LLM scores), as Main's tree member ec_ns (eda/180q sets("ec_ns"))
Usage: python 196d_tabpfn_ec.py <ec|ecns> [est=32] [orig=2026-06]; writes out/q_backtest_tabpfn_v2_<set>_e<est>[_o<orig>].parquet."""
import sys, time, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
KW = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); SET = [a for a in sys.argv[1:] if "=" not in a][0]; EST = int(KW.get("est", 32))
T = pd.read_parquet("data/fk_tab_ec_table.parquet"); COLS = json.load(open("data/fk_tab_ec_meta.json"))[SET]
QS = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
ORIG = KW["orig"].split(",") if "orig" in KW else [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
rows, t0 = [], time.time()
for o in ORIG:
    oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()]; te = T[T.origin == o]
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, n_estimators=EST, device="cuda", ignore_pretraining_limits=True, random_state=0)
    m.fit(tr[COLS].values.astype("float32"), tr.z.values.astype("float32"))
    zq = np.sort(np.column_stack(m.predict(te[COLS].values.astype("float32"), output_type="quantiles", quantiles=QS)), axis=1)
    R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
    for j, q in enumerate(QS): R[f"q{int(q * 100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
    rows.append(R); w = (R.q50 - R.ytot).abs().sum() / R.ytot.sum() if R.ytot.notna().any() else float("nan")
    print("tabpfn v2", SET, o, "context", len(tr), "test", len(te), f"P50 WAPE {w:.3f}", f"{time.time() - t0:.0f}s", flush=True)
tag = f"{SET}_e{EST}" + (f"_o{KW['orig']}" if "orig" in KW else ""); pd.concat(rows).to_parquet(f"out/q_backtest_tabpfn_v2_{tag}.parquet"); print("saved", tag, flush=True)
