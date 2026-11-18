"""Fork 2 (user: "lets use tabpfn as initial model, then stack the ensemble afterwards"). TabPFN v2 (feat=dyn inputs, as the pf2 members,
eda/196b) predicts z = log((total + 1) / (4 level + 1)) for every row a backtest origin's tree would train on, out of fold, so the tree can
boost from TabPFN (init_score) and fit only the residual:
  for each origin o: training rows = windows closed by o (o' + 4 <= o, outcome known); 5 folds by style (no style in its own context);
  each fold predicted by TabPFN fitted on the other four; the test rows at o predicted by TabPFN fitted on all training rows.
Writes out/tabpfn_oof.parquet: origin (the backtest origin), row_origin, style_color, part (train|test), z_pf (median).
Usage (GPU): python eda/210_tabpfn_oof.py [est=4] [orig=...]"""
import sys, time, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
KW = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); EST = int(KW.get("est", 4))
T = pd.read_parquet("data/fk_explain_table.parquet"); META = json.load(open("data/fk_explain_meta.json")); COLS = META["FULL"]
for c in META["CAT"]: T[c] = T[c].astype("category").cat.codes.astype(float)
DY = ["dy_nb_sea4", "dy_nb_lvl", "dy_nb_mom3", "dy_nb_yoy"]; T = T.merge(pd.read_parquet("data/fk_dyn_feats.parquet")[["origin", "style_color"] + DY], on=["origin", "style_color"], how="left"); COLS = COLS + DY
T["fold"] = T.style_color.str.split("_").str[0].map(lambda s: sum(map(ord, s)) % 5)
ORIG = KW["orig"].split(",") if "orig" in KW else [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
def pf(a, b):
    m = TabPFNRegressor.create_default_for_version(ModelVersion("v2"), n_estimators=EST, device=KW.get("device", "cuda"), ignore_pretraining_limits=True, random_state=0)
    return m.fit(a[COLS].values, a.z.values).predict(b[COLS].values, output_type="median")
out, t0 = [], time.time()
if KW.get("mode") == "wf":   # walk-forward (user: "tabpfn + twin tree recommender as beginning, with trees and nn stacked secondarily"): TabPFN's
    # prediction at every origin from its own closed past, so a row at origin o' carries the prediction TabPFN made at o' (level-1 input)
    for o in ORIG:
        oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()].reset_index(drop=True); te = T[T.origin == o].reset_index(drop=True)
        if len(tr) < 200: continue
        out.append(pd.DataFrame({"origin": o, "style_color": te.style_color, "z_pf": pf(tr, te)})); print(o, len(tr), "train rows", round(time.time() - t0), "s", flush=True)
        pd.concat(out).to_parquet("out/tabpfn_wf.parquet", index=False)
    print("done", round(time.time() - t0), "s", flush=True); raise SystemExit
for o in ORIG:
    oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()].reset_index(drop=True); te = T[T.origin == o].reset_index(drop=True); zt = np.full(len(tr), np.nan)
    for k in range(5):
        m = (tr.fold == k).values; zt[m] = pf(tr[~m], tr[m])
    out.append(pd.DataFrame({"origin": o, "row_origin": tr.origin, "style_color": tr.style_color, "part": "train", "z_pf": zt}))
    out.append(pd.DataFrame({"origin": o, "row_origin": te.origin, "style_color": te.style_color, "part": "test", "z_pf": pf(tr, te)}))
    print(o, len(tr), "train rows", round(time.time() - t0), "s", flush=True)
    pd.concat(out).to_parquet("out/tabpfn_oof.parquet", index=False)
print("done", round(time.time() - t0), "s", flush=True)
