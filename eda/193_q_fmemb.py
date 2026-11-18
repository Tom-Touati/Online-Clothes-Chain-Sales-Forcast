"""D198 (fork): time-series foundation-model embeddings as tree features (remote). The D159 quantile model (lean + occasion timing +
cart ramp + LLM scores + temperature + events; features from data/fk_explain_table.parquet) is refitted at the 13 backtest origins as
eda/20b does (7 quantiles, the two feature sets averaged, sorted), with extra features:
  none        no extra features (reproduces the D159 model)
  c2pca       16 principal components of the Chronos-2 encoder embedding of the item's weekly series at the origin
  boltpca     16 principal components of the Chronos-Bolt embedding
  dyn / dynrep  the dynamic co-occurrence neighbour features (eda/182) added / replacing the static cart features (D198)
  dyn_fa / dyn_ch / dyn_nn / dyn_all   D205: dyn + the fashion-attribute trend / cluster-history / 5-NN blocks of eda/182b, or all three
The embedding space is pretrained and fixed, so coordinates mean the same at every origin; PCA is fitted on the 2024 origins only
(before any test origin). Embeddings exist for every origin 2024-03 .. 2026-06 (eda/184 allorig).
Writes out/q_backtest_fm_<variant>.parquet. Usage: python 193_q_fmemb.py [variants...]"""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.decomposition import PCA
T = pd.read_parquet("data/fk_explain_table.parquet"); META = json.load(open("data/fk_explain_meta.json"))
BASE0, FULL0, P = META["BASE"], META["FULL"], dict(META["PARAMS"]); P["n_jobs"] = 4
for c in META["CAT"]: T[c] = T[c].astype("category")
QS = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]


def emb_feats(path, tag, k=16):
    Z = np.load(path, allow_pickle=True); origins = sorted({x for x in Z.files if not x.endswith("_items")})
    fit = np.concatenate([Z[o].astype(np.float32) for o in origins if o < "2025-01"]); pca = PCA(k, random_state=0).fit(fit)
    rows = []
    for o in origins:
        E = pca.transform(Z[o].astype(np.float32)); d = pd.DataFrame(E, columns=[f"{tag}{j}" for j in range(k)]); d.insert(0, "style_color", Z[o + "_items"]); d.insert(0, "origin", o); rows.append(d)
    print(tag, "PCA variance explained", pca.explained_variance_ratio_.sum().round(3), flush=True)
    return pd.concat(rows, ignore_index=True), [f"{tag}{j}" for j in range(k)]


VAR = {"none": None, "c2pca": ("out/fk_tsfm_emb_chronos2_all.npz", "c2_"), "boltpca": ("out/fk_tsfm_emb_bolt_all.npz", "bo_")}
DY = ["dy_nb_sea4", "dy_nb_lvl", "dy_nb_mom3", "dy_nb_yoy"]
FEATVAR = {"dyn": ("data/fk_dyn_feats.parquet", DY, []), "dynrep": ("data/fk_dyn_feats.parquet", DY, ["cp_nb_sea4", "cp_nb_lvl"])}   # D198: dynamic co-occurrence
FM_ = json.load(open("data/fk_fash_meta.json")) if __import__("os").path.exists("data/fk_fash_meta.json") else {"FA": [], "CH": [], "NN": []}   # D205: eda/182b
FF = "data/fk_fash_feats.parquet"
FEATVAR |= {"dyn_fa": ([("data/fk_dyn_feats.parquet", DY), (FF, FM_["FA"])], []), "dyn_ch": ([("data/fk_dyn_feats.parquet", DY), (FF, FM_["CH"])], []),
            "dyn_nn": ([("data/fk_dyn_feats.parquet", DY), (FF, FM_["NN"])], []), "dyn_all": ([("data/fk_dyn_feats.parquet", DY), (FF, FM_["FA"] + FM_["CH"] + FM_["NN"])], [])}
FL_ = FM_.get("FL", []); FF2 = "data/fk_fash2_feats.parquet"   # D205b: eda/182c cycle / combination / regional-diffusion block
FEATVAR |= {"dyn_fl": ([("data/fk_dyn_feats.parquet", DY), (FF2, FL_)], []), "dyn_all2": ([("data/fk_dyn_feats.parquet", DY), (FF, FM_["FA"] + FM_["CH"] + FM_["NN"]), (FF2, FL_)], [])}
XF = lambda cols: [(FF, [c for c in cols if not c.startswith("fl_")]), (FF2, [c for c in cols if c.startswith("fl_")])]   # D205c: early-origin selected sets
FEATVAR |= {"dyn_lean": ([("data/fk_dyn_feats.parquet", DY)] + XF(FM_.get("LEAN", [])), []), "dyn_core": ([("data/fk_dyn_feats.parquet", DY)] + XF(FM_.get("CORE", [])), [])}
FNC = "data/fk_nnc_feats.parquet"   # D205d: 5-NN in Main's v4 content embedding (eda/182d)
FEATVAR |= {"dyn_nc": ([("data/fk_dyn_feats.parquet", DY), (FNC, FM_.get("NC", []))], []), "dyn_ncsel": ([("data/fk_dyn_feats.parquet", DY), (FNC, FM_.get("NCSEL", []))], [])}
for var in sys.argv[1:] or list(VAR) + list(FEATVAR):
    TT, NEW, DROP = T, [], []
    if var in FEATVAR:
        spec = FEATVAR[var]
        if len(spec) == 3: spec = ([(spec[0], spec[1])], spec[2])
        srcs, DROP = spec; TT = T
        for path, cols in srcs: TT = TT.merge(pd.read_parquet(path)[["origin", "style_color"] + cols], on=["origin", "style_color"], how="left"); NEW = NEW + cols
    elif VAR[var]: X, NEW = emb_feats(*VAR[var]); TT = T.merge(X, on=["origin", "style_color"], how="left")
    BASE, FULL = [c for c in BASE0 if c not in DROP] + NEW, [c for c in FULL0 if c not in DROP] + NEW; out, t0 = [], time.time()
    for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
        oo = pd.Period(o).ordinal; tr = TT[(TT.o_ord + 4 <= oo) & TT.ytot.notna()]; te = TT[TT.origin == o]
        z = np.zeros((len(te), len(QS)))
        for cols in (BASE, FULL):
            for j, q in enumerate(QS): z[:, j] += lgb.LGBMRegressor(objective="quantile", alpha=q, **P).fit(tr[cols], tr.z).predict(te[cols]) / 2
        z = np.sort(z, axis=1); R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
        for j, q in enumerate(QS): R[f"q{int(q * 100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(z[:, j]) - 1, 0, None)
        out.append(R)
    pd.concat(out).to_parquet(f"out/q_backtest_fm_{var}.parquet"); print(var, len(NEW), "extra features", round(time.time() - t0), "s", flush=True)
