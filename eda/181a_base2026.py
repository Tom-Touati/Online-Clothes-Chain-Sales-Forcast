"""D262: June 2026 rows for the correction re-fits, appended to data/dec_base_ec_ns.parquet -> data/dec_base_ec_ns26.parquet.
Members at the 2026-06 origin (4-month totals): rules (agent_eval/2026-06, months summed), the D253 tree member at 2026-06
(data/q_backtest_nx_ec_ns_2026.parquet; column 'qdyn' as in dec_base_ec_ns), NN (finalS + finalS_s10 averaged), conformal TabPFN and
TabPFN dyn 32 (data/fk_2026_{pf2c,pf2d32}.parquet). Missing rules / NN are filled with the mean of the other members (as eda/179f).
F = arithmetic mean; R3 = 4 x mean(Apr-Jun 2026); descriptors as eda/179d_base.py; look clusters by the same k-means (K = 40) at 2026-06.
y, e, g, ae are missing (the outcome is unknown). The corrections then run with DEC=_ec_ns26 (they train only on o' + 4 <= 2026-06)."""
import sys, numpy as np, pandas as pd
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import KMeans
sys.stdout.reconfigure(encoding="utf-8")
import os; NNS = os.environ.get("NNS", "nn_backtest_preds_finalS,nn_backtest_preds_finalS_s10").split(","); SRC = os.environ.get("SRC", "ec_ns")   # D273: NNS=..finalSec,..finalSec_s10 SRC=ecnsnn
O = "2026-06"; B0 = pd.read_parquet(f"data/dec_base_{SRC}.parquet"); MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]
X = pd.read_parquet("data/fk_explain_table.parquet"); X = X[X.origin == O][["origin", "style_color", "level", "age_m", "R3", "target"]].copy()
it = pd.read_parquet("data/items.parquet").set_index("style_color")
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
X["R3"] = 4 * M.reindex(X.style_color)[["2026-04", "2026-05", "2026-06"]].mean(1).values
ru = pd.read_csv("agent_eval/2026-06/predictions_verified.csv").groupby("style_color").pred.sum()
nn = sum(pd.read_parquet(f"data/{f}.parquet").groupby("style_color").q50.sum() for f in NNS) / 2
q = lambda f: pd.read_parquet(f).drop_duplicates("style_color").set_index("style_color").q50
X["rules"] = ru.reindex(X.style_color).values; X["qdyn"] = q(os.environ.get("MEMF", "data/q_backtest_nx_ec_ns_2026.parquet")).reindex(X.style_color).values   # D356: MEMF picks the tree member
X["nn"] = nn.reindex(X.style_color).values; X["pf2c"] = q("data/fk_2026_pf2c.parquet").reindex(X.style_color).values; X["pf2d32"] = q("data/fk_2026_pf2d32.parquet").reindex(X.style_color).values
for c in ["rules", "nn"]: X[c] = X[c].fillna(X[["qdyn", "pf2c", "pf2d32"]].mean(1))
if os.environ.get("RC") == "1":   # D356: Fork 2's recommender as a sixth member (rc_deepfm at 2026-06)
    _rc = pd.read_parquet("data/rc_recsys_preds.parquet"); _rc = _rc[_rc.origin == O].drop_duplicates("style_color").set_index("style_color").rc_deepfm
    X["rc"] = _rc.reindex(X.style_color).values; X["rc"] = X.rc.fillna(X[MEM].mean(1)); MEM = MEM + ["rc"]
_W = np.array([1.0] * (len(MEM) - 1) + [float(os.environ.get("RCW", 1.0))]) if "rc" in MEM else np.ones(len(MEM))   # D360: RCW = the recommender's weight
X["F"] = (X[MEM].clip(lower=0).values * _W).sum(1) / _W.sum()
X["ptype"] = it.product_type.reindex(X.style_color).fillna("na").values; X["style"] = X.style_color.str.split("_").str[0]
MAC = {"Top": "top", "Tank Top": "top", "T-Shirt": "top", "Crop Top": "top", "Corset": "top", "Strapless": "top", "Bodysuit": "top", "Shirt": "top", "Knitted Top": "top",
       "Jeans": "bottom", "Shorts": "bottom", "Skirt": "bottom", "Skort": "bottom", "Pants": "bottom", "Short Jeans": "bottom", "Sweatpants": "warm",
       "Hoodie": "warm", "Sweater": "warm", "Jacket": "warm", "Cardigan": "warm", "Dress": "dress", "Swimwear Top": "swim", "Swimwear Bottom": "swim"}
X["macro"] = X.ptype.map(MAC).fillna("other"); X["bf_window"] = 0
X["spread"] = np.log(X[MEM].clip(lower=1).max(1) / X[MEM].clip(lower=1).min(1))
F0 = pd.read_parquet("data/features.parquet"); F0 = F0[(F0.h == 1) & (F0.origin == O)][["origin", "style_color", "sh_tt", "disc_item", "own_sea", "trend91", "f_r28"]]
X = X.merge(F0, on=["origin", "style_color"], how="left")
items = np.sort(it.index.values); n = len(items); P = pd.read_parquet("data/basket_pairs_month.parquet"); Pm = lambda o, k: str(pd.Period(o, "M") + k)
C = np.zeros((n, n), np.float32)
for m, g in P[(P.m >= Pm(O, -11)) & (P.m <= O)].groupby("m"): c = np.zeros((n, n), np.float32); c[g.it.values, g.it_b.values] = g.n.values; C += c + c.T
CE = np.load("data/content_emb.npz", allow_pickle=True); cix = pd.Series(range(len(CE["items"])), index=CE["items"].astype(str)).reindex(items).values
CON = np.hstack([CE["img"][cix], CE["txt"][cix]]); u, s_, _ = randomized_svd(CON, 24, random_state=0); CONV = u * s_
rown = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
r = C.sum(1); tot = max(r.sum(), 1); ctx = r ** 0.75; ctx = ctx / max(ctx.sum(), 1e-9)
with np.errstate(divide="ignore", invalid="ignore"): pmi = np.nan_to_num(np.maximum(np.log((C / tot) / np.outer(r / tot, ctx)), 0), posinf=0, neginf=0)
uu, ss, _ = randomized_svd(rown(pmi), 24, random_state=0); Z = np.hstack([np.sqrt(0.5) * rown(uu * np.sqrt(ss)), np.sqrt(0.5) * rown(CONV)])
lab = pd.Series(KMeans(40, n_init=4, random_state=0).fit_predict(Z), index=items)
X["look"] = O + "_L" + lab.reindex(X.style_color).astype(str).values
for c in ["y", "e", "g", "ae"]: X[c] = np.nan
B = pd.concat([B0, X[[c for c in B0.columns if c in X.columns]]], ignore_index=True); B.to_parquet(f"data/dec_base_{SRC}26.parquet", index=False)
TG = X[X.target]; print(f"2026-06 rows {len(X)}, targets {len(TG)}; F on targets {TG.F.sum():,.0f} (members: " + ", ".join(f"{m} {TG[m].sum():,.0f}" for m in MEM) + f"); R3 {TG.R3.sum():,.0f}")
