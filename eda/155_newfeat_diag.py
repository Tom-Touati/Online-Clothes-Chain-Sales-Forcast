"""D150/D152 diagnostic: do the co-purchase neighbour features and the regional features carry information the three-way lacks?
Correlation of each feature (at the origin, time-honest) with the three-way's item log residual log((y + 1) / (forecast + 1)) on the
nine recent origins, raw and partial (after regressing both on the item's own 3-month momentum and the three-way's log ratio to R3),
for all items and for young items (under 6 months). Spearman, item-origins with >= 20 actual units."""
import sys, contextlib, io, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding="utf-8")
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
Pm = lambda o, k: str(pd.Period(o, "M") + k)
T["res"] = np.log((T.y + 1) / (T.r_q_nn + 1)); T["fr3"] = np.log((T.r_q_nn + 1) / (T.R3 + 1))
T["own_mom3"] = [np.log((M.loc[s, [Pm(o, -k) for k in range(3)]].sum() + 1) / (M.loc[s, [Pm(o, -k) for k in range(3, 6)]].reindex().fillna(0).sum() + 1)) if s in M.index else np.nan
                 for o, s in zip(T.origin, T.style_color)]
it = pd.read_parquet("data/items.parquet").set_index("style_color"); fs = pd.to_datetime(it.first_sale)
T["age"] = [(pd.Period(o, "M").to_timestamp(how="end") - fs.get(s)).days / 30.4 for o, s in zip(T.origin, T.style_color)]
key = ["origin", "style_color"]
for f in ["data/basket_feats_cp.parquet", "data/basket_feats_attr.parquet", "data/region_feats.parquet"]:
    X = pd.read_parquet(f); tag = f.split("/")[-1].replace(".parquet", "").replace("basket_feats_", "b_").replace("region_feats", "reg")
    X = X.rename(columns={c: f"{tag}:{c}" for c in X.columns if c not in key}); T = T.merge(X, on=key, how="left")
REC_ = [o for o in ORIGINS if o >= "2025-06"]; F = T[T.origin.isin(REC_) & (T.y >= 20)].copy()
FEATS = [c for c in F.columns if ":" in c]
def partial(a, b, Z):
    m = np.isfinite(a) & np.isfinite(b) & np.all(np.isfinite(Z), 1)
    if m.sum() < 50: return np.nan, int(m.sum())
    Zm = np.column_stack([np.ones(m.sum()), Z[m]]); ra = a[m] - Zm @ np.linalg.lstsq(Zm, a[m], rcond=None)[0]; rb = b[m] - Zm @ np.linalg.lstsq(Zm, b[m], rcond=None)[0]
    return spearmanr(ra, rb)[0], int(m.sum())
print("feature | all items: raw rho, partial rho, n | young (< 6 months): raw, partial, n")
for c in FEATS:
    out = []
    for D in [F, F[F.age < 6]]:
        a = D[c].values.astype(float); b = D.res.values; ok = np.isfinite(a)
        raw = spearmanr(a[ok], b[ok])[0] if ok.sum() > 50 else np.nan
        p, nn_ = partial(a, b, D[["own_mom3", "fr3"]].values.astype(float)); out += [raw, p, nn_]
    print(f"{c:28s} | {out[0]:+.3f} {out[1]:+.3f} {out[2]:5d} | {out[3]:+.3f} {out[4]:+.3f} {out[5]:5d}")
