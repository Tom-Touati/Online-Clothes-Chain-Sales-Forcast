"""D264 (idea 5 of 18 Nov): the launch plan at look level instead of product-type level (D253: type-level PLAN hurt, 0.4632).
Known only if the merchandising launch calendar is known at the origin (launch dates from the catalogue). For each item at origin o,
the launches with first sale in (o, o + 4 months]:
  pl_same_style   new colours of the item's own style launching in the window
  pl_sim85 / pl_sim75   launches whose product photo + text embedding (CLIP, static) has cosine >= 0.85 / 0.75 with the item
  pl_simsum       sum over launches of max(0, cosine - 0.6) (weighted look-alike launch pressure)
  pl_type_sim     launches of the same product type with cosine >= 0.75
Writes data/nf4_feats.parquet."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
T = pd.read_parquet("data/fk_explain_table.parquet")[["origin", "style_color"]]
it = pd.read_parquet("data/items.parquet").set_index("style_color"); items = np.sort(it.index.values); ix = {s: i for i, s in enumerate(items)}
pub = pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
start = pd.concat([pub, pd.to_datetime(it.first_sale)], axis=1).min(axis=1).reindex(items)
fm = np.array([str(pd.Period(s, "M")) if pd.notna(s) else "9999-12" for s in start]); sty = pd.Series(items).str.split("_").str[0].values
pt = it.product_type.reindex(items).fillna("na").values
CE = np.load("data/content_emb.npz", allow_pickle=True); cix = pd.Series(range(len(CE["items"])), index=CE["items"].astype(str)).reindex(items).values
Z = np.hstack([CE["img"][cix] / np.linalg.norm(CE["img"][cix], axis=1, keepdims=True), CE["txt"][cix] / np.linalg.norm(CE["txt"][cix], axis=1, keepdims=True)]) / np.sqrt(2)
Z = np.nan_to_num(Z); S = Z @ Z.T; np.fill_diagonal(S, 0)   # items without photo / text: similarity 0
rows = []
for o, g in T.groupby("origin"):
    P = pd.Period(o, "M"); fut = (fm > o) & (fm <= str(P + 4)); idx = np.array([ix[s] for s in g.style_color]); Sf = S[np.ix_(idx, np.where(fut)[0])]
    fs, ft = sty[fut], pt[fut]
    rows.append(pd.DataFrame({"origin": o, "style_color": g.style_color.values,
        "pl_same_style": [(fs == sty[i]).sum() for i in idx], "pl_sim85": (Sf >= 0.85).sum(1), "pl_sim75": (Sf >= 0.75).sum(1),
        "pl_simsum": np.clip(Sf - 0.6, 0, None).sum(1), "pl_type_sim": [((Sf[j] >= 0.75) & (ft == pt[i])).sum() for j, i in enumerate(idx)]}))
NF = pd.concat(rows, ignore_index=True); NF.to_parquet("data/nf4_feats.parquet"); print(NF.describe().round(3).T)
