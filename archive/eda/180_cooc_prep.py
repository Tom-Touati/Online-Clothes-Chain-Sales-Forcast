"""D193 (fork): inputs for the dynamic co-occurrence network (eda/181, remote). Exports
  data/fk_cooc_inputs.npz: items, months, X (descriptor matrix: tagger one-hots as in eda/153 + 14 LLM scores, standardised),
  R (months x items: orders containing the item), N (months: online orders), multi (months x items: orders with another item),
  and the pair counts per month as (month index, i, j, n) arrays."""
import sys, glob, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
items = np.sort(pd.read_parquet("data/items.parquet").style_color.values); n = len(items)
src = open("eda/153_basket_emb.py", encoding="utf-8").read(); exec(src[src.index("it = pd.read_parquet"):src.index("P = pd.read_parquet")])
Lr = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("data/llm/scores_part*.csv"))]).drop_duplicates("style_color").set_index("style_color").reindex(items)
X = np.hstack([ATT, Lr.values]); X = ((X - X.mean(0)) / np.maximum(X.std(0), 1e-6)).astype(np.float32)
P = pd.read_parquet("data/basket_pairs_month.parquet"); B = pd.read_parquet("data/basket_item_month.parquet")
months = sorted(B.m.unique()); mi = {m: k for k, m in enumerate(months)}
R = np.zeros((len(months), n), np.float32); Mu = np.zeros((len(months), n), np.float32)
R[B.m.map(mi).values, B.it.values] = B.orders.values; Mu[B.m.map(mi).values, B.it.values] = B.multi_orders.values
d = pd.read_parquet("Dataset.parquet", columns=["order_name", "ny_date", "class", "cancelled"]); d = d[(~d.cancelled) & (d["class"] == "ONLINE")]
d["m"] = pd.PeriodIndex(pd.to_datetime(d.ny_date), freq="M").astype(str); N = d.groupby("m").order_name.nunique().reindex(months).values.astype(np.float64)
np.savez_compressed("data/fk_cooc_inputs.npz", items=items.astype(str), months=np.array(months), X=X, R=R, N=N, multi=Mu,
                    pm=P.m.map(mi).values.astype(np.int16), pi=P.it.values.astype(np.int16), pj=P.it_b.values.astype(np.int16), pn=P.n.values.astype(np.float32))
print(len(months), "months", X.shape, "pairs", len(P), "orders/month", N[:3], N[-3:])
