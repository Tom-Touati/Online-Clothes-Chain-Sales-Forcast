"""D181 (overnight M5 goal, direction 6): Google Trends search interest for the fashion vocabulary of the catalogue.
Terms: the most common tagger aesthetics ("<x> aesthetic"), trend tags, fabrics, silhouettes and product types (each term covering
>= 15 items), fetched one term per request (weekly US interest, 2022-01 .. 2026-10, 0-100 relative to the term's own peak).
Only the search terms are sent to Google. Weekly values are averaged to months.
Item features at each origin (data up to the origin month only), averaged over the terms the item carries:
  gt_mom3   log(mean interest last 3 months / mean of the 3 before)       gt_yoy  log(last 3 months / same 3 months a year before)
  gt_ramp   log(interest in the 4 target months last year / the 3 origin months last year): the search seasonality of its looks
  gt_lvl    mean interest last 3 months (0-100 scale of each term)
Writes data/gtrends_terms_month.parquet and data/gtrends_feats.parquet."""
import sys, time, re, numpy as np, pandas as pd
from pytrends.request import TrendReq
sys.stdout.reconfigure(encoding="utf-8")
A = pd.read_csv("item_attributes.csv").set_index("style_color"); it = pd.read_parquet("data/items.parquet").set_index("style_color")
items = np.sort(it.index.values); A = A.reindex(items)
def split(s): return [t.strip() for t in str(s).split("|") if t.strip() and t.strip().lower() != "nan"]
TERMS = {}
for col, fmt in [("aesthetics", "{} aesthetic"), ("trend_tags", "{}"), ("fabric_feel", "{} clothing"), ("silhouette", "{} {}")]:
    vc = pd.Series([t for v in A[col] for t in split(v)]).value_counts()
    for t in vc[vc >= 15].index[:20]: TERMS[(col, t)] = fmt.format(t, "") .strip() if col != "silhouette" else None
for t in it.product_type.reindex(items).value_counts().head(20).index: TERMS[("product_type", t)] = t.lower()
TERMS = {k: v for k, v in TERMS.items() if v}
print(len(TERMS), "terms:", list(TERMS.values())[:12], flush=True)
p = TrendReq(hl="en-US", tz=300, timeout=(10, 30)); out = {}
for (col, t), q in TERMS.items():
    for a in range(5):
        try:
            p.build_payload([q], timeframe="2022-01-01 2026-10-31", geo="US"); d = p.interest_over_time()
            if len(d): out[(col, t)] = d[q].astype(float); break
        except Exception as e: time.sleep(20 * (a + 1))
    time.sleep(4)
    print(f"{q:30s} {'ok' if (col, t) in out else 'FAILED'}", flush=True)
TM = pd.DataFrame({f"{c}|{t}": s for (c, t), s in out.items()}); TM.index = pd.to_datetime(TM.index)
TMm = TM.resample("MS").mean(); TMm.index = TMm.index.to_period("M").astype(str); TMm.to_parquet("data/gtrends_terms_month.parquet")
mem = {}
for (c, t) in out:
    if c == "product_type": m = (it.product_type.reindex(items) == t).values
    else: m = np.array([t in split(v) for v in A[c]])
    mem[f"{c}|{t}"] = m
Mm = pd.DataFrame(mem, index=items)
Pm = lambda o, k: str(pd.Period(o, "M") + k); rows = []
for o in [str(x) for x in pd.period_range("2024-03", "2026-06", freq="M")]:
    g = lambda ks: TMm.reindex([Pm(o, k) for k in ks]).mean()
    l3, p3, y3 = g([0, -1, -2]), g([-3, -4, -5]), g([-12, -13, -14]); t4 = g([-11, -10, -9, -8])
    tf = pd.DataFrame({"gt_mom3": np.log((l3 + 1) / (p3 + 1)), "gt_yoy": np.log((l3 + 1) / (y3 + 1)), "gt_ramp": np.log((t4 + 1) / (y3 + 1)), "gt_lvl": l3})
    W = Mm[tf.index].values.astype(float); cnt = W.sum(1, keepdims=True)
    F = pd.DataFrame((W @ tf.fillna(0).values) / np.maximum(cnt, 1), columns=tf.columns, index=items); F[cnt[:, 0] == 0] = np.nan
    F["origin"] = o; rows.append(F.reset_index().rename(columns={"index": "style_color"}))
pd.concat(rows, ignore_index=True).to_parquet("data/gtrends_feats.parquet", index=False); print("saved; terms fetched", len(out), "of", len(TERMS))
