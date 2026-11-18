"""Data for the interactive series explorer (D102). Writes data/explorer/data.json, read by eda/explorer.html.

Everything is non-cancelled units by NY date unless named otherwise (cancelled units are kept as a separate site series).
- items: master fields + effective start (min(published in NY time, first sale)), macro group, tagger attributes, target flag
- W: weekly units per item by class (ONLINE / RETAIL / DROP), 130 Monday-start weeks 2024-01-01 .. 2026-06-22 (all full weeks)
- WD: weekly sum of discount x units on ONLINE lines per item (group discount = sum WD / sum online units)
- M: monthly units per item by class, 2024-01 .. 2026-06
- daily site series: units by class, cancelled units, edikted.com units and unit-weighted discount, open retail stores, live items
- D: daily units per item for the 100 targets and the items in data/suspected_stockouts.csv
- backtests: monthly LightGBM ensemble (best_backtest_preds) and the 6-seed sequence NN (S13x6 q50), 13 origins x h=1..4
- forecasts: submitted (final_forecast_q4) and candidate (cand_nn_final_forecast_q4) monthly forecasts for the targets
- embeddings: product-type similarity matrix and 2-D item map from eda/74 (if present)
"""
import json, os, numpy as np, pandas as pd, pyarrow.parquet as pq, warnings
warnings.filterwarnings("ignore")
OUT = "data/explorer"; os.makedirs(OUT, exist_ok=True)
D0, D1 = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-06-30")
days = pd.date_range(D0, D1); weeks = pd.date_range(D0, periods=130, freq="7D"); months = pd.period_range("2024-01", "2026-06", freq="M")

# ---------- items
it = pd.read_parquet("data/items.parquet").set_index("style_color")
it["pub"] = pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
it["first_sale"] = pd.to_datetime(it.first_sale); it["last_sale"] = pd.to_datetime(it.last_sale)
it["start"] = it[["pub", "first_sale"]].min(axis=1)
MACRO = {"Top": "top", "Tank Top": "top", "T-Shirt": "top", "Crop Top": "top", "Corset": "top", "Strapless": "top", "Bodysuit": "top", "Shirt": "top", "Knitted Top": "top",
         "Jeans": "bottom", "Shorts": "bottom", "Skirt": "bottom", "Skort": "bottom", "Pants": "bottom", "Short Jeans": "bottom", "Sweatpants": "warm",
         "Hoodie": "warm", "Sweater": "warm", "Jacket": "warm", "Cardigan": "warm", "Dress": "dress", "Swimwear Top": "swim", "Swimwear Bottom": "swim"}
it["macro"] = it.product_type.map(MACRO).fillna("other")
it["occ"] = it.occasion.fillna("none")
it["style"] = it.index.str.split("_").str[0]
T = set(pd.read_csv("data/targets.csv").style_color)
a = pd.read_csv("item_attributes.csv").set_index("style_color").reindex(it.index)
def norm(s, keep):
    s = s.astype("string").str.lower().str.split("|").str[0].str.strip().str.replace(r"\s+", " ", regex=True)
    top = s.value_counts().head(keep).index
    return s.where(s.isin(top), "other").where(s.notna(), "untagged").astype(str)
w = a.fabric_weight.astype("string").str.lower()
attrs = {"fabric weight": np.select([w.str.contains("sheer|ultra|light", na=False), w.str.contains("heavy|chunky|thick", na=False), w.notna()], ["light", "heavy", "medium"], "untagged")}
for c, k, lab in [("fabric_feel", 10, "fabric feel"), ("print_pattern", 8, "print"), ("silhouette", 8, "silhouette"), ("length", 6, "length"), ("occasion_tier", 7, "occasion tier"), ("aesthetics", 10, "aesthetic")]:
    attrs[lab] = norm(a[c], k).values
items = it.index.tolist(); idx = {s: i for i, s in enumerate(items)}
ds = lambda s: [None if pd.isna(x) else x.strftime("%Y-%m-%d") for x in s]
I = {"id": items, "name": it.product_name.tolist(), "type": it.product_type.tolist(), "macro": it.macro.tolist(), "coll": it.product_collection.fillna("none").tolist(),
     "occ": it.occ.tolist(), "style": it["style"].tolist(), "color_num": it.color_num.astype(int).tolist(), "start": ds(it.start), "pub": ds(it.pub), "last": ds(it.last_sale),
     "target": [s in T for s in items], "attrs": {k: list(map(str, v)) for k, v in attrs.items()}}

# ---------- daily item x class
d = pd.read_parquet("data/daily_item_chan.parquet"); d["ny_date"] = pd.to_datetime(d.ny_date)
canc = d[d.cancelled].groupby("ny_date").qty.sum().reindex(days, fill_value=0)
d = d[~d.cancelled]
d["di"] = (d.ny_date - D0).dt.days; d["wi"] = d.di // 7; d["ii"] = d.style_color.map(idx)
d["mi"] = (d.ny_date.dt.year - 2024) * 12 + d.ny_date.dt.month - 1
CL = ["ONLINE", "RETAIL", "DROP"]
W, M = {}, {}
for c in CL:
    x = d[(d["class"] == c) & (d.wi < 130)]
    W[c] = np.zeros((len(items), 130)); np.add.at(W[c], (x.ii.values, x.wi.values), x.qty.values)
    x = d[d["class"] == c]
    M[c] = np.zeros((len(items), 30)); np.add.at(M[c], (x.ii.values, x.mi.values), x.qty.values)
x = d[(d["class"] == "ONLINE") & (d.wi < 130)]
WD = np.zeros((len(items), 130)); np.add.at(WD, (x.ii.values, x.wi.values), x.disc_w.clip(lower=0).values)   # disc_w = sum of discount over units
# daily item units for targets + suspected stock-outs
so = pd.read_csv("data/suspected_stockouts.csv")
dsel = sorted(set(T) | set(so.style_color))
Dd = np.zeros((len(items), len(days))); np.add.at(Dd, (d.ii.values, d.di.values), d.qty.values)
Dsel = {s: Dd[idx[s]].astype(int).tolist() for s in dsel}

# ---------- daily site series
site = {c: d[d["class"] == c].groupby("ny_date").qty.sum().reindex(days, fill_value=0) for c in CL}
web = d[d.chan == "edikted.com"].groupby("ny_date")[["qty", "disc_w"]].sum().reindex(days, fill_value=0)
web_disc = (web.disc_w.clip(lower=0) / web.qty.replace(0, np.nan)).clip(0, 1)
ret = d[d["class"] == "RETAIL"].groupby("ny_date")[["qty", "disc_w"]].sum().reindex(days, fill_value=0)
ret_disc = (ret.disc_w.clip(lower=0) / ret.qty.replace(0, np.nan)).clip(0, 1)
live = np.array([((it.start <= t) & (it.last_sale >= t)).sum() for t in days])
chan_m = d.groupby([d.chan, d.mi]).qty.sum().unstack(fill_value=0).reindex(columns=range(30), fill_value=0)
chan_m = chan_m[chan_m.sum(1) > 500]
del d
r = pq.read_table("Dataset.parquet", columns=["ny_date", "class", "store", "cancelled", "qty_ordered"]).to_pandas()
r = r[(r["class"] == "RETAIL") & ~r.cancelled]
st = r.groupby("store").agg(first=("ny_date", "min"), last=("ny_date", "max"), units=("qty_ordered", "sum")).sort_values("first")
st = st[st.units > 0]; del r
st["first"] = pd.to_datetime(st["first"]); st["last"] = pd.to_datetime(st["last"])
stores_open = np.array([((st["first"] <= t) & (st["last"] >= t)).sum() for t in days])

# ---------- backtests (monthly by item x origin x h)
b = pd.read_parquet("data/best_backtest_preds.parquet")
n = pd.read_parquet("data/nn_backtest_preds_S13x6.parquet")[["origin", "style_color", "h", "q50", "level"]]
b = b.merge(n, on=["origin", "style_color", "h"], how="left")
origins = sorted(b.origin.unique())
B = {"origins": origins, "o": b.origin.map({o: i for i, o in enumerate(origins)}).astype(int).tolist(), "i": b.style_color.map(idx).astype(int).tolist(),
     "h": b.h.astype(int).tolist(), "y": b.y.round(0).astype(int).tolist(), "lgb": b.pred.round(1).tolist(), "nn": b.q50.round(1).tolist(), "level": b.level.round(1).tolist()}

# ---------- final forecasts for the targets
def fc(path):
    f = pd.read_csv(path, index_col=0)
    return {s: [int(f.at[s, c]) for c in ["f_jul", "f_aug", "f_sep", "f_oct"]] + [int(f.at[s, "conf"])] for s in f.index}
F = {"submitted": fc("data/final_forecast_q4.csv"), "candidate": fc("data/cand_nn_final_forecast_q4.csv")}

# ---------- embeddings (eda/74)
E = None
if os.path.exists(f"{OUT}/ptype_sim.csv"):
    S = pd.read_csv(f"{OUT}/ptype_sim.csv", index_col=0)
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    o = leaves_list(linkage(squareform((1 - S.values).clip(0) * (1 - np.eye(len(S))), checks=False), "average")); S = S.iloc[o, o]   # clustered order
    v2 = pd.read_csv(f"{OUT}/item_vec2d.csv", index_col=0).reindex(items)
    E = {"types": S.index.tolist(), "sim": S.values.round(3).tolist(), "x": [None if pd.isna(v) else round(float(v), 3) for v in v2.x], "y": [None if pd.isna(v) else round(float(v), 3) for v in v2.y]}

plan = [("Summer Sale", "2026-06-28", "2026-07-19"), ("Back to School", "2026-07-20", "2026-08-31"), ("Clearance", "2026-08-12", "2026-08-19"), ("Labor Day", "2026-09-03", "2026-09-08"),
        ("Sale80", "2026-09-24", "2026-09-29"), ("Sale80", "2026-10-07", "2026-10-18")]
ri = lambda A: A.round(0).astype(int).tolist()
out = {"built": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "d0": "2024-01-01", "ndays": len(days), "nweeks": 130,
       "months": [str(m) for m in months], "items": I,
       "W": {c: ri(W[c]) for c in CL}, "WD": ri(WD * 100), "M": {c: ri(M[c]) for c in CL},
       "site": {"units": {c: ri(site[c].values) for c in CL}, "cancelled": ri(canc.values), "web_units": ri(web.qty.values),
                "web_disc": [None if pd.isna(v) else round(float(v), 4) for v in web_disc.values], "ret_disc": [None if pd.isna(v) else round(float(v), 4) for v in ret_disc.values], "stores": stores_open.tolist(), "live": live.tolist()},
       "stores": [{"store": s, "first": r_["first"].strftime("%Y-%m-%d"), "last": r_["last"].strftime("%Y-%m-%d"), "units": int(r_.units)} for s, r_ in st.iterrows()],
       "chan_m": {c: ri(chan_m.loc[c].values) for c in chan_m.index},
       "D": Dsel, "stockouts": so.to_dict("records"),
       "oos_runs": pd.read_csv("data/oos_runs.csv")[["style_color", "start", "end", "days"]].values.tolist(),
       "bt": B, "fc": F, "emb": E, "plan": plan}
s = json.dumps(out, separators=(",", ":"), allow_nan=False)
open(f"{OUT}/data.json", "w", encoding="utf-8").write(s)
print(f"wrote {OUT}/data.json: {len(s)/1e6:.1f} MB | items {len(items)} | daily items {len(Dsel)} | stores {len(st)} | chans {len(chan_m)} | emb {'yes' if E else 'no'}")
# checks against the report
tot = sum(np.array(out["site"]["units"][c]) for c in CL)
print("total units", int(tot.sum()), "| 2025-11-28:", int(tot[(pd.Timestamp('2025-11-28') - D0).days]), "| cancelled 2025-10-14:", int(canc.loc['2025-10-14']),
      "| stores open 2024-01-15 / 2026-06-30:", stores_open[14], stores_open[-1])
