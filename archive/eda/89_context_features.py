"""M5-style context features at each origin (D112), one row per item x origin month-end; data <= origin only.
Occasion timing (the item's own tags; NaN when untagged):
  occ_next  = days from the origin to the next date of any of the item's occasions
  occ_last  = days since the most recent date of any of its occasions
  occ_n4    = number of its occasions falling in the next 122 days (the 4-month window)
  Dates: Valentine's Feb 14, St. Patrick's Mar 17, Graduation Jun 1 (season anchor), 4th of July Jul 4, Homecoming Oct 1 (season anchor),
  Halloween Oct 31, New Year's Eve Dec 31, World Cup 2022-11-20 and 2026-06-11 (tournament starts).
Discount momentum (edikted.com + other ONLINE lines, unit-weighted discount d; price index = 1 - d, as M5's price momentum):
  dm_7_28   = log((1 - d_7) / (1 - d_28))            last week against the last four
  dm_28_p91 = log((1 - d_28) / (1 - d_prev91))        last four weeks against the 91 days before them
  dm_vs_min = log((1 - d_28) / (1 - d_min))           against the item's lowest monthly discount so far (its 'full price', months with >= 10 units)
  s_dm_28_p91 = the same as dm_28_p91 for the whole site
Writes data/context_feats.parquet."""
import numpy as np, pandas as pd
it = pd.read_parquet("data/items.parquet").set_index("style_color")
D = {"Valentine's Day": [(2, 14)], "St. Patrick's Day": [(3, 17)], "Graduation": [(6, 1)], "4th of July": [(7, 4)], "Homecoming": [(10, 1)],
     "Halloween": [(10, 31)], "New Year's Eve": [(12, 31)]}
WC = [pd.Timestamp("2022-11-20"), pd.Timestamp("2026-06-11")]
def dates(tag):
    if tag == "World Cup": return WC
    return [pd.Timestamp(y, m, d) for y in range(2022, 2028) for m, d in D[tag]]
TAGS = {s: [t.strip() for t in str(v).split(";") if t.strip() in D or t.strip() == "World Cup"] for s, v in it.occasion.items() if pd.notna(v)}
d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[(~d.cancelled) & (d["class"] == "ONLINE")]; d["ny_date"] = pd.to_datetime(d.ny_date)
days = pd.date_range("2024-01-01", "2026-06-30"); items = sorted(set(d.style_color) | set(it.index))
def cum(col):
    m = d.groupby(["style_color", "ny_date"])[col].sum().unstack(fill_value=0).reindex(index=items, columns=days, fill_value=0).values
    return np.concatenate([np.zeros((len(items), 1)), np.cumsum(m, axis=1)], axis=1)
Cq, Cw = cum("qty"), cum("disc_w")
mq = d.assign(m=d.ny_date.dt.to_period("M")).groupby(["style_color", "m"])[["qty", "disc_w"]].sum()
mdisc = (mq.disc_w / mq.qty).where(mq.qty >= 10).unstack().reindex(items)
win = lambda C, e, a, b: C[..., e + 1 - a] - C[..., max(e + 1 - b, 0)]          # days (a, b] before the end, a < b
rows = []
for o in pd.period_range("2024-03", "2026-06", freq="M"):
    O = o.end_time.normalize(); e = (O - days[0]).days
    q7, w7 = win(Cq, e, 0, 7), win(Cw, e, 0, 7); q28, w28 = win(Cq, e, 0, 28), win(Cw, e, 0, 28); qp, wp = win(Cq, e, 28, 119), win(Cw, e, 28, 119)
    with np.errstate(divide="ignore", invalid="ignore"):
        d7, d28, dp = w7 / q7, w28 / q28, wp / qp
        dmin = mdisc.loc[:, [c for c in mdisc.columns if c <= o]].min(axis=1).values
        X = pd.DataFrame({"origin": str(o), "style_color": items,
                          "dm_7_28": np.where(q7 >= 3, np.log((1 - d7) / (1 - d28)), np.nan),
                          "dm_28_p91": np.where((q28 >= 5) & (qp >= 5), np.log((1 - d28) / (1 - dp)), np.nan),
                          "dm_vs_min": np.where(q28 >= 5, np.log((1 - d28) / (1 - dmin)), np.nan)})
        X["s_dm_28_p91"] = np.log((1 - w28.sum() / q28.sum()) / (1 - wp.sum() / qp.sum()))
    nx, ls, n4 = [], [], []
    for s in items:
        t = TAGS.get(s, [])
        if not t: nx.append(np.nan); ls.append(np.nan); n4.append(np.nan); continue
        ds = [x for tg in t for x in dates(tg)]
        fut = [(x - O).days for x in ds if x > O]; past = [(O - x).days for x in ds if x <= O]
        nx.append(min(fut) if fut else np.nan); ls.append(min(past) if past else np.nan); n4.append(sum(1 for f in fut if f <= 122))
    X["occ_next"], X["occ_last"], X["occ_n4"] = nx, ls, n4
    rows.append(X)
CF = pd.concat(rows, ignore_index=True); CF.to_parquet("data/context_feats.parquet", index=False)
print(CF.describe().T.round(3).to_string())
print(CF[CF.origin == "2025-06"].dropna(subset=["occ_next"]).groupby("occ_n4").size())
