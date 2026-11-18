"""D276 (fork; user: "full thorough error analysis ... cohorts and potential axes of information"): raw-data axes per item-month, from the
order lines (Dataset.parquet), for the error analysis. One row per (style_color, month), non-cancelled units unless stated:
  units, u_cancel (cancelled units), u_<channel> (edikted.com, Tiktok, RETAIL own stores, each wholesale / marketplace partner),
  n_stores (own stores selling it), n_partners (partners with sales), n_sizes, size_hhi, n_states, state_hhi, u_intl (non-US units),
  disc_web (units-weighted discount on edikted.com), multi_share (units in lines of 2+), wkend_share,
  top_day_share (largest single day / month, all channels), top_hour_share (largest single clock hour / month, online only),
  days_sold (days with any sale).
Writes data/fk_axes_month.parquet."""
import sys, numpy as np, pandas as pd, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
cols = ["ny_timestamp", "style_color", "size", "qty_ordered", "avg_discounts", "class", "o_type", "store", "country", "state", "cancelled"]
T = pq.read_table("Dataset.parquet", columns=cols, read_dictionary=["style_color", "size", "class", "o_type", "store", "country", "state"]).to_pandas()
print("rows", len(T), flush=True)
T["m"] = T.ny_timestamp.dt.to_period("M").astype(str).astype("category"); T["q"] = T.qty_ordered.astype("float32")
ok = ~T.cancelled; K = ["style_color", "m"]
def S(x):   # plain string index, so results from differently filtered frames align
    x = x.copy(); x.index = pd.MultiIndex.from_arrays([x.index.get_level_values(0).astype(str), x.index.get_level_values(1).astype(str)], names=K); return x
g = lambda d, c="q": S(d.groupby(K, observed=True)[c].sum())
A = pd.DataFrame({"units": g(T[ok]), "u_cancel": g(T[~ok])})
ch = S(T[ok].groupby(K + ["o_type"], observed=True).q.sum().unstack(fill_value=0)); ch.columns = ["u_" + str(c).lower().replace(".", "_") for c in ch.columns]
cl = S(T[ok].groupby(K + ["class"], observed=True).q.sum().unstack(fill_value=0)); cl.columns = ["uc_" + str(c).lower() for c in cl.columns]
A = A.join(ch, how="outer").join(cl, how="outer").fillna(0)
ret = T[ok & (T["class"] == "RETAIL")]; A["n_stores"] = S(ret.groupby(K, observed=True).store.nunique())
dr = T[ok & (T["class"] == "DROP")]; A["n_partners"] = S(dr.groupby(K, observed=True).o_type.nunique())
def hhi(d, c):
    s = d.groupby(K + [c], observed=True).q.sum(); t = s.groupby(level=[0, 1], observed=True).transform("sum"); return S(((s / t) ** 2).groupby(level=[0, 1], observed=True).sum())
web = T[ok & (T["class"] == "ONLINE")]
A["n_sizes"] = S(T[ok].groupby(K, observed=True)["size"].nunique()); A["size_hhi"] = hhi(T[ok], "size")
A["n_states"] = S(web.groupby(K, observed=True).state.nunique()); A["state_hhi"] = hhi(web, "state")
A["u_intl"] = g(T[ok & (T.country != "United States")])
w = T[ok & (T.o_type == "edikted.com")]; A["disc_web"] = S((w.q * w.avg_discounts).groupby([w.style_color, w.m], observed=True).sum()) / g(w)
A["multi_share"] = g(T[ok & (T.q > 1)]) / A.units
A["wkend_share"] = g(T[ok & (T.ny_timestamp.dt.dayofweek >= 5)]) / A.units
T["day"] = T.ny_timestamp.dt.floor("D"); T["hour"] = T.ny_timestamp.dt.floor("h")
dd = T[ok].groupby(K + ["day"], observed=True).q.sum(); A["top_day_share"] = S(dd.groupby(level=[0, 1], observed=True).max()) / A.units; A["days_sold"] = S(dd.groupby(level=[0, 1], observed=True).size())
hh = T[ok & (T["class"] == "ONLINE")].groupby(K + ["hour"], observed=True).q.sum(); A["top_hour_share"] = S(hh.groupby(level=[0, 1], observed=True).max()) / S(hh.groupby(level=[0, 1], observed=True).sum())
A = A.reset_index()
A.to_parquet("data/fk_axes_month.parquet", index=False); print(A.shape); print(A.describe().T[["mean", "50%", "max"]].round(3).to_string())
