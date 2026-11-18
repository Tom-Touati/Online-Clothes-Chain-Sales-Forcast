"""Retail rollout features (D122), per item x origin month-end, data <= origin only (retail = class RETAIL, by store):
  rr_n28      stores that sold the item in the last 28 days
  rr_cov      rr_n28 / stores open (stores with any sale in the last 28 days)
  rr_dn       rr_n28 minus the same count for the 28 days before (rollout momentum)
  rr_rate     the item's retail units per selling store per day, last 28 days (log1p)
  rr_age      days since the item's first retail sale (NaN if none: online-only so far)
  rr_newshare share of the item's last-28-day retail units sold in stores opened within the last 180 days
  s_open      stores open (site), s_new90 stores opened in the last 90 days (site)
Writes data/rollout_feats.parquet. Usage: python eda/107_retail_rollout.py, then python eda/90_q_context.py-style run via eda/108."""
import duckdb, numpy as np, pandas as pd, sys
sys.stdout.reconfigure(encoding="utf-8")
con = duckdb.connect(); con.execute("SET memory_limit='1500MB'; SET threads=2")
S = con.execute("SELECT store, style_color, ny_date::DATE d, sum(qty_ordered) q FROM read_parquet('Dataset.parquet') WHERE NOT cancelled AND class='RETAIL' GROUP BY ALL").fetchdf()
S["d"] = pd.to_datetime(S.d)
sfirst = S.groupby("store").d.min(); ifirst = S.groupby("style_color").d.min()
rows = []
for o in pd.period_range("2024-03", "2026-06", freq="M"):
    E = o.end_time.normalize(); a, b = E - pd.Timedelta(days=27), E - pd.Timedelta(days=55)
    w1 = S[(S.d >= a) & (S.d <= E)]; w0 = S[(S.d >= b) & (S.d < a)]
    open_ = w1.store.nunique(); new90 = int(((sfirst > E - pd.Timedelta(days=90)) & (sfirst <= E)).sum())
    newst = set(sfirst[(sfirst > E - pd.Timedelta(days=180)) & (sfirst <= E)].index)
    g1 = w1.groupby("style_color").agg(n=("store", "nunique"), q=("q", "sum")); g0 = w0.groupby("style_color").store.nunique()
    gn = w1[w1.store.isin(newst)].groupby("style_color").q.sum()
    X = pd.DataFrame({"style_color": ifirst.index})
    X["rr_n28"] = g1.n.reindex(X.style_color).fillna(0).values
    X["rr_cov"] = X.rr_n28 / max(open_, 1)
    X["rr_dn"] = X.rr_n28 - g0.reindex(X.style_color).fillna(0).values
    X["rr_rate"] = np.log1p((g1.q / g1.n / 28).reindex(X.style_color).fillna(0).values)
    fa = (E - ifirst.reindex(X.style_color)).dt.days.values; X["rr_age"] = np.where(fa >= 0, fa, np.nan)
    X["rr_newshare"] = (gn / g1.q).reindex(X.style_color).fillna(0).values
    X["s_open"] = open_; X["s_new90"] = new90; X["origin"] = str(o)
    rows.append(X)
RF = pd.concat(rows, ignore_index=True); RF.to_parquet("data/rollout_feats.parquet", index=False)
print(RF.groupby("origin")[["s_open", "s_new90"]].first().T.to_string())
print(RF[RF.origin == "2026-06"].describe().T.round(3).to_string())
