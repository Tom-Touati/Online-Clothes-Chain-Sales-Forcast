"""D279 (fork): does an item's appeal carry from one own store to another, and from the web to stores? Same-month units per item
(data/fk_axes_month.parquet), items sold in both places; Spearman by month, averaged. Also how early the web leads the stores.
Adds J["launch"]["carry"] to data/fk_err_analysis.json."""
import json, itertools, numpy as np, pandas as pd
from scipy.stats import spearmanr
A = pd.read_parquet("data/fk_axes_month.parquet"); NON = {"u_cancel", "u_edikted_com", "u_tiktok", "u_flip", "u_pacsun", "u_nordstrom", "u_macys", "u_tillys", "u_bloomingdales", "u_urban", "u_hottopic", "u_intl"}
ST = [c for c in A.columns if c.startswith("u_") and c not in NON]; PA = ["u_pacsun", "u_nordstrom", "u_macys", "u_tillys", "u_bloomingdales", "u_urban"]; A = A[A.m >= "2025-01"]
def pair(a, b, minn=60):
    out = []
    for m, g in A.groupby("m"):
        z = g[(g[a] > 0) & (g[b] > 0)]
        if len(z) >= minn: out.append(spearmanr(z[a], z[b])[0])
    return np.mean(out) if out else np.nan
ss = [pair(a, b) for a, b in itertools.combinations(ST, 2)]; ws = [pair("u_edikted_com", s) for s in ST]; wp = [pair("u_edikted_com", p) for p in PA if p in A]; wt = pair("u_edikted_com", "u_tiktok")
A["u_st"] = A[ST].sum(1); A["n_st"] = (A[ST] > 0).sum(1); A["rate"] = A.u_st / A.n_st.where(A.n_st > 0)
AR = pd.read_parquet("data/fk_store_arrivals.parquet"); w = AR[AR.kind == "web"].set_index("item").a; s1 = AR[AR.kind == "store"].groupby("item").a.min(); lead = ((s1 - w.reindex(s1.index)).dt.days).dropna(); lead = lead[w.reindex(lead.index) >= pd.Timestamp("2024-02-01")]
r = lambda x, k=3: None if not np.isfinite(x) else round(float(x), k)
C = dict(store_store=r(np.nanmean(ss)), store_store_min=r(np.nanmin(ss)), store_store_max=r(np.nanmax(ss)), n_pairs=int(np.isfinite(ss).sum()), web_store=r(np.nanmean(ws)), web_partner=r(np.nanmean(wp)), web_tiktok=r(wt),
         web_vs_rate_per_store=r(np.mean([spearmanr(g.u_edikted_com[g.n_st > 0], g.rate[g.n_st > 0])[0] for _, g in A.groupby("m")])), lead_median=r(float(lead.median()), 0), web_first=r(float((lead > 0).mean())), same_week=r(float((lead.abs() <= 7).mean())),
         lead_14=r(float((lead >= 14).mean())), n_items=int(len(lead)))
print(C); J = json.load(open("data/fk_err_analysis.json")); J["launch"]["carry"] = C; json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0)
