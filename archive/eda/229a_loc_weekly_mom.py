"""Fork "fashion" (D403; user, 18 Nov: "run a version with recent last month weekly momentum (for each momentum feature the 25th percentile
and 75th percentile of momentum across all regions)"). Location-level weekly momentum features, known at each month-end.
Locations as the recommender's panel (eda/201_rc_panel.py, cancelled lines dropped): web:<census region> (Northeast, Midwest, South, West,
other), tiktok, store:<name>, partner:<name> (31 locations). Daily units per item x location from Dataset.parquet via DuckDB.
At each month-end o (all month-ends Jan 2024 .. Jun 2026), per item x location, from the 8 weeks (56 days) ending on the last day of o:
  mom4  log((units in the last 4 weeks + 1) / (units in the 4 weeks before + 1))
  mom2  log((last 2 weeks + 1) / (the 2 weeks before + 1))
  mom1  log((last week x 4 + 1) / (last 4 weeks + 1))            (the latest week against the month's weekly pace)
A location counts for an item if it sold >= 5 units in those 8 weeks. Per item, the 25th and 75th percentile of each measure across its
counted locations: lm_<m>_p25 / lm_<m>_p75 over all locations, and lr_<m>_p25 / lr_<m>_p75 over the web regions only, plus the number of
counted locations (lm_n, lr_n). Items with no counted location get NaN. Writes data/loc_mom_feats.parquet."""
import sys, numpy as np, pandas as pd, duckdb
sys.stdout.reconfigure(encoding="utf-8")
REG = {"Northeast": "CT ME MA NH RI VT NJ NY PA", "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD", "South": "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX PR",
       "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA"}
src = open("eda/201_rc_panel.py", encoding="utf-8").read(); AB = eval(src.split("AB = ", 1)[1].split("\nS2R", 1)[0])   # the same state-name table as the panel
S2R = {s: r for r, ss in REG.items() for s in ss.split()}
def region(state, country):
    s = str(state).strip()
    if str(country).strip() not in ("United States", "US", "USA", "nan", "None", ""): return "other"
    ab = s.upper() if len(s) == 2 else AB.get(s.lower())
    return S2R.get(ab, "other")
con = duckdb.connect(); con.execute("SET threads TO 3; SET memory_limit = '1.5GB'")
D = con.execute("""SELECT style_color, ny_date, o_type, "class" AS cls, CASE WHEN o_type = 'edikted.com' THEN coalesce(state, '') ELSE '' END AS state,
                          CASE WHEN o_type = 'edikted.com' THEN coalesce(country, '') ELSE '' END AS country, sum(qty_ordered) AS q
                   FROM 'Dataset.parquet' WHERE NOT cancelled GROUP BY ALL""").df()
print("daily groups", len(D), flush=True)
sc = D[["state", "country"]].drop_duplicates(); sc["reg"] = [region(s, c) for s, c in zip(sc.state, sc.country)]; D = D.merge(sc, on=["state", "country"], how="left")
D["loc"] = np.where(D.o_type == "edikted.com", "web:" + D.reg, np.where(D.o_type == "Tiktok", "tiktok", np.where(D.cls == "RETAIL", "store:" + D.o_type.astype(str), "partner:" + D.o_type.astype(str))))
D = D.groupby(["style_color", "loc", "ny_date"], observed=True).q.sum().reset_index(); D["ny_date"] = pd.to_datetime(D.ny_date)
print("item x location x day rows", len(D), "| locations", D["loc"].nunique(), "| days", D.ny_date.min().date(), D.ny_date.max().date(), flush=True)
out = []
for o in pd.period_range("2024-01", "2026-06", freq="M"):
    end = o.end_time.normalize(); W = D[(D.ny_date > end - pd.Timedelta(days=56)) & (D.ny_date <= end)].copy()
    if W.empty: continue
    W["k"] = ((end - W.ny_date).dt.days // 7).astype(int)   # 0 = the last 7 days ... 7 = days 50-56 back
    P = W.pivot_table(index=["style_color", "loc"], columns="k", values="q", aggfunc="sum").reindex(columns=range(8)).fillna(0)
    P = P[P.sum(axis=1) >= 5]; w = P.values
    m = pd.DataFrame({"mom4": np.log((w[:, 0:4].sum(1) + 1) / (w[:, 4:8].sum(1) + 1)), "mom2": np.log((w[:, 0:2].sum(1) + 1) / (w[:, 2:4].sum(1) + 1)),
                      "mom1": np.log((w[:, 0] * 4 + 1) / (w[:, 0:4].sum(1) + 1))}, index=P.index).reset_index()
    for tag, sel in [("lm", np.ones(len(m), bool)), ("lr", m["loc"].str.startswith("web:").values)]:
        g = m[sel].groupby("style_color")
        f = pd.concat([g[c].quantile(q).rename(f"{tag}_{c}_p{int(q * 100)}") for c in ("mom4", "mom2", "mom1") for q in (0.25, 0.75)] + [g.size().rename(f"{tag}_n")], axis=1)
        out.append(f.assign(origin=str(o)).reset_index())
F = pd.concat(out).groupby(["origin", "style_color"]).first().reset_index()
F.to_parquet("data/loc_mom_feats.parquet", index=False)
print("rows", len(F), "| origins", F.origin.nunique(), "| coverage (all locations / web regions) at 2025-06:", round(F[F.origin == "2025-06"].lm_mom4_p25.notna().mean(), 3), round(F[F.origin == "2025-06"].lr_mom4_p25.notna().mean(), 3))
print(F[F.origin == "2026-06"].describe().round(3).to_string())
