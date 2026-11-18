"""D158: profiles of the multi-view clusters at the last origin (June 2026; data/clusters_2026-06.parquet from eda/163).
Per cluster: size, units over the last 12 months, top product types and aesthetics, mean LLM scores, temperature preference
(beta_T: log-units change per deg C across states), warm-state skew, 2025 monthly profile (units by month / the cluster's 2025 mean),
Halloween lift (Oct / Sep 2025), Black Friday lift (Nov / Oct 2025), and how much of each cluster's month-to-month variation in
2025 a national temperature curve explains (R^2 of log monthly units on the sales-weighted national temperature, 2024-07..2026-06).
Writes data/cluster_profiles_2026-06.csv and eda/figs/cluster_profiles.png."""
import sys, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
C = pd.read_parquet("data/clusters_2026-06.parquet")[["style_color", "cluster"]]; C = C[C.cluster >= 0]
it = pd.read_parquet("data/items.parquet").set_index("style_color"); A = pd.read_csv("item_attributes.csv").set_index("style_color")
L = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("data/llm/scores_part*.csv"))]).drop_duplicates("style_color").set_index("style_color")
WF = pd.read_parquet("data/weather_event_feats.parquet"); WF = WF[WF.origin == "2026-06"].set_index("style_color")
RF = pd.read_parquet("data/region_feats.parquet"); RF = RF[RF.origin == "2026-06"].set_index("style_color")
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
W = pd.read_parquet("data/weather_state_month.parquet"); S = pd.read_parquet("data/state_item_month.parquet")
sw = S[S.m.between("2025-01", "2025-12")].groupby("state").units.sum(); sw = sw / sw.sum()
natT = W[W.state.isin(sw.index)].assign(w=lambda d: d.state.map(sw)).groupby("m").apply(lambda d: (d.tmean * d.w).sum() / d.w.sum())
m25 = [f"2025-{k:02d}" for k in range(1, 13)]; mT = [m for m in M.columns if "2024-07" <= m <= "2026-06"]
rows = []
for k, g in C.groupby("cluster"):
    ids = g.style_color.values; u = M.reindex(ids).fillna(0)
    prof = u[m25].sum(); prof = prof / max(prof.mean(), 1e-9)
    y = np.log1p(u[mT].sum().values); x = natT.reindex(mT).values; ok = np.isfinite(x)
    r2 = np.corrcoef(x[ok], y[ok])[0, 1] ** 2 if ok.sum() > 6 and y[ok].std() > 0 else np.nan
    sl = np.polyfit(x[ok], y[ok], 1)[0] if ok.sum() > 6 else np.nan
    tags = A.reindex(ids).aesthetics.dropna().str.split(r"\s*\|\s*").explode().value_counts().head(3).index.tolist()
    rows.append({"cluster": k, "items": len(ids), "units_12m": int(u[[m for m in M.columns if "2025-07" <= m <= "2026-06"]].sum().sum()),
                 "top_types": ", ".join(it.reindex(ids).product_type.value_counts().head(3).index), "aesthetics": ", ".join(tags),
                 "beta_T": WF.reindex(ids).beta_T.mean(), "warm_skew": RF.reindex(ids).clim_warm_rel.mean(),
                 "temp_R2": r2, "temp_slope": sl, "halloween_lift": prof["2025-10"] / max(prof["2025-09"], 1e-9), "bf_lift": prof["2025-11"] / max(prof["2025-10"], 1e-9),
                 "peak_month": prof.idxmax()[-2:], **{f"llm_{c}": L.reindex(ids)[c].mean() for c in ["warmth", "summer", "winter", "party", "halloween", "festival", "back_to_school"]},
                 **{f"p{m[-2:]}": prof[m] for m in m25}})
P = pd.DataFrame(rows).sort_values("units_12m", ascending=False); P.to_csv("data/cluster_profiles_2026-06.csv", index=False)
print(P[["cluster", "items", "units_12m", "top_types", "aesthetics", "peak_month", "beta_T", "warm_skew", "temp_R2", "temp_slope", "halloween_lift", "bf_lift"]].round(3).to_string(index=False))
print("\nunits-weighted mean R^2 of log monthly units on national temperature:", round(np.average(P.temp_R2.fillna(0), weights=P.units_12m), 3))
print("correlation across clusters, beta_T (cross-state) vs temp_slope (over time):", round(P[["beta_T", "temp_slope"]].corr().iloc[0, 1], 3))
fig, ax = plt.subplots(1, 2, figsize=(14, 0.32 * len(P) + 1.8), gridspec_kw={"width_ratios": [4, 1]})
H = np.log2(P[[f"p{m[-2:]}" for m in m25]].clip(lower=0.05).values)
im = ax[0].imshow(H, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
ax[0].set_yticks(range(len(P))); ax[0].set_yticklabels([f"{r.cluster:2d} {r.top_types[:34]} ({r.items})" for r in P.itertuples()], fontsize=7)
ax[0].set_xticks(range(12)); ax[0].set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]); ax[0].set_title("2025 monthly units / cluster mean (log2)")
plt.colorbar(im, ax=ax[0], fraction=0.02)
ax[1].barh(range(len(P)), P.beta_T.values * 100, color=np.where(P.beta_T.values > 0, "#c0504d", "#4f81bd")); ax[1].invert_yaxis(); ax[1].set_yticks([])
ax[1].set_title("temperature preference\n(% units per deg C, across states)", fontsize=9); ax[1].axvline(0, color="grey", lw=0.5)
plt.tight_layout(); plt.savefig("eda/figs/cluster_profiles.png", dpi=130); print("saved figure")
