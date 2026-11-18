"""Fork 2 (user, 18 Nov: "about the 100 items - create an interface where i can select an item from a list, see its entire history, the time
series of its higher levels (category, product type, embedding, cluster), nearest neighbors, attributes etc. show me static timeseries of
generic stuff. show me rule based. the projection for the 4 missing months till october, run it through the explainability pipelines. try to
convince me why this is the projection and if it needs adjustment in your eyes.")
Builds data/item_explorer.json for eda/item_explorer.html (the page embeds it). Sources:
  history       data/daily_item_chan.parquet (cancelled lines dropped): weekly units, monthly units by channel group, online discount
  levels        data/monthly_item.parquet summed by product type, macro group, cluster (data/clusters_2026-06.parquet, D158 multi-view
                embedding as of June 2026: cart co-occurrence, sales co-movement, fashion descriptors, weather) and the 10 nearest
                neighbours in that embedding (cosine)
  projection    data/final_2026_nx4_cascade.csv (the submitted forecast: R3, seasonal baseline, five members, correction, monthly split,
                P10 / P90, buying, explanation) + P25 / P75 and the recommender from the candidate (final_2026_nx5_cascade.csv)
  rules         agent_eval/2026-06/predictions.csv (the rule agents' monthly forecast)
  drivers       data/fk_2026_explain.parquet (tree SHAP grouped into 11 driver families, log scale against R3)
  checks        the item's own, its type's and its neighbours' Jul-Oct 2025 / (4 x Apr-Jun 2025 monthly mean) ratios, compared with
                the projection / R3; 4-week momentum; member spread; age. A verdict per item (Fork 2's reading, rule-based, stated as such)."""
import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
r = lambda x, k=1: None if x is None or not np.isfinite(x) else round(float(x), k)
C = pd.read_csv("data/final_2026_nx12_cascade.csv")   # D419: nx12 (nx11 numbers; own buying simulation, metrics sheet, D431) is the submission
C_ = None  # was final_2026_nx4_cascade.csv
C = C.rename(columns={"Unnamed: 0": "style_color"})
C5 = pd.read_csv("data/final_2026_nx12_cascade.csv").rename(columns={"Unnamed: 0": "style_color"}).set_index("style_color")
EXP = pd.read_csv("data/fk_expectation_2026.csv").set_index("style_color")
TGT = list(C.style_color)
IT = pd.read_parquet("data/items.parquet").set_index("style_color")
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; MON = list(M.columns)
MAC = pd.read_parquet("data/features.parquet", columns=["style_color", "macro"]).drop_duplicates("style_color").set_index("style_color").macro
FT = pd.read_parquet("data/fk_explain_table.parquet"); FT = FT[FT.origin == FT.origin.max()].set_index("style_color")
CL = pd.read_parquet("data/clusters_2026-06.parquet").set_index("style_color"); XC = [c for c in CL.columns if c.startswith("x")]
PROF = pd.read_csv("data/cluster_profiles_2026-06.csv").set_index("cluster")
EX = pd.read_parquet("data/fk_2026_explain.parquet").set_index("style_color")
RU = pd.read_csv("agent_eval/2026-06/predictions.csv"); RU = RU.pivot(index="style_color", columns="month", values="pred")
D = pd.read_parquet("data/daily_item_chan.parquet", columns=["style_color", "ny_date", "chan", "cancelled", "qty", "disc_w"]); D = D[~D.cancelled]
D["d"] = pd.to_datetime(D.ny_date); D["m"] = D.d.dt.to_period("M").astype(str)
D["grp"] = np.select([D.chan == "edikted.com", D.chan == "Tiktok", D.chan == "RETAIL"], ["web", "tiktok", "stores"], "partners")
ptype = IT.product_type.reindex(M.index).fillna("other"); macro = MAC.reindex(M.index).fillna("other"); clus = CL.cluster.reindex(M.index)
# ---------- generic series (whole business)
G = D.groupby(["m", "grp"]).qty.sum().unstack(fill_value=0).reindex(MON, fill_value=0)
web = D[D.grp == "web"].groupby("m").agg(q=("qty", "sum"), w=("disc_w", "sum")).reindex(MON)
active = (M > 0).sum(0)
nstores = D[D.grp == "stores"].groupby("m").chan.size()  # placeholder size, replaced below by distinct stores where available
GEN = dict(months=MON, total=[int(x) for x in M.sum(0)], web=[int(x) for x in G.get("web", 0)], tiktok=[int(x) for x in G.get("tiktok", 0)],
           stores=[int(x) for x in G.get("stores", 0)], partners=[int(x) for x in G.get("partners", 0)],
           disc=[r(x, 3) for x in (web.w / web.q).values], active=[int(x) for x in active])
try:   # number of own stores open (data/rc_locations.parquet first_m)
    L = pd.read_parquet("data/rc_locations.parquet"); st = L[L["loc"].str.startswith("store:")]
    GEN["n_stores"] = [int((st.first_m.astype(str) <= m).sum()) for m in MON]
except Exception: pass
# ---------- embedding neighbours
X = CL[XC].fillna(0).values; X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9); cix = {s: k for k, s in enumerate(CL.index)}
def ratio(series_m, ok=True):   # Jul-Oct 2025 / (4 x Apr-Jun 2025 monthly mean): last year's seasonal step from the same origin month
    b = np.mean([series_m[m] for m in ("2025-04", "2025-05", "2025-06")]); a = sum(series_m[m] for m in ("2025-07", "2025-08", "2025-09", "2025-10"))
    return a / (4 * b) if b > 0 else np.nan
ty_tot = M.groupby(ptype.values).sum(); ma_tot = M.groupby(macro.values).sum(); cl_tot = M.groupby(clus.values).sum()
W = D[D.style_color.isin(TGT)].groupby(["style_color", pd.Grouper(key="d", freq="W-SUN")]).qty.sum()
CHM = D[D.style_color.isin(TGT)].groupby(["style_color", "m", "grp"]).qty.sum().unstack(fill_value=0)
DSC = D[D.style_color.isin(TGT) & (D.grp == "web")].groupby(["style_color", "m"]).agg(q=("qty", "sum"), w=("disc_w", "sum"))
DRV = ["baseline_drift", "short_term_momentum", "item_lifecycle", "upcoming_events", "past_event_hangover", "seasonal_weather", "regional_climate",
       "last_year_seasonality", "shopper_cluster_seasonality", "product_style", "item_scale_maturity"]
DRVN = {"baseline_drift": "Portfolio baseline (typical drift below recency)", "short_term_momentum": "Short-term momentum (recent weeks)", "item_lifecycle": "Item life cycle",
        "upcoming_events": "Upcoming events and occasions", "past_event_hangover": "Past event hangover", "seasonal_weather": "Seasonal weather", "regional_climate": "Regional climate",
        "last_year_seasonality": "Last year's seasonality", "shopper_cluster_seasonality": "Shopper-cluster seasonality (co-purchase neighbours)", "product_style": "Product type and style",
        "item_scale_maturity": "Item scale and maturity"}
ATTR = ["a_fabric_feel", "a_length", "a_occasion_tier", "a_print_pattern", "a_silhouette", "a_sleeve", "a_weight"]
SCORES = ["summer", "winter", "warmth", "party", "statement", "staple", "formality", "layering", "versatility", "festival", "halloween", "back_to_school", "fit_tightness", "coverage"]
OUT = []
for s in TGT:
    c = C.set_index("style_color").loc[s]; it = IT.loc[s] if s in IT.index else None; mser = M.loc[s] if s in M.index else pd.Series(0, index=MON)
    ty, mc, cl = ptype.get(s, "other"), macro.get(s, "other"), clus.get(s, np.nan)
    # neighbours
    nb = []
    if s in cix:
        sim = X @ X[cix[s]]; order = [k for k in np.argsort(-sim) if CL.index[k] != s][:10]
        for k in order:
            n = CL.index[k]; nm = M.loc[n] if n in M.index else None
            nb.append(dict(id=n, name=str(IT.product_name.get(n, n)), type=str(IT.product_type.get(n, "")), sim=r(sim[k], 3), last12=int(nm[MON[-12:]].sum()) if nm is not None else 0,
                           ratio=r(ratio(nm), 2) if nm is not None else None, spark=[int(x) for x in nm[MON[-18:]]] if nm is not None else []))
    nb_sum = M.reindex([n["id"] for n in nb]).fillna(0).sum(0) if nb else None
    # weekly + channels + discount
    w = W.loc[s] if s in W.index.get_level_values(0) else pd.Series(dtype=float); w = w[w.index <= "2026-06-30"]
    wk = dict(start=str(w.index.min().date()) if len(w) else None, v=[int(x) for x in w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="W-SUN"), fill_value=0)] if len(w) else [])
    ch = CHM.loc[s].reindex(MON, fill_value=0) if s in CHM.index.get_level_values(0) else pd.DataFrame(0, index=MON, columns=["web"])
    ds = DSC.loc[s].reindex(MON) if s in DSC.index.get_level_values(0) else None
    # checks
    R3 = float(c.R3); F = float(c.forecast); own_r = ratio(mser) if mser[["2025-04", "2025-05", "2025-06"]].min() > 0 else np.nan
    ty_r = ratio(ty_tot.loc[ty]) if ty in ty_tot.index else np.nan; nb_r = np.nanmedian([n["ratio"] for n in nb if n["ratio"] is not None]) if nb else np.nan
    mom = (w.iloc[-4:].sum() / max(w.iloc[-12:-4].sum() / 2, 1)) if len(w) >= 12 else np.nan
    age = float(FT.age_m.get(s, np.nan)) if "age_m" in FT.columns else np.nan
    bench = [x for x in (own_r, ty_r, nb_r) if np.isfinite(x)]; b = float(np.median(bench)) if bench else np.nan; imp = F / R3 if R3 > 0 else np.nan
    ex = EX.loc[s] if s in EX.index else None
    OUT.append(dict(
        id=s, name=str(EXP.name.get(s, IT.product_name.get(s, s))), type=ty, macro=mc, cluster=None if not np.isfinite(cl) else int(cl),
        meta=dict(collection=str(it.product_collection) if it is not None else "", color=str(it.color) if it is not None else "", occasion=str(it.occasion) if it is not None else "",
                  published=str(it.published_date)[:10] if it is not None else "", first_sale=str(it.first_sale)[:10] if it is not None else "", sizes=int(it.n_sizes) if it is not None and pd.notna(it.n_sizes) else None,
                  skus=int(it.n_skus) if it is not None and pd.notna(it.n_skus) else None, age_m=r(age, 1), lifetime_units=int(mser.sum())),
        attrs={a[2:]: str(FT[a].get(s)) for a in ATTR if a in FT.columns and s in FT.index},
        scores={k: r(FT[k].get(s), 2) for k in SCORES if k in FT.columns and s in FT.index},
        monthly=[int(x) for x in mser], ch={g: [int(x) for x in ch[g]] if g in ch.columns else [0] * len(MON) for g in ("web", "tiktok", "stores", "partners")},
        disc=[r(x, 3) for x in (ds.w / ds.q).values] if ds is not None else [], weekly=wk,
        levels=dict(type=[int(x) for x in ty_tot.loc[ty]] if ty in ty_tot.index else [], macro=[int(x) for x in ma_tot.loc[mc]] if mc in ma_tot.index else [],
                    cluster=[int(x) for x in cl_tot.loc[cl]] if np.isfinite(cl) and cl in cl_tot.index else [], neighbours=[int(x) for x in nb_sum] if nb_sum is not None else []),
        cluster_info=dict(top_types=str(PROF.top_types.get(cl, "")), aesthetics=str(PROF.aesthetics.get(cl, "")), items=int(PROF["items"].get(cl, 0)) if np.isfinite(cl) else 0, peak=int(PROF.peak_month.get(cl, 0)) if np.isfinite(cl) else 0) if np.isfinite(cl) and cl in PROF.index else {},
        nb=nb,
        proj=dict(months=["2026-07", "2026-08", "2026-09", "2026-10"], v=[int(c.jul), int(c.aug), int(c.sep), int(c.oct)], total=int(c.final), R3=r(R3, 0), season=r(float(c.season), 0),
                  members={k: r(float(c[k]), 0) for k in ("rules", "qnx", "nn", "pf2c", "pf2d32", "rc")}, avg5=r(float(c.avg5), 0), corr=r(float(c.correction), 3), forecast=r(F, 0),
                  P10=r(float(c.P10), 0), P90=r(float(c.P90), 0), P25=r(float(C5.P25.get(s, np.nan)), 0), P75=r(float(C5.P75.get(s, np.nan)), 0), rc=r(float(C5.rc.get(s, np.nan)), 0),
                  buying=r(float(c.buying), 0), confidence=int(c.confidence), spread=r(float(c.member_spread), 2), explanation=str(c.explanation),
                  rules_m=[r(float(RU.loc[s, m]), 0) if s in RU.index and m in RU.columns else None for m in ["2026-07", "2026-08", "2026-09", "2026-10"]]),
        drivers=dict(tree_F=r(float(ex.tree_F), 0), tree_dev=r(float(ex.tree_dev), 3), parts=[dict(k=DRVN[k], v=r(float(ex[k]), 3)) for k in DRV], why=str(ex.why_text)) if ex is not None else None,
        checks=dict(implied=r(imp, 2), own=r(own_r, 2), type=r(ty_r, 2), nb=r(nb_r, 2), bench=r(b, 2), mom4=r(mom, 2), age=r(age, 1))))
json.dump(dict(items=OUT, generic=GEN, months=MON), open("data/item_explorer.json", "w", encoding="utf-8"), ensure_ascii=False)
print(len(OUT), "items")
print(pd.DataFrame([dict(id=o["id"], F=o["proj"]["forecast"], **o["checks"]) for o in OUT]).describe().round(2).to_string())
