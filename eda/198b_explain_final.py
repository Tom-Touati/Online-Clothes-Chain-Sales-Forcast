"""D210 (fork; user: "prepare the submission including explainability"): explanation layer for the final system at the June 2026
origin. Two parts.
(1) Tree member SHAP (time-honest refit of the volume-weighted D198 LightGBM median, `w03`, both feature sets averaged, rows weighted
    (4 x level + 1)^0.3, trained on every row whose 4-month outcome is known by June 2026). TreeSHAP splits its predicted deviation
    z_hat = log((F + 1) / (4 x level + 1)) exactly; the level offset log((4 x level + 1) / (R3 + 1)) turns it into the deviation from
    R3. Features map to the D194 categories (data/fk_categories.json, primary map; the dynamic co-purchase neighbours count as
    shopper-cluster seasonality).
(2) Member decomposition of the final forecast (arithmetic mean of members, given as name=path files at the 2026-06 origin, q50):
    each member's log deviation from R3 and its share of the final total.
Also, on the 13 backtest origins: how much of the final system's deviation from R3 the tree member's categories explain (R^2 of the
final log deviation on the 11 category sums, fitted on earlier origins only, scored on the nine recent ones).
Usage: python eda/198b_explain_final.py final=<4-month totals parquet with origin, style_color, final> [member=path ...]
Writes data/fk_explain_final_targets.csv, data/fk_explain_final_shap.parquet and stdout (eda/explain_final_output.txt)."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
KW = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
T = pd.read_parquet("data/fk_explain_table.parquet"); META = json.load(open("data/fk_explain_meta.json")); CAT = json.load(open("data/fk_categories.json"))
DY = ["dy_nb_sea4", "dy_nb_lvl", "dy_nb_mom3", "dy_nb_yoy"]
T = T.merge(pd.read_parquet("data/fk_dyn_feats.parquet")[["origin", "style_color"] + DY], on=["origin", "style_color"], how="left")
for c in META["CAT"]: T[c] = T[c].astype("category")
BASE, FULL = META["BASE"] + DY, META["FULL"] + DY; FEATS = sorted(set(BASE) | set(FULL)); P = dict(META["PARAMS"]); P["n_jobs"] = int(KW.get("jobs", 6))
FM = {f: CAT["feature_map"][f]["primary"] if f in CAT["feature_map"] else "shopper_cluster_seasonality" for f in FEATS}
NAME = {c["id"]: c["name"] for c in CAT["categories"]}; EXP = [c["id"] for c in CAT["categories"] if c["kind"] == "explained"]


def shap_at(o):
    oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()]; te = T[T.origin == o]; w = (4 * tr.level.values + 1) ** 0.3
    S = np.zeros((len(te), len(FEATS))); base = np.zeros(len(te)); zh = np.zeros(len(te))
    for cols in (BASE, FULL):
        m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **P).fit(tr[cols], tr.z, sample_weight=w)
        c = m.predict(te[cols], pred_contrib=True); S[:, [FEATS.index(f) for f in cols]] += c[:, :-1] / 2; base += c[:, -1] / 2; zh += m.predict(te[cols]) / 2
    C = pd.DataFrame(0.0, index=te.index, columns=EXP)
    for j, f in enumerate(FEATS): C[FM[f]] += S[:, j]
    C[CAT["special_terms"]["base"]] += base; C[CAT["special_terms"]["level_offset"]] += np.log((4 * te.level.values + 1) / (te.R3.values + 1))
    out = te[["origin", "style_color", "R3", "level", "ytot", "target"]].copy(); out["tree_F"] = (4 * te.level.values + 1) * np.exp(zh) - 1
    out["tree_dev"] = np.log((out.tree_F + 1) / (out.R3 + 1)); return pd.concat([out, C], axis=1)


# backtest: how much of the final system's deviation do the tree categories explain?
ORIG = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
BT = pd.concat([shap_at(o) for o in ORIG], ignore_index=True)
if "final_bt" in KW:
    Fb = pd.read_parquet(KW["final_bt"]); col = KW.get("final_col", "final")
    BT = BT.merge(Fb[["origin", "style_color", col]].rename(columns={col: "final"}), on=["origin", "style_color"], how="inner")
    BT["dev"] = np.log((BT.final + 1) / (BT.R3 + 1)); from sklearn.linear_model import RidgeCV
    pr = np.full(len(BT), np.nan)
    for o in [x for x in ORIG if x >= "2025-06"]:
        tr = (BT.origin != o).values; te = (BT.origin == o).values   # leave-one-origin-out (an explanation fit, not a forecast)
        pr[te] = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(BT.loc[tr, EXP], BT.dev[tr], sample_weight=np.log1p(BT.R3[tr])).predict(BT.loc[te, EXP])
    m = ~np.isnan(pr); w = np.log1p(BT.R3[m]); r2 = 1 - (w * (BT.dev[m] - pr[m]) ** 2).sum() / (w * (BT.dev[m] - np.average(BT.dev[m], weights=w)) ** 2).sum()
    r2t = 1 - (w * (BT.dev[m] - BT.tree_dev[m]) ** 2).sum() / (w * (BT.dev[m] - np.average(BT.dev[m], weights=w)) ** 2).sum()
    print(f"backtest, nine recent origins: final-system deviation from R3 explained by the tree member's 11 categories (ridge, leave-one-origin-out) R^2 {r2:.3f}; "
          f"by the tree member's own deviation R^2 {r2t:.3f}; direction agreement {np.mean(np.sign(pr[m]) == np.sign(BT.dev[m])):.3f}")
# June 2026
J = shap_at("2026-06"); J.to_parquet("data/fk_explain_final_shap.parquet", index=False)
it = pd.read_parquet("data/items.parquet").set_index("style_color"); J.insert(2, "product", it.product_name.reindex(J.style_color).values)
mem = {k: v for k, v in KW.items() if k not in ("final", "final_bt", "final_col", "jobs")}
for k, p in mem.items():
    q = pd.read_parquet(p).set_index("style_color"); col = "q50" if "q50" in q.columns else q.columns[-1]; J[f"m_{k}"] = q[col].reindex(J.style_color).values
    J[f"dev_{k}"] = np.log((J[f"m_{k}"] + 1) / (J.R3 + 1))
if "final" in KW:
    Fj = pd.read_parquet(KW["final"]).set_index("style_color"); J["final"] = Fj[KW.get("final_col", "final")].reindex(J.style_color).values; J["final_dev"] = np.log((J.final + 1) / (J.R3 + 1))
def drivers(r):
    s = r[EXP].astype(float).sort_values(key=np.abs, ascending=False).head(3); return "; ".join(f"{NAME[k]} {np.expm1(v):+.0%}" for k, v in s.items())
J["top_drivers"] = J.apply(drivers, axis=1)
TGT = set(pd.read_csv("data/targets.csv").style_color.astype(str)); J["target"] = J.style_color.isin(TGT)   # the 100 workbook targets (99 at the origin; S16079_DARK-GRAY is ruled, D034)
TG = J[J.target].sort_values("R3", ascending=False); TG.to_csv("data/fk_explain_final_targets.csv", index=False)
w = TG.R3.values / TG.R3.sum()
print(f"\nJune 2026, {len(TG)} targets in the tree table: R3 {TG.R3.sum():,.0f}, tree member {TG.tree_F.sum():,.0f} ({TG.tree_F.sum() / TG.R3.sum() - 1:+.1%})" + (f", final {TG.final.sum():,.0f} ({TG.final.sum() / TG.R3.sum() - 1:+.1%})" if "final" in TG else ""))
for k in mem: print(f"  member {k:10s} total {TG[f'm_{k}'].sum():,.0f} ({TG[f'm_{k}'].sum() / TG.R3.sum() - 1:+.1%})")
print("\n== tree member: R3-weighted mean contribution by category (log units; % effect)")
for k, v in pd.Series({NAME[c]: (TG[c].values * w).sum() for c in EXP}).sort_values(key=np.abs, ascending=False).items(): print(f"  {k:62s} {v:+.3f} ({np.expm1(v):+.1%})")
print("\n== the ten largest targets"); print(TG.head(10)[["style_color", "product", "R3", "tree_F"] + (["final"] if "final" in TG else []) + ["top_drivers"]].round(0).to_string(index=False))
