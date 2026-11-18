"""D320 (fork; proposal #12a of the error study, assigned in D314: "Commit fabric and capacity on look or type totals, where +/-20% holds most
of the time, and split to items late"). July-October 2026 totals of the submitted system for every product group, product type and look
(all 1,423 items at the June 2026 origin, data/fk_err_master_2026.parquet, column F), each with an 80% range and the share of such totals that
landed within +/-20% in the backtest.
  ranges   from the backtest: at each recent origin the group totals' log error log((y + 1) / (F + 1)); its 10th and 90th percentiles by level
           and by size third of the forecast total (bigger groups miss by less), applied to the 2026 totals
  hit rate the backtest share within +/-20% for the same level and size third
Groups with fewer than 3 items are folded into "other" within their level. Each look is described by its three largest product types and its largest item. The totals are the models' (the World Cup cap and the out-of-stock rule apply to the workbook only). Writes data/fk_totals_2026.csv; adds J["totals26"];
stdout -> eda/totals_2026_output.txt."""
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
r = lambda x, k=3: None if x is None or not np.isfinite(x) else round(float(x), k)
D = pd.read_parquet("data/fk_err_master.parquet"); D = D[D.origin >= "2025-06"]; T = pd.read_parquet("data/fk_err_master_2026.parquet")
W = pd.read_excel("Edikted_Forecasting_Exercise_filled.xlsx", sheet_name="Forecast"); TG = set(W.iloc[:, 1].dropna().astype(str))
LV = {"product group": "macro", "product type": "ptype", "look": "look"}; cal, rows = [], []
for lab, key in LV.items():
    g = D.groupby(["origin", key]).agg(F=("F", "sum"), y=("y", "sum"), n=("F", "size")).reset_index(); g = g[g.n >= 3]; g["e"] = np.log((g.y + 1) / (g.F + 1)); g["hit"] = (g.F - g.y).abs() <= 0.2 * g.y
    g["third"] = g.groupby("origin").F.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False)) if lab != "product group" else 0
    for t, z in g.groupby("third"):
        cal.append(dict(level=lab, third=int(t), lo=float(z.e.quantile(0.1)), hi=float(z.e.quantile(0.9)), hit=float(z.hit.mean()), n=int(len(z)), f_min=float(z.F.min()), f_max=float(z.F.max()), june_hit=r(float(z[z.origin == "2025-06"].hit.mean())) if (z.origin == "2025-06").any() else None))
    a = T.groupby(key).agg(F=("F", "sum"), n=("F", "size"), R3=("R3", "sum")); tg = T[T.style_color.isin(TG)].groupby(key).F.sum(); a["targets_F"] = tg.reindex(a.index).fillna(0)
    small = a.n < 3
    if small.any(): o = a[small].sum(); a = a[~small]; a.loc["other (groups under 3 items)"] = o
    edges = np.quantile(g.F, [1 / 3, 2 / 3]) if lab != "product group" else None   # size thirds from the backtest's own distribution of group totals
    for k, v in a.iterrows():
        t = 0 if lab == "product group" else int(np.searchsorted(edges, v.F)); c = next(x for x in cal if x["level"] == lab and x["third"] == t)
        rows.append(dict(level=lab, group=str(k), items=int(v.n), forecast=round(float(v.F)), R3=round(float(v.R3)), targets_forecast=round(float(v.targets_F)), P10=round(float((v.F + 1) * np.exp(c["lo"]) - 1)), P90=round(float((v.F + 1) * np.exp(c["hi"]) - 1)), backtest_hit20=r(c["hit"]), size_third=t + 1))
NMS = pd.read_parquet("data/items.parquet").set_index("style_color").product_name
def describe(lab, k):
    z = T[T[LV[lab]] == k]
    if lab != "look" or not len(z): return ""
    sh = z.groupby("ptype").F.sum().sort_values(ascending=False); sh = sh / sh.sum(); top = z.sort_values("F", ascending=False).style_color.iloc[0]
    return "; ".join(f"{t} {100 * v:.0f}%" for t, v in sh.head(3).items()) + f" | largest: {NMS.get(top, top)}"
for d in rows: d["contents"] = describe(d["level"], d["group"])
C = pd.DataFrame(cal); print(C.round(3).to_string(index=False)); O = pd.DataFrame(rows).sort_values(["level", "forecast"], ascending=[True, False]); O.to_csv("data/fk_totals_2026.csv", index=False)
for lab in LV: print(f"\n{lab}:"); print(O[O.level == lab].head(12).to_string(index=False))
print("\nall items:", round(float(T.F.sum())), "| targets:", round(float(T[T.style_color.isin(TG)].F.sum())))
J = json.load(open("data/fk_err_analysis.json")); J["totals26"] = dict(calibration=[{k: (r(v) if isinstance(v, float) else v) for k, v in d.items()} for d in cal], counts={lab: int((O.level == lab).sum()) for lab in LV}, all_items=round(float(T.F.sum())),
                                                       top_types=O[O.level == "product type"].head(10).to_dict("records"), groups=O[O.level == "product group"].to_dict("records"))
json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
