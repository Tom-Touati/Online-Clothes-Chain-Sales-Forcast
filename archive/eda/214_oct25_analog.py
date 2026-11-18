"""D395 (Fork 2 handed-off session; user: "add a specific evaluation metric for the same months we dont have data on and are evaluated on
(october 26 etc). perform this specific back test on the same months in october 25, with and without the known futures we add. output all
metrics for this specific back test as is is the most important."). The 2026 deliverable forecasts July-October 2026 from data to the end of
June 2026; its analog is the 2025-06 origin (July-October 2025). Every model, member and system is scored at that origin only, on all 768
items and on the 47 of the 100 2026 targets that existed then. Known-future variants (realised plans, an upper bound of what a plan gives)
are scored next to their no-plan versions, alone and in the system (five members, Huber tree, + the variant at full weight, x the submitted
correction multiplier). Writes eda/oct25_analog_output.txt (stdout) and data/oct25_analog_metrics.csv.
Usage: python eda/214_oct25_analog.py [extra=label:file:col ...]"""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
O = "2025-06"; D = pd.read_parquet("data/fk_err_master.parquet"); D = D[D.origin == O].reset_index(drop=True); K = D[["origin", "style_color"]]
T = pd.read_parquet("data/fk_explain_table.parquet", columns=["origin", "style_color", "own_sea4", "level"]); T["origin"] = T.origin.astype(str)
D = D.drop(columns=["level"]).merge(T, on=["origin", "style_color"], how="left")
TG = set(pd.read_csv("data/final_2026_nx4_cascade.csv")["Unnamed: 0"]); tgm = D.style_color.isin(TG).values
y = D.y.values; MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; top = D[MEM].max(1).values; corr = (D.F / D.F0).values
def col(fn, c, cap=True):
    v = K.merge(pd.read_parquet(fn).assign(origin=lambda d: d.origin.astype(str))[["origin", "style_color", c]].drop_duplicates(["origin", "style_color"]), how="left")[c].astype(float).values
    return np.minimum(np.where(np.isnan(v), D.R3, v), 3 * top) if cap else v
hub = col("data/dec_base_ecnsnnhub.parquet", "qdyn", cap=False); X5 = D[MEM].values.copy(); X5[:, 1] = hub
L4 = 4 * D.level.values + 1; sea = np.where(np.isfinite(D.own_sea4), L4 * np.exp(np.nan_to_num(D.own_sea4)) - 1, D.R3)
F = {"R3 (recent 3 months x 4)": D.R3.values, "seasonal naive (level x last year's window ratio; R3 if none)": sea,
     "rules": D.rules.values, "tree, quantile (submitted member)": D.qdyn.values, "tree, Huber (candidate member)": hub, "neural net": D.nn.values,
     "TabPFN conformal": D.pf2c.values, "TabPFN 32 estimators": D.pf2d32.values,
     "recommender DeepFM (honest)": col("data/rc_recsys_preds.parquet", "rc_deepfm"), "recommender tree twin (honest)": col("data/rc_recsys_preds.parquet", "rc_gbm"),
     "SYSTEM submitted (nx4)": D.F.values, "SYSTEM five members, Huber tree": X5.mean(1) * corr,
     "SYSTEM candidate D358 (+ DeepFM x0.25)": (X5.sum(1) + 0.25 * col("data/rc_recsys_preds.parquet", "rc_deepfm")) / 5.25 * corr}
KF = [("realised plan: locations + discounts (recommender twin)", "data/rc_recsys_preds_e10.parquet", "f_plan_gbm"), ("realised stores (twin)", "data/rc_recsys_preds_if_storeplan_ps0.parquet", "f_storeplan_gbm"),
      ("realised partners (twin)", "data/rc_recsys_preds_if_partnerplan_ps0.parquet", "f_partnerplan_gbm"), ("realised discounts (twin)", "data/rc_recsys_preds_x_disc.parquet", "f_discplan_gbm"),
      ("business plan: stores + partners + discounts + launches (twin)", "data/rc_recsys_preds_x_biz.parquet", "f_biz_gbm"),
      ("one model, plan given (aug)", "data/rc_recsys_preds_x_aug.parquet", "f_aug_plan_gbm"), ("one model, plan missing (aug)", "data/rc_recsys_preds_x_aug.parquet", "f_aug_none_gbm"),
      ("plan imputed from the past (bizimpaug)", "data/rc_recsys_preds_x_imp.parquet", "f_bizimpaug_gbm"),
      ("fork plan tree: plan given", "data/fk_plan_tree_preds.parquet", "plan"), ("fork plan tree: no plan", "data/fk_plan_tree_preds.parquet", "noplan")]
for a in sys.argv[1:]:
    if a.startswith("extra="): lab, rest = a[6:].split(":", 1); fn, c = rest.rsplit(":", 1); KF.append((lab, fn, c))
for lab, fn, c in KF:
    try: f = col(fn, c)
    except Exception as e: print("skip", lab, e); continue
    kf = not lab.startswith("TREE "); F[("KNOWN FUTURE " if kf else "") + lab] = f; F[("KNOWN FUTURE SYSTEM + " + lab) if kf else "SYSTEM five, Huber swapped for " + lab[5:]] = (X5.sum(1) + f) / 6 * corr if kf else (X5.sum(1) - X5[:, 1] + f) / 5 * corr
ref = F["SYSTEM submitted (nx4)"]
def met(f, m):
    yy, ff, rr = y[m], np.clip(f[m], 0, None), D.R3.values[m]; e = np.abs(ff - yy); e3 = np.abs(rr - yy)
    ok = (yy > 0) & (e3 > 0); P = D.ptype.astype(str).values[m]
    agg = pd.DataFrame({"p": P, "f": ff, "y": yy, "r": rr}).groupby("p").sum(); big = yy >= np.quantile(yy, 0.9)
    return dict(n=int(m.sum()), wape_R3=e.sum() / e3.sum(), wape_units=e.sum() / yy.sum(), bias_pct=100 * (ff.sum() / yy.sum() - 1), total_err_pct=100 * abs(ff.sum() - yy.sum()) / yy.sum(),
                hit20=float((e <= 0.2 * yy).mean()), hit40=float((e <= 0.4 * yy).mean()), gmrae=float(np.exp(np.mean(np.log((e[ok] + 1) / (e3[ok] + 1))))), mdape=float(np.median(e[yy > 0] / yy[yy > 0])),
                type_wape_R3=float(np.abs(agg.f - agg.y).sum() / np.abs(agg.r - agg.y).sum()), top10_bias_pct=100 * (ff[big].sum() / yy[big].sum() - 1), win_vs_sub=float((e < np.abs(ref[m] - yy)).mean()))
rows = []
for lab, f in F.items():
    for sl, m in [("all 768", np.ones(len(D), bool)), ("47 of the 2026 targets", tgm)]: rows.append(dict(model=lab, slice=sl, **met(f, m)))
R = pd.DataFrame(rows); R.to_csv("data/oct25_analog_metrics.csv", index=False)
cov = lambda lo, hi, m: float(((y[m] >= lo[m]) & (y[m] <= hi[m])).mean())
print(f"Origin {O}: forecasts of July-October 2025 from data to the end of June 2025 (the 2026 deliverable's analog).")
print("Columns: WAPE/R3, WAPE in units, bias %, |total error| %, share within +-20% / +-40%, GMRAE vs R3, median APE, product-type WAPE/R3, top-decile bias %, share of items better than the submitted system.\n")
for sl in ["all 768", "47 of the 2026 targets"]:
    print("==", sl); print(R[R.slice == sl].drop(columns="slice").set_index("model").round(3).to_string()); print()
print("System P10-P90 coverage (stored p10 / p90):", round(cov(D.p10.values, D.p90.values, np.ones(len(D), bool)), 3), "all |", round(cov(D.p10.values, D.p90.values, tgm), 3), "targets")
