"""D266 (user, 18 Nov: "add the scores of how accurate our confidence intervals are; perhaps a confusion matrix").
The assignment's scale: 1 = the 4-month total lands within +/-20% of actual, 2 = within +/-50%, 3 = off by more (|F - y| / y; y = 0 counts
as within only if F = 0). Backtest item-origins (data/dec_base*.parquet), forecast = the current best system E:ec_ns_x_geo (D259) and,
for reference, p2c32a.
CONFIDENCE RULES
  rule_wb   the workbook rule (eda/179f / 181f): width of the conformal TabPFN member's P25-P75 around its median, w = (q75 - q25) / (2 q50):
            <= 0.2 -> 1, <= 0.5 -> 2, else 3; one level worse if the five members differ by more than 2x
  rule_ml   time-honest learned rule: at each origin, a LightGBM 3-class model of the realised band trained on earlier origins (o' + 4 <= o)
            from w, member spread, the forecast's ratio to R3, level, age, volume rank, discount, trend, the D247 breakout score; the
            predicted level is the most likely band (rule_ml) or the band whose cumulative probability first passes 50% (rule_ml_med)
SCORES: 3 x 3 confusion matrix (counts and row %), per-level hit rate ('labelled 1, landed within 20%'), accuracy, quadratic-weighted
kappa, unit-weighted versions.
INTERVALS for the final forecast: (a) the conformal TabPFN member's relative quantiles q_tau / q50 put on the final forecast;
(b) time-honest conformal: empirical quantiles of log((y + 1) / (F + 1)) on earlier origins, global, and per rule_wb level.
Coverage of P10-P90 (nominal 80%) and P25-P75 (50%), overall and per level; median width ratio P90 / P10.
Writes eda/confidence_output.txt (stdout) and data/calibration.json (for the dashboard)."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb, warnings
from sklearn.metrics import cohen_kappa_score
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 200)
import os; SRC = os.environ.get("SRC", "ec_ns"); OUTS = "" if SRC == "ec_ns" else "_" + SRC   # D273: SRC=ecnsnn scores the system with the new NN
B = pd.read_parquet(f"data/dec_base_{SRC}.parquet")                       # members with the new tree member (qdyn column = ec_ns)
B = B.merge(pd.read_parquet(f"data/geo_{SRC}_x_preds.parquet")[["origin", "style_color", "geo"]], on=["origin", "style_color"])
B = B.merge(pd.read_parquet("data/dec_base.parquet")[["origin", "style_color", "F"]].rename(columns={"F": "p2c32a"}), on=["origin", "style_color"])
Q = pd.read_parquet("data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet").drop_duplicates(["origin", "style_color"])[["origin", "style_color", "q10", "q25", "q50", "q75", "q90"]]
B = B.merge(Q, on=["origin", "style_color"], how="left")
K = pd.read_parquet(f"data/breakout_preds_{SRC}_x.parquet")[["origin", "style_color", "rhat"]]; B = B.merge(K, on=["origin", "style_color"], how="left")
MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]
def band(F, y): r = np.where(y > 0, np.abs(F - y) / np.maximum(y, 1e-9), np.where(F == 0, 0, np.inf)); return np.where(r <= 0.2, 1, np.where(r <= 0.5, 2, 3))
B["w"] = (B.q75 - B.q25) / (2 * B.q50.clip(lower=1)); B["mspread"] = B[MEM].clip(lower=1).max(1) / B[MEM].clip(lower=1).min(1)
B["rule_wb"] = np.minimum(3, np.where(B.w <= 0.2, 1, np.where(B.w <= 0.5, 2, 3)) + (B.mspread > 2).astype(int))
B["vrank"] = B.groupby("origin").R3.rank(pct=True); B["f_r3"] = np.log((B.geo + 1) / (B.R3 + 1))
for sysn in ["geo", "p2c32a"]: B[f"band_{sysn}"] = band(B[sysn].values, B.y.values)
X = ["w", "mspread", "f_r3", "level", "age_m", "vrank", "disc_item", "trend91", "rhat", "own_sea", "sh_tt"]
oo = pd.PeriodIndex(B.origin, freq="M").astype(int); B["rule_ml"] = np.nan; B["rule_ml_med"] = np.nan
for o in sorted(B.origin.unique()):
    tr = np.asarray(oo + 4 <= pd.Period(o).ordinal); te = (B.origin == o).values
    if B.origin[tr].nunique() < 2: continue
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.04, num_leaves=7, min_child_samples=50, reg_lambda=5, verbose=-1, n_jobs=8).fit(B.loc[tr, X], B.loc[tr, "band_geo"])
    p = m.predict_proba(B.loc[te, X]); cls = m.classes_
    B.loc[te, "rule_ml"] = cls[p.argmax(1)]; cp = p.cumsum(1); B.loc[te, "rule_ml_med"] = cls[(cp >= 0.5).argmax(1)]
REC = B[(B.origin >= "2025-06")].copy()
out = {}
def report(D, rule, bandc, name):
    D = D[D[rule].notna()]; cm = pd.crosstab(D[rule].astype(int), D[bandc], margins=True); cmu = pd.crosstab(D[rule].astype(int), D[bandc], values=D.y, aggfunc="sum").fillna(0)
    rowp = pd.crosstab(D[rule].astype(int), D[bandc], normalize="index")
    acc = (D[rule] == D[bandc]).mean(); kap = cohen_kappa_score(D[rule].astype(int), D[bandc], weights="quadratic")
    print(f"\n-- {name}: n {len(D)}, accuracy {acc:.3f}, quadratic kappa {kap:+.3f}, unit-weighted accuracy {np.average(D[rule] == D[bandc], weights=D.y + 1):.3f}")
    print("   rows = stated confidence, columns = realised band (1: within 20%, 2: 20-50%, 3: more than 50%); counts:"); print(cm.to_string())
    print("   row %:"); print((rowp * 100).round(1).to_string())
    print("   cumulative hit rate per stated level: " + "; ".join(f"level {int(l)}: within 20% {(g[bandc] == 1).mean():.0%}, within 50% {(g[bandc] <= 2).mean():.0%} (n {len(g)})" for l, g in D.groupby(rule)))
    return {"n": int(len(D)), "accuracy": round(float(acc), 4), "kappa": round(float(kap), 4), "counts": {str(int(i)): {str(int(j)): int(cm.loc[i, j]) for j in [1, 2, 3] if j in cm.columns} for i in cm.index if i != "All"},
            "units": {str(int(i)): {str(int(j)): float(cmu.loc[i, j]) for j in cmu.columns} for i in cmu.index}}
print("== realised bands of the final system, recent origins:", REC.band_geo.value_counts(normalize=True).sort_index().round(3).to_dict(), "| p2c32a:", REC.band_p2c32a.value_counts(normalize=True).sort_index().round(3).to_dict())
for rule in ["rule_wb", "rule_ml", "rule_ml_med"]:
    out[rule] = report(REC, rule, "band_geo", f"{rule} vs the final system's realised band (recent origins)")
out["rule_wb_all13"] = report(B, "rule_wb", "band_geo", "rule_wb, all 13 origins")
# intervals
REC["ly"] = np.log((REC.y + 1) / (REC.geo + 1))
for t in ["q10", "q25", "q75", "q90"]: REC[f"a_{t}"] = REC.geo * REC[t] / REC.q50.clip(lower=1)
B["ly"] = np.log((B.y + 1) / (B.geo + 1))
for t, tau in [("q10", 0.1), ("q25", 0.25), ("q75", 0.75), ("q90", 0.9)]: REC[f"g_{t}"] = np.nan; REC[f"c_{t}"] = np.nan
for o in sorted(REC.origin.unique()):
    tr = B[pd.PeriodIndex(B.origin, freq="M").astype(int) + 4 <= pd.Period(o).ordinal]; te = (REC.origin == o).values
    for t, tau in [("q10", 0.1), ("q25", 0.25), ("q75", 0.75), ("q90", 0.9)]:
        REC.loc[te, f"g_{t}"] = (REC.geo[te] + 1) * np.exp(np.quantile(tr.ly, tau)) - 1
        lv = tr.groupby("rule_wb").ly.quantile(tau); REC.loc[te, f"c_{t}"] = (REC.geo[te] + 1) * np.exp(REC.rule_wb[te].map(lv).fillna(np.quantile(tr.ly, tau)).values) - 1
def cov(D, p):
    c80 = ((D.y >= D[f"{p}_q10"]) & (D.y <= D[f"{p}_q90"])).mean(); c50 = ((D.y >= D[f"{p}_q25"]) & (D.y <= D[f"{p}_q75"])).mean()
    return c80, c50, float(np.median((D[f"{p}_q90"] + 1) / (D[f"{p}_q10"] + 1)))
print("\n== intervals around the final forecast, recent origins: coverage of P10-P90 (nominal 80%) / P25-P75 (nominal 50%) / median width P90 / P10")
out["intervals"] = {}
for p, nm in [("a", "TabPFN conformal member's relative quantiles"), ("g", "time-honest conformal, global"), ("c", "time-honest conformal, per stated level")]:
    c80, c50, wd = cov(REC, p); out["intervals"][nm] = {"cov80": round(c80, 3), "cov50": round(c50, 3), "width": round(wd, 2), "by_level": {}}
    print(f"  {nm:48s} {c80:.3f} / {c50:.3f} / {wd:.2f}")
    for l, g in REC.groupby("rule_wb"):
        a, b, c = cov(g, p); out["intervals"][nm]["by_level"][str(int(l))] = {"cov80": round(a, 3), "cov50": round(b, 3), "width": round(c, 2), "n": int(len(g))}
        print(f"      level {int(l)} (n {len(g)}): {a:.3f} / {b:.3f} / {c:.2f}")
json.dump(out, open(f"data/calibration{OUTS}.json", "w"), indent=1)
# dashboard datasets ({rows, meta}; shares in percent, rounded as displayed)
code = open(__file__, encoding="utf-8").read(); META = [{"label": "183_confidence.py", "value": code, "type": "code", "lang": "python"},
       {"label": "Input", "value": "Backtest item-origins, nine recent origins (Jun 2025 - Feb 2026): data/dec_base_ec_ns.parquet, data/geo_ec_ns_x_preds.parquet, the conformal TabPFN member's quantiles"}]
NAMES = {"rule_wb": ("wb", "Workbook rule (TabPFN range width + model disagreement)"), "rule_ml": ("ml", "Learned rule (time-honest 3-class model)")}
cm_rows = []
for rule, (rid, rname) in NAMES.items():
    D = REC[REC[rule].notna()]
    for l, g in D.groupby(rule):
        cm_rows.append({"rule": rid, "rule_name": rname, "stated": int(l), "n": int(len(g)), "within20": round(100 * (g.band_geo == 1).mean(), 1),
                        "b20to50": round(100 * (g.band_geo == 2).mean(), 1), "over50": round(100 * (g.band_geo == 3).mean(), 1), "within50": round(100 * (g.band_geo <= 2).mean(), 1)})
json.dump({"rows": cm_rows, "meta": META}, open(f"data/calib_cm{OUTS}.json", "w", encoding="utf-8"))
MID = {"TabPFN conformal member's relative quantiles": "pfn", "time-honest conformal, global": "conf", "time-honest conformal, per stated level": "conf_lvl"}
int_rows = []
for nm, v in out["intervals"].items():
    int_rows.append({"method": MID[nm], "method_name": nm[0].upper() + nm[1:], "level": "all", "n": int(len(REC)), "cov80": round(100 * v["cov80"], 1), "cov50": round(100 * v["cov50"], 1), "width": v["width"]})
    for l, b in v["by_level"].items(): int_rows.append({"method": MID[nm], "method_name": nm[0].upper() + nm[1:], "level": l, "n": b["n"], "cov80": round(100 * b["cov80"], 1), "cov50": round(100 * b["cov50"], 1), "width": b["width"]})
json.dump({"rows": int_rows, "meta": META}, open(f"data/calib_int{OUTS}.json", "w", encoding="utf-8")); print("dashboard files written")
