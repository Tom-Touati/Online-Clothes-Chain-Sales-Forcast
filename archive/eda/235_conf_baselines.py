"""D440-D445 (side session of Fork 2; the user's two questions on the submitted forecast: "how we lost to the other models, using the
confidence levels" and "use some of the simpler models for confidence estimation"). Analysis only: the submission is not changed.
Backtest: the 13 time-honest month-ends 2025-02 .. 2026-02, next 4 months' total per item (data/fixes_items.parquet's 11,727 item-origins).
  SYS = geo (data/geo_nx5r_fash_x_preds.parquet) x exp(0.22 x_cmp) (= scoreboard nx9_geo, the submitted numbers); R3 = last 3 months x 4;
  SN = R3's monthly level x the product type's like-for-like seasonal ratio per target month (as eda/159); AR = the agents' rules
  (data/dec_base_nx5r26.parquet, column rules). Metrics as eda/232 / eda/score_lib.py (pinball: the stored quantile spread x 1.25 round each centre).
Q1: SYS against R3 / SN / AR by confidence level (the submitted hybrid D389k: D318's level 1, 3 where the calibrated ranges give
  P(within +/-50%) < 0.5, else 2; and D318's own levels, 3 = widest fifth of the TabPFN width), per month-end, and segments where SYS loses.
Q2: confidence rules from the simple models: SYS's agreement with AR / SN / R3 and their spread, the item's own past windows (only windows
  that had ended by the origin: o' + 4 <= o), similar items' past errors (product type x age band); alone and with the current rule's inputs.
  Thresholds chosen on the tune month-ends 2025-06 .. 09; judged on held out 2025-10 .. 2026-02, the recent nine and the two focus windows.
  Claim met: level 1 within +/-20%, 2 within +/-50%, 3 off by more than 50%. Style-bootstrap 95% intervals of the difference to the current rule.
  Level 1 never meets more claims than level 2 for the same item (within 20% implies within 50%), so the share of claims met always rises
  when level 1 shrinks: level-1 candidates are size-matched (same number of level-1 items per month-end as D318) and judged on which items they pick.
stdout -> eda/conf_baselines_output.txt; writes data/conf_baselines_q1.csv, data/conf_baselines_bt.parquet, data/conf_baselines_2026.csv."""
import sys, importlib.util, warnings, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 60)
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
K = ["origin", "style_color"]; MODELS = ["SYS", "R3", "SN", "AR"]

# ------------------------------------------------------------------ data
T = pd.read_parquet("data/fixes_items.parquet")[K + ["y", "R3", "target"]].reset_index(drop=True)
IX = pd.MultiIndex.from_frame(T[K])
def col(f, c): return pd.read_parquet(f).drop_duplicates(K).set_index(K).reindex(IX)[c].values
x_cmp = np.nan_to_num(col("data/campaign_feats.parquet", "x_cmp").astype(float))
T["geo"] = col("data/geo_nx5r_fash_x_preds.parquet", "geo"); T["SYS"] = T.geo * np.exp(0.22 * x_cmp)
T["AR"] = col("data/dec_base_nx5r26.parquet", "rules")
fe = pd.read_parquet("data/features.parquet"); fe["st"] = fe.m3 * np.exp(fe.lfl_type.fillna(0))
T["SN"] = fe.groupby(K).st.sum().reindex(IX).values
for c in ["trend91", "has_ly", "style", "ptype", "age_m"]: T[c] = col("data/fk_err_master.parquet", c)
qc = pd.read_parquet("data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet").drop_duplicates(K).set_index(K).reindex(IX)
T["iw"] = np.log((qc.q90.values + 1) / (qc.q10.values + 1))
for c in ["p20", "p50"]: T[c] = col("data/confidence_new_bt_nx5r_fash.parquet", c)
qq = pd.read_parquet("data/q_backtest.parquet").drop_duplicates(K).set_index(K).reindex(IX)
LQ = np.log1p(qq[["q10", "q25", "q50", "q75", "q90"]].values); D = {c: LQ[:, j] - LQ[:, 2] for j, c in enumerate(["q10", "q25", "q50", "q75", "q90"])}
T["o_ord"] = T.origin.str[:4].astype(int) * 12 + T.origin.str[5:].astype(int)
O = T.origin.values; y = T.y.values
print(f"{len(T)} item-origins, {T.origin.nunique()} month-ends; missing forecasts: " + ", ".join(f"{c} {int(T[c].isna().sum())}" for c in MODELS))

# ------------------------------------------------------------------ levels: D318 and the submitted hybrid (D389k)
T["fifth"] = T.groupby("origin")["iw"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)) + 1
T["gate"] = (T.trend91.fillna(-1) > 0) & (T.has_ly.fillna(0) == 1)
T["d318"] = np.where((T.fifth == 1) & T.gate, 1, np.where(T.fifth == 5, 3, 2))
T["hyb"] = np.where(T.d318 == 1, 1, np.where(T.p50 < 0.5, 3, 2))
ape = lambda c: np.abs(T[c].values - y) / np.maximum(y, 1e-9)
eS, eG = ape("SYS"), ape("geo")
def met(lv, e=eS): return np.where(lv == 1, e <= 0.2, np.where(lv == 2, e <= 0.5, e > 0.5))
SUBS = {"all 13": np.ones(len(T), bool), "early 4 (2025-02..05)": O <= "2025-05", "recent 9 (2025-06..2026-02)": O >= "2025-06",
        "tune (2025-06..09)": (O >= "2025-06") & (O <= "2025-09"), "held out (2025-10..2026-02)": O >= "2025-10",
        "Jul-Oct 25 (2025-06)": O == "2025-06", "Mar-Jun 26 (2026-02)": O == "2026-02"}
TUNE, HOLD, REC = SUBS["tune (2025-06..09)"], SUBS["held out (2025-10..2026-02)"], SUBS["recent 9 (2025-06..2026-02)"]
w = lambda c, m: np.abs(T[c].values[m] - y[m]).sum() / y[m].sum()
print("check, recent 9 WAPE (metrics table nx12: SYS 0.265, R3 0.699, SN 0.436, AR 0.341):", {c: round(w(c, REC), 4) for c in MODELS})
E4 = SUBS["early 4 (2025-02..05)"]
print("check, early 4 WAPE / R3 (brief: SYS 0.68, AR 0.59):", {c: round(w(c, E4) / w("R3", E4), 3) for c in ["SYS", "SN", "AR"]})
print(f"check, hybrid claims met held out: on geo (pre-campaign, as D389k) {met(T.hyb.values, eG)[HOLD].mean():.3f} (D389k 0.813); on SYS {met(T.hyb.values)[HOLD].mean():.3f}")

# ================================================================== Q1: SYS against the baselines by confidence level
def mets(m):
    F = T[m]; yy = F.y.values; r = {}
    for c in MODELS:
        f = F[c].values.astype(float); e = np.abs(f - yy); a = e / np.maximum(yy, 1e-9)
        r[("WAPE", c)] = e.sum() / yy.sum(); r[("median err", c)] = np.median(a[yy > 0]) if (yy > 0).any() else np.nan
        r[("within20", c)] = ((e <= 0.2 * yy) & (yy > 0)).mean(); r[("bias", c)] = f.sum() / yy.sum() - 1; r[("pinball", c)] = sl.pin(F, c, D)
    es = np.abs(F.SYS.values - yy)
    for c in ["R3", "SN", "AR"]:
        eb = np.abs(F[c].values - yy); r[("closer than SYS", c)] = ((eb < es) + 0.5 * (eb == es)).mean()
    return r
rows = []
for sub, ms in SUBS.items():
    rows.append({"subset": sub, "levels": "-", "level": "all", "n": int(ms.sum()), "share": 1.0, **mets(ms)})
    for sch, lab in [("hyb", "submitted hybrid"), ("d318", "D318")]:
        for lv in (1, 2, 3):
            m = ms & (T[sch].values == lv)
            if m.sum(): rows.append({"subset": sub, "levels": lab, "level": lv, "n": int(m.sum()), "share": m.sum() / ms.sum(), **mets(m)})
Q1 = pd.DataFrame(rows).set_index(["subset", "levels", "level", "n", "share"]); Q1.columns = pd.MultiIndex.from_tuples(Q1.columns)
Q1f = Q1.copy(); Q1f.columns = [f"{a} {b}" for a, b in Q1.columns]; Q1f.to_csv("data/conf_baselines_q1.csv")
print("\n" + "=" * 100 + "\nQ1. SYS against R3 / SN / AR by confidence level (backtest, next 4 months' total per item)\n" + "=" * 100)
A = pd.DataFrame(index=Q1.index)
for c in MODELS: A[f"WAPE {c}"] = Q1[("WAPE", c)]
for c in ["SYS", "SN", "AR"]: A[f"{c}/R3"] = Q1[("WAPE", c)] / Q1[("WAPE", "R3")]
A["SYS/SN"] = Q1[("WAPE", "SYS")] / Q1[("WAPE", "SN")]; A["SYS/AR"] = Q1[("WAPE", "SYS")] / Q1[("WAPE", "AR")]
for c in ["R3", "SN", "AR"]: A[f"{c} closer"] = Q1[("closer than SYS", c)]
print("\n-- WAPE, WAPE as a share of R3's, SYS's WAPE over SN's / AR's (> 1 = SYS loses), share of items where the baseline is closer than SYS (ties half)")
print(A.round(3).to_string())
Bt = pd.DataFrame(index=Q1.index)
for met_name in ["median err", "within20", "bias", "pinball"]:
    for c in MODELS: Bt[f"{met_name} {c}"] = Q1[(met_name, c)]
Bt["pinball SYS/AR"] = Q1[("pinball", "SYS")] / Q1[("pinball", "AR")]
print("\n-- median per-item error, share of items within +/-20%, bias (forecast / actual - 1), pinball P10-P90 (/ units)")
print(Bt.round(3).to_string())

# per month-end
print("\n-- per month-end (all items): WAPE / R3, SYS over SN / AR, baseline closer than SYS, bias; SYS / AR WAPE within each submitted level")
pr = []
for o in sorted(T.origin.unique()):
    m = O == o; r = mets(m); d = {"origin": o, "n": int(m.sum())}
    for c in ["SYS", "SN", "AR"]: d[f"{c}/R3"] = r[("WAPE", c)] / r[("WAPE", "R3")]
    d["SYS/SN"] = r[("WAPE", "SYS")] / r[("WAPE", "SN")]; d["SYS/AR"] = r[("WAPE", "SYS")] / r[("WAPE", "AR")]
    d["SN closer"] = r[("closer than SYS", "SN")]; d["AR closer"] = r[("closer than SYS", "AR")]
    d["bias SYS"] = r[("bias", "SYS")]; d["bias SN"] = r[("bias", "SN")]; d["bias AR"] = r[("bias", "AR")]
    for lv in (1, 2, 3):
        mm = m & (T.hyb.values == lv); d[f"SYS/AR L{lv}"] = w("SYS", mm) / w("AR", mm) if mm.sum() else np.nan
    pr.append(d)
print(pd.DataFrame(pr).set_index("origin").round(3).to_string())

# segments where SYS loses
lgv = {c: np.log1p(np.clip(T[c].values.astype(float), 0, None)) for c in MODELS}
segs = {"age at the month-end": pd.cut(T.age_m, [-np.inf, 3, 6, 12, 24, np.inf], labels=["<3m", "3-6m", "6-12m", "12-24m", "24m+"]).astype(str),
        "sold in the same months last year": T.has_ly.map({1: "yes", 0: "no"}).fillna("unknown"),
        "TabPFN width fifth (1 narrow)": T.fifth.astype(int).astype(str),
        "91-day trend": np.where(T.trend91.fillna(-1) > 0, "rising", "not rising"),
        "volume fifth (R3, within month-end; 1 low)": (T.groupby("origin").R3.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)) + 1).astype(str),
        "|log SYS / AR|": pd.cut(np.abs(lgv["SYS"] - lgv["AR"]), [-1, 0.1, 0.25, 0.5, np.inf], labels=["<0.10", "0.10-0.25", "0.25-0.50", ">0.50"]).astype(str),
        "SYS against AR": np.where(T.SYS > T.AR, "SYS above AR", "SYS at or below AR"),
        "|log SYS / SN|": pd.cut(np.abs(lgv["SYS"] - lgv["SN"]), [-1, 0.1, 0.25, 0.5, np.inf], labels=["<0.10", "0.10-0.25", "0.25-0.50", ">0.50"]).astype(str),
        "the 100 targets": np.where(T.target.fillna(False).astype(bool), "target", "other")}
top_pt = T.ptype.value_counts().index[:10]; segs["product type (top 10)"] = np.where(T.ptype.isin(top_pt), T.ptype.astype(str), "other")
for sub in ["early 4 (2025-02..05)", "recent 9 (2025-06..2026-02)"]:
    ms = SUBS[sub]; out = []
    for name, s in segs.items():
        s = np.asarray(s)
        for v in sorted(pd.unique(s[ms])):
            m = ms & (s == v)
            if m.sum() < 60: continue
            r = mets(m)
            out.append({"segment": name, "value": v, "n": int(m.sum()), "SYS/R3": r[("WAPE", "SYS")] / r[("WAPE", "R3")], "SN/R3": r[("WAPE", "SN")] / r[("WAPE", "R3")],
                        "AR/R3": r[("WAPE", "AR")] / r[("WAPE", "R3")], "SYS/SN": r[("WAPE", "SYS")] / r[("WAPE", "SN")], "SYS/AR": r[("WAPE", "SYS")] / r[("WAPE", "AR")],
                        "SN closer": r[("closer than SYS", "SN")], "AR closer": r[("closer than SYS", "AR")], "bias SYS": r[("bias", "SYS")], "bias AR": r[("bias", "AR")],
                        "units share": y[m].sum() / y[ms].sum(), "loses to": " ".join(b for b in ["SN", "AR"] if r[("WAPE", "SYS")] > r[("WAPE", b)])})
    print(f"\n-- segments, {sub} (>= 60 item-origins): WAPE / R3, SYS over SN / AR (> 1 = SYS loses on units), share of items where the baseline is closer")
    print(pd.DataFrame(out).set_index(["segment", "value"]).round(3).to_string())

# ================================================================== Q2: confidence from the simple models
print("\n" + "=" * 100 + "\nQ2. Confidence rules from the simple models (claims met by the brief's definitions; SYS's errors)\n" + "=" * 100)
T["aAR"] = np.abs(lgv["SYS"] - lgv["AR"]); T["aSN"] = np.abs(lgv["SYS"] - lgv["SN"]); T["aR3"] = np.abs(lgv["SYS"] - lgv["R3"])
T["spread"] = np.nanstd(np.column_stack([lgv[c] for c in MODELS]), axis=1)          # spread of log(1 + forecast) across SYS, R3, SN, AR
T["ageb"] = pd.cut(T.age_m, [-np.inf, 6, 12, np.inf], labels=["<6m", "6-12m", "12m+"]).astype(str); T["cell"] = T.ptype.astype(str) + "|" + T.ageb
ly = np.log1p(y); LE = {c: np.abs(lgv[c] - ly) for c in MODELS}
PW = pd.DataFrame({"style_color": T.style_color, "o_ord": T.o_ord, "cell": T.cell, "ageb": T.ageb, "eS": LE["SYS"], "eA": LE["AR"], "eN": LE["SN"], "eR": LE["R3"],
                   "w20": (eS <= 0.2).astype(float), "w50": (eS <= 0.5).astype(float)})
AGG = dict(eS="mean", eA="mean", eN="mean", eR="mean", w20="mean", w50="mean")
def past_feats(oo, R):
    """Features of rows R (style_color, cell, ageb, lS = log1p SYS, lA, lN, lR) at month-end oo from the windows that had ended by oo."""
    P = PW[PW.o_ord + 4 <= oo]; n = len(R); out = pd.DataFrame(index=range(n))
    if not len(P):
        for c in ["pe_S", "pe_A", "pe_N", "pe_R", "pw20", "pw50", "npast", "gap", "a_best", "cw20", "cw50", "ceS", "ceA", "ceN", "ceR"]: out[c] = np.nan
        out["npast"] = 0.0; out["best_base"] = "none"; return out
    it = P.groupby("style_color").agg(**{f"pe_{k}": (f"e{k}", "mean") for k in "SANR"}, pw20=("w20", "mean"), pw50=("w50", "mean"), npast=("eS", "size"))
    it = it.reindex(R.style_color.values).reset_index(drop=True); out = out.join(it); out["npast"] = out.npast.fillna(0)
    ce = P.groupby("cell").agg(**{f"ce{k}": (f"e{k}", "mean") for k in "SANR"}, cw20=("w20", "mean"), cw50=("w50", "mean"), cn=("eS", "size"))
    ab = P.groupby("ageb").agg(**{f"ce{k}": (f"e{k}", "mean") for k in "SANR"}, cw20=("w20", "mean"), cw50=("w50", "mean"), cn=("eS", "size"))
    c1 = ce.reindex(R.cell.values).reset_index(drop=True); c2 = ab.reindex(R.ageb.values).reset_index(drop=True)
    small = ~(c1.cn.fillna(0) >= 30).values; c1.loc[small] = c2.loc[small].values           # fewer than 30 past rows in the cell: the age band alone
    out = out.join(c1.drop(columns="cn"))
    pb = out[["pe_A", "pe_N", "pe_R"]].values; has = np.isfinite(pb).any(1)
    best = np.full(n, -1); best[has] = np.nanargmin(np.where(np.isfinite(pb[has]), pb[has], np.inf), axis=1)
    out["gap"] = out.pe_S.values - np.where(has, np.nanmin(np.where(np.isfinite(pb), pb, np.inf), axis=1), np.nan)   # > 0: a baseline beat SYS on this item before
    lb = np.column_stack([R.lA.values, R.lN.values, R.lR.values])
    out["a_best"] = np.where(best >= 0, np.abs(R.lS.values - lb[np.arange(n), np.clip(best, 0, 2)]), np.nan)     # agreement with the item's best past baseline
    out["best_base"] = np.where(best >= 0, np.array(["AR", "SN", "R3"])[np.clip(best, 0, 2)], "none")
    return out
R_all = pd.DataFrame({"style_color": T.style_color, "cell": T.cell, "ageb": T.ageb, "lS": lgv["SYS"], "lA": lgv["AR"], "lN": lgv["SN"], "lR": lgv["R3"]})
PF = []
for o in sorted(T.origin.unique()):
    m = (O == o); f = past_feats(int(T.o_ord[m].iloc[0]), R_all[m].reset_index(drop=True)); f.index = np.flatnonzero(m); PF.append(f)
PF = pd.concat(PF).sort_index()
for c in PF.columns: T[c] = PF[c].values
T["has_past"] = (T.npast > 0).astype(float); T["rising"] = (T.trend91.fillna(-1) > 0).astype(float); T["ly"] = T.has_ly.fillna(0).astype(float)
print("past windows available (share of item-origins with >= 1 own past window): " + ", ".join(f"{o} {T.has_past[O == o].mean():.2f}" for o in sorted(T.origin.unique()) if o >= "2025-06"))
print("item's most accurate past baseline (recent 9): " + str(T.loc[REC, "best_base"].value_counts(normalize=True).round(3).to_dict()))

FA = ["aAR", "aSN", "aR3", "spread"]                                                   # agreement with the simple models
FP = ["has_past", "pe_S", "gap", "a_best", "pw20", "pw50", "cw20", "cw50", "ceS", "ceA", "ceN", "ceR"]   # own past windows + similar items
FC = ["iw", "rising", "ly"]                                                                # D318's inputs (the ranges are used only as the current rule does: p50 < t)
W20, W50 = (eS <= 0.2).astype(int), (eS <= 0.5).astype(int)
def fit_predict(feats, target, origins, Xnew=None, onew=None):
    """Expanding window: the model for month-end o is fitted on item-origins whose window had ended by o (o' + 4 <= o)."""
    P = np.full(len(T), np.nan); pn = None
    for o in list(origins) + ([onew] if Xnew is not None else []):
        oo = int(o[:4]) * 12 + int(o[5:]); tr = (T.o_ord.values + 4 <= oo) & np.isfinite(y)
        X = T.loc[tr, feats]; use = [c for c in feats if X[c].notna().any() and X[c].std() > 0]
        med = X[use].median(); Xtr = X[use].fillna(med); mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
        mdl = LogisticRegression(C=1.0, max_iter=3000).fit((Xtr - mu) / sd, target[tr])
        if Xnew is not None and o == onew: pn = mdl.predict_proba((Xnew[use].fillna(med) - mu) / sd)[:, 1]
        else: te = O == o; P[te] = mdl.predict_proba((T.loc[te, use].fillna(med) - mu) / sd)[:, 1]
    return P, pn
ORIG_Q2 = [o for o in sorted(T.origin.unique()) if o >= "2025-06"]
MODS = {"A": FA, "AP": FA + FP, "APC": FA + FP + FC}
for k, f in MODS.items():
    T[f"P20_{k}"], _ = fit_predict(f, W20, ORIG_Q2); T[f"P50_{k}"], _ = fit_predict(f, W50, ORIG_Q2)
print("calibration, held out: mean P(within 50%) vs actual " + ", ".join(f"{k} {T.loc[HOLD, f'P50_{k}'].mean():.3f}" for k in MODS) + f" | actual {W50[HOLD].mean():.3f};"
      " P(within 20%) " + ", ".join(f"{k} {T.loc[HOLD, f'P20_{k}'].mean():.3f}" for k in MODS) + f" | actual {W20[HOLD].mean():.3f}")

# --- how much each signal sorts the items (AUC; higher signal = more confident; 0.5 = no information)
SIG = {"-TabPFN width (D318)": -T.iw, "p20 (ranges)": T.p20, "p50 (ranges)": T.p50, "-|log SYS/AR|": -T.aAR, "-|log SYS/SN|": -T.aSN, "-|log SYS/R3|": -T.aR3,
       "-spread SYS,R3,SN,AR": -T.spread, "-|log SYS/best past baseline|": -T.a_best, "-SYS's past log error": -T.pe_S, "-gap (SYS past err - best baseline's)": -T.gap,
       "SYS's past within-20 rate": T.pw20, "similar items' within-20 rate": T.cw20, "similar items' within-50 rate": T.cw50,
       "model A (agreement)": None, "model AP (+ past, similar)": None, "model APC (+ D318 inputs)": None}
au = []
for name, s in SIG.items():
    d = {"signal": name}
    for lab, m in [("tune", TUNE), ("held out", HOLD), ("recent 9", REC)]:
        for tgt, Y in [("w20", W20), ("w50", W50)]:
            if s is None: ss = T[f"P{tgt[1:]}_" + name.split()[1]].values
            else: ss = np.asarray(s, float)
            ok = m & np.isfinite(ss); d[f"{lab} {tgt}"] = roc_auc_score(Y[ok], ss[ok]) if ok.sum() > 50 and len(set(Y[ok])) == 2 else np.nan
        if s is not None: d[f"{lab} cover"] = np.isfinite(np.asarray(s, float))[m].mean()
    au.append(d)
print("\n-- AUC of each signal for landing within +/-20% (w20) and within +/-50% (w50) (0.5 = no information); cover = share of items with the signal")
print(pd.DataFrame(au).set_index("signal").round(3).to_string())

# beyond the ranges: AUC inside fifths of p50 (and of the TabPFN width), n-weighted; ~0.5 = nothing beyond what the current rule already reads
def strat_auc(ss, Y, m, by):
    tot, n = 0.0, 0
    for g in np.unique(by[m & np.isfinite(by)]):
        k = m & (by == g) & np.isfinite(ss)
        if k.sum() > 30 and len(set(Y[k])) == 2: tot += roc_auc_score(Y[k], ss[k]) * k.sum(); n += k.sum()
    return tot / n if n else np.nan
p50f = (T.groupby("origin").p50.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)) + 1).values.astype(float)
sa = []
for name, s in SIG.items():
    if s is None or name.startswith("p") or "width" in name: continue
    ss = np.asarray(s, float); d = {"signal": name}
    for lab, m in [("held out", HOLD), ("recent 9", REC)]:
        for tgt, Y in [("w20", W20), ("w50", W50)]:
            d[f"{lab} {tgt} | p50 fifth"] = strat_auc(ss, Y, m, p50f); d[f"{lab} {tgt} | width fifth"] = strat_auc(ss, Y, m, T.fifth.values.astype(float))
    sa.append(d)
print("\n-- the same AUCs inside fifths of p50 and of the TabPFN width (the current rule's inputs): what the simple models add beyond them")
print(pd.DataFrame(sa).set_index("signal").round(3).to_string())

# within D318's level 1: does agreement pick out the hits?
L1m = T.d318.values == 1; cut = np.nanmedian(T.spread.values[L1m & TUNE])
print(f"\n-- inside D318's level 1: within +/-20% by spread (cut = tune median {cut:.3f})")
for lab, m in [("tune", TUNE), ("held out", HOLD), ("recent 9", REC)]:
    lo, hi = m & L1m & (T.spread.values <= cut), m & L1m & (T.spread.values > cut)
    print(f"   {lab:9s} low spread n {lo.sum():4d} within20 {W20[lo].mean():.3f} | high spread n {hi.sum():4d} within20 {W20[hi].mean():.3f}")

# --- the candidate rules
L1d = T.d318.values == 1; p50 = T.p50.values
def lvl(L1, L3): return np.where(L1, 1, np.where(L3, 3, 2)).astype(int)
cur = lvl(L1d, p50 < 0.5)
kref = T.groupby("origin").d318.transform(lambda s: int((s == 1).sum())).values
def topk(score, elig=None, k=kref):
    s = np.where(np.isfinite(score), score, -np.inf)
    if elig is not None: s = np.where(elig, s, -np.inf)
    r = pd.Series(s).groupby(O).rank(ascending=False, method="first").values
    return (r <= k) & np.isfinite(s)
def tune_grid(make, grid):
    sc = {g: met(make(g))[TUNE].mean() for g in grid}; b = max(sc, key=sc.get); return b, sc[b]
qs = lambda v, ps: [float(np.nanquantile(v[TUNE], p)) for p in ps]
RULES, THR = {}, {}
RULES["current hybrid (D389k)"] = cur
RULES["[ref] everything 2"] = np.full(len(T), 2)
RULES["[ref] no level 1, ranges' level 3"] = lvl(np.zeros(len(T), bool), p50 < 0.5)
# level 3 (level 1 = D318's, as now)
sg = qs(T.spread.values, [0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999]) + [np.inf]
b, _ = tune_grid(lambda s: lvl(L1d, T.spread.values >= s), sg); THR["L3 spread"] = b; RULES["L3: spread >= s"] = lvl(L1d, T.spread.values >= b)
b, _ = tune_grid(lambda s: lvl(L1d, (p50 < 0.5) | (T.spread.values >= s)), sg); THR["L3 p50 or spread"] = b; RULES["L3: p50 < 0.5 or spread >= s"] = lvl(L1d, (p50 < 0.5) | (T.spread.values >= b))
g2 = [(t, s) for t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.01] for s in qs(T.spread.values, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95])]
b, _ = tune_grid(lambda g: lvl(L1d, (p50 < g[0]) & (T.spread.values >= g[1])), g2); THR["L3 p50 and spread"] = b
RULES["L3: p50 < t and spread >= s"] = lvl(L1d, (p50 < b[0]) & (T.spread.values >= b[1]))
PCT = [0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
ab_ = qs(T.a_best.values, PCT) + [np.inf]
b, _ = tune_grid(lambda s: lvl(L1d, (p50 < 0.5) | (T.a_best.values >= s)), ab_); THR["L3 p50 or a_best"] = b; RULES["L3: p50 < 0.5 or |log SYS/best past baseline| >= s"] = lvl(L1d, (p50 < 0.5) | (T.a_best.values >= b))
for key, cname, lab in [("L3 p50 or gap", "gap", "SYS's past error - best baseline's"), ("L3 p50 or pe_S", "pe_S", "SYS's past log error")]:
    gg = qs(T[cname].values, PCT) + [np.inf]
    b, _ = tune_grid(lambda s: lvl(L1d, (p50 < 0.5) | (T[cname].values >= s)), gg); THR[key] = b; RULES[f"L3: p50 < 0.5 or {lab} >= s"] = lvl(L1d, (p50 < 0.5) | (T[cname].values >= b))
tg = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
for k in MODS:
    P = T[f"P50_{k}"].values
    b, _ = tune_grid(lambda t: lvl(L1d, P < t), tg); THR[f"L3 model {k}"] = b; RULES[f"L3: model {k} P(w50) < t"] = lvl(L1d, P < b)
    b, _ = tune_grid(lambda t: lvl(L1d, (p50 < 0.5) | (P < t)), tg); THR[f"L3 p50 or model {k}"] = b; RULES[f"L3: p50 < 0.5 or model {k} P(w50) < t"] = lvl(L1d, (p50 < 0.5) | (P < b))
# level 1 (level 3 = the ranges', as now), size-matched to D318's count per month-end
L3c = p50 < 0.5
gate = T.gate.values
rk = lambda v: pd.Series(v).groupby(O).rank(pct=True).values
L1C = {"L1: gate, lowest spread": topk(-T.spread.values, gate),
       "L1: gate, width rank + spread rank": topk(-(rk(T.iw.values) + rk(T.spread.values)), gate),
       "L1: gate, width rank + |log SYS/AR| rank": topk(-(rk(T.iw.values) + rk(T.aAR.values)), gate),
       "L1: model A top-k": topk(T.P20_A.values), "L1: model AP top-k": topk(T.P20_AP.values), "L1: model APC top-k": topk(T.P20_APC.values),
       "L1: gate, model APC top-k": topk(T.P20_APC.values, gate), "L1: gate, model AP top-k": topk(T.P20_AP.values, gate)}
for name, L1 in L1C.items(): RULES[name + " (size-matched)"] = lvl(L1, L3c)
# combined: the best level 1 and the best level 3 on the tune month-ends
l3_names = [n for n in RULES if n.startswith("L3:")]; l1_names = [n for n in RULES if n.startswith("L1:")]
b3 = max(l3_names, key=lambda n: met(RULES[n])[TUNE].mean()); b1 = max(l1_names, key=lambda n: met(RULES[n])[TUNE].mean())
L1b = RULES[b1] == 1; L3b = RULES[b3] == 3
RULES[f"COMBINED: best L1 on tune + best L3 on tune"] = lvl(L1b, L3b)
print("\nthresholds chosen on the tune month-ends:", {k: (round(v, 3) if isinstance(v, float) else tuple(round(x, 3) for x in v)) for k, v in THR.items()})
print(f"best level 3 on tune: {b3} | best level 1 on tune (size-matched): {b1}")

# --- scoring with style-bootstrap intervals against the current rule
styles, sidx = np.unique(T["style"].astype(str).values, return_inverse=True)
def sboot(d, m, B=2000, seed=0):
    rng = np.random.default_rng(seed); Ds = np.bincount(sidx[m], weights=d[m], minlength=len(styles)); Ns = np.bincount(sidx[m], minlength=len(styles))
    pres = np.flatnonzero(Ns > 0); dr = pres[rng.integers(0, len(pres), (B, len(pres)))]
    v = Ds[dr].sum(1) / Ns[dr].sum(1); return np.percentile(v, [2.5, 97.5])
EVAL = [("tune", TUNE), ("held out", HOLD), ("recent 9", REC), ("Jul-Oct 25", O == "2025-06"), ("Mar-Jun 26", O == "2026-02")]
mc = met(cur).astype(float); res = []
for name, lv in RULES.items():
    mm = met(lv).astype(float); d = {"rule": name}
    for lab, m in EVAL:
        d[f"{lab} met"] = mm[m].mean()
        if lab != "tune":
            lo, hi = sboot(mm - mc, m); d[f"{lab} diff"] = mm[m].mean() - mc[m].mean(); d[f"{lab} CI"] = f"[{lo:+.3f}, {hi:+.3f}]"
    for lab, m in [("held out", HOLD)]:
        for k in (1, 3):
            mk = m & (lv == k); d[f"HO L{k} share"] = mk.sum() / m.sum()
            d[f"HO L{k} claim"] = (eS[mk] <= 0.2).mean() if k == 1 and mk.any() else ((eS[mk] > 0.5).mean() if mk.any() else np.nan)
    res.append(d)
RES = pd.DataFrame(res).set_index("rule")
print("\n-- share of claims met (SYS's errors); diff = rule - current hybrid with a 95% style-bootstrap interval (2,000 draws; whole styles resampled)")
print(RES[[c for c in RES.columns if c.endswith(" met") or c.endswith(" diff") or c.endswith(" CI")]].round(3).to_string())
print("\n-- held out: level sizes and how often each level's claim holds (L1: within +/-20%; L3: off by more than 50%)")
print(RES[[c for c in RES.columns if c.startswith("HO ")]].round(3).to_string())
# the per-level read of the main candidates on each window
print("\n-- per level, main candidates: share of items | within +/-20% | within +/-50% per level")
main = ["current hybrid (D389k)", "[ref] everything 2", b3, b1, "COMBINED: best L1 on tune + best L3 on tune", "L3: spread >= s", "L3: model A P(w50) < t", "L1: model A top-k (size-matched)"]
for name in dict.fromkeys(main):
    lv = RULES[name]
    for lab, m in EVAL[1:]:
        s = f"   {name[:58]:58s} {lab:10s} met {met(lv)[m].mean():.3f} |"
        for k in (1, 2, 3):
            mk = m & (lv == k); s += f" L{k} {mk.sum() / m.sum():5.1%} w20 {np.mean(eS[mk] <= .2) if mk.any() else np.nan:.2f} w50 {np.mean(eS[mk] <= .5) if mk.any() else np.nan:.2f} |"
        print(s)
# is there a recommendation? beats the current rule on held out with the interval above zero, not a degenerate reference, level 1 kept
ok = [n for n in RES.index if not n.startswith("[ref]") and n != "current hybrid (D389k)" and float(RES.at[n, "held out CI"].split(",")[0][1:]) > 0 and RES.at[n, "HO L1 share"] > 0]
print("\nrules beating the current hybrid on held out with the style-bootstrap interval above zero (level 1 kept):", ok if ok else "none")

# ================================================================== 2026: what the candidates would do to the 100 targets
C = pd.read_csv("data/final_2026_nx14_cascade.csv", index_col=0)
fc = pd.read_csv("data/fk_confidence_2026.csv").set_index("style_color").reindex(C.index)
db = pd.read_parquet("data/dec_base_nx5r26.parquet"); db = db[db.origin == "2026-06"].drop_duplicates("style_color").set_index("style_color").reindex(C.index)
N = pd.DataFrame({"style_color": C.index, "SYS": C.final.values, "AR": C.rules.values, "SN": C.season.values, "R3": C.R3.values, "p50": C.p_within50.values,
                  "cur": C.confidence.values, "d318_L1": (fc.level.values == 1), "iw": fc.iw.values, "rising": fc.rising.astype(float).values, "ly": fc.ly.astype(float).values,
                  "age_m": db.age_m.values, "ptype": C.product_type.values})
for c, l in [("SYS", "lS"), ("AR", "lA"), ("SN", "lN"), ("R3", "lR")]: N[l] = np.log1p(np.clip(N[c].astype(float), 0, None))
N["aAR"] = np.abs(N.lS - N.lA); N["aSN"] = np.abs(N.lS - N.lN); N["aR3"] = np.abs(N.lS - N.lR); N["spread"] = np.nanstd(N[["lS", "lR", "lN", "lA"]].values, axis=1)
N["ageb"] = pd.cut(N.age_m, [-np.inf, 6, 12, np.inf], labels=["<6m", "6-12m", "12m+"]).astype(str); N["cell"] = N.ptype.astype(str) + "|" + N.ageb
pf = past_feats(2026 * 12 + 6, N[["style_color", "cell", "ageb", "lS", "lA", "lN", "lR"]].reset_index(drop=True))
for c in pf.columns: N[c] = pf[c].values
N["has_past"] = (N.npast > 0).astype(float)
for k, f in MODS.items():
    _, N[f"P20_{k}"] = fit_predict(f, W20, [], N[f], "2026-06"); _, N[f"P50_{k}"] = fit_predict(f, W50, [], N[f], "2026-06")
HAND = C.index[(C.confidence == 3) & ~(C.p_within50 < 0.5)].tolist()            # World Cup x3 and the out-of-stock row stay at 3 (hand rules)
print("\n" + "=" * 100 + f"\n2026 targets (origin 2026-06; 100 items; hand-ruled rows kept at 3: {HAND})\n" + "=" * 100)
k26 = int(N.d318_L1.sum())
def lv26(L1, L3):
    v = np.where(L1, 1, np.where(L3, 3, 2)); v[np.isin(N.style_color, HAND)] = 3; return v
ptk = lambda s, elig=None: (lambda r: r <= k26)(pd.Series(np.where(elig if elig is not None else True, np.where(np.isfinite(s), s, -np.inf), -np.inf)).rank(ascending=False, method="first").values)
rk26 = lambda v: pd.Series(v).rank(pct=True).values
gate26 = (N.rising.values == 1) & (N.ly.values == 1)
L1_26 = {"L1: gate, lowest spread": ptk(-N.spread.values, gate26), "L1: gate, width rank + spread rank": ptk(-(rk26(N.iw.values) + rk26(N.spread.values)), gate26),
         "L1: gate, width rank + |log SYS/AR| rank": ptk(-(rk26(N.iw.values) + rk26(N.aAR.values)), gate26),
         "L1: model A top-k": ptk(N.P20_A.values), "L1: model AP top-k": ptk(N.P20_AP.values), "L1: model APC top-k": ptk(N.P20_APC.values),
         "L1: gate, model APC top-k": ptk(N.P20_APC.values, gate26), "L1: gate, model AP top-k": ptk(N.P20_AP.values, gate26)}
p5 = N.p50.values; L1n = N.d318_L1.values
R26 = {"current hybrid (D389k)": lv26(L1n, p5 < 0.5), "L3: spread >= s": lv26(L1n, N.spread.values >= THR["L3 spread"]),
       "L3: p50 < 0.5 or spread >= s": lv26(L1n, (p5 < 0.5) | (N.spread.values >= THR["L3 p50 or spread"])),
       "L3: p50 < t and spread >= s": lv26(L1n, (p5 < THR["L3 p50 and spread"][0]) & (N.spread.values >= THR["L3 p50 and spread"][1])),
       "L3: p50 < 0.5 or |log SYS/best past baseline| >= s": lv26(L1n, (p5 < 0.5) | (N.a_best.values >= THR["L3 p50 or a_best"])),
       "L3: p50 < 0.5 or SYS's past error - best baseline's >= s": lv26(L1n, (p5 < 0.5) | (N.gap.values >= THR["L3 p50 or gap"])),
       "L3: p50 < 0.5 or SYS's past log error >= s": lv26(L1n, (p5 < 0.5) | (N.pe_S.values >= THR["L3 p50 or pe_S"]))}
for k in MODS:
    R26[f"L3: model {k} P(w50) < t"] = lv26(L1n, N[f"P50_{k}"].values < THR[f"L3 model {k}"])
    R26[f"L3: p50 < 0.5 or model {k} P(w50) < t"] = lv26(L1n, (p5 < 0.5) | (N[f"P50_{k}"].values < THR[f"L3 p50 or model {k}"]))
for name, L1 in L1_26.items(): R26[name + " (size-matched)"] = lv26(L1, p5 < 0.5)
R26["COMBINED: best L1 on tune + best L3 on tune"] = lv26(R26[b1] == 1, R26[b3] == 3)
cnt = pd.DataFrame({n: pd.Series(v).value_counts().reindex([1, 2, 3]).fillna(0).astype(int) for n, v in R26.items()}).T
cnt["moved vs current"] = [int((v != R26["current hybrid (D389k)"]).sum()) for v in R26.values()]
print("levels 1 / 2 / 3 among the 100 targets, and the number of items whose level would change:"); print(cnt.to_string())
assert (R26["current hybrid (D389k)"] == N.cur.values).all(), "the rebuilt current rule does not match the submitted confidence column"
for name in dict.fromkeys([b3, b1, "COMBINED: best L1 on tune + best L3 on tune"] + ok):
    mv = R26[name] != N.cur.values
    if mv.any():
        print(f"\n   {name}: items that would change")
        print(N.loc[mv, ["style_color", "SYS", "AR", "SN", "R3", "p50", "spread", "aAR", "P50_A", "P20_APC", "cur"]].assign(new=R26[name][mv]).round(3).to_string(index=False))
out26 = N[["style_color", "SYS", "AR", "SN", "R3", "p50", "aAR", "aSN", "aR3", "spread", "a_best", "best_base", "pe_S", "gap", "npast"] + [f"P{t}_{k}" for k in MODS for t in (20, 50)] + ["cur"]].copy()
for n, v in R26.items(): out26[n] = v
out26.to_csv("data/conf_baselines_2026.csv", index=False)
T[K + ["y"] + MODELS + ["hyb", "d318", "p20", "p50", "iw", "aAR", "aSN", "aR3", "spread", "a_best", "best_base", "pe_S", "gap", "pw20", "cw20", "cw50", "npast"] + [f"P{t}_{k}" for k in MODS for t in (20, 50)]].to_parquet("data/conf_baselines_bt.parquet", index=False)
print("\nwritten: data/conf_baselines_q1.csv, data/conf_baselines_bt.parquet, data/conf_baselines_2026.csv")
