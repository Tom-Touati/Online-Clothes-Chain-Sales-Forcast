"""D240: middle-out through type x look cells (D239: knowing type x look totals would cut WAPE / R3 from 0.398 to 0.260).
Cells at each origin = product type x look cluster (data/dec_base.parquet, D238; also look-only and type-only for comparison).
Cell features (data up to the origin; cell history = the current members' past sales summed): summed level L, cell lags (6 months over L),
cell momentum (last 3 vs 3 before), cell last-year ramp into the 4 target months (members alive a year ago), share of young members,
number of members and styles, level-weighted means of the members' D159 features, and the bottom-up member totals as log ratios to 4L
(rules, D198 quantile, NN, conformal TabPFN, TabPFN dyn, and the final system F).
Cell models (LightGBM median + L2; time-honest, origins o' + 4 <= o): C1 cell features only; C2 + bottom-up member ratios (stacking;
falls back to C1 until 3 training origins exist); C3 residual on bottom-up F (target log((Y + 1) / (BU_F + 1))).
Item versions of F: scaled to the cell forecast fully / at 0.3 / 0.5 and an inverse-variance mix of BU_F and C2 (by cell size, time-honest).
Scored: cell-level WAPE (BU vs models) and the standard item table with bootstrap against F. Writes eda/cells_output.txt, data/cells_preds.parquet."""
import sys, contextlib, io, importlib.util, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
src = open("eda/164_q_weather.py", encoding="utf-8").read(); _a = sys.argv; sys.argv = sys.argv[:1]
exec(src[src.index("import sys"):src.index("for var in RUN:")]); sys.argv = _a
B = pd.read_parquet("data/dec_base.parquet"); MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32", "F"]
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; Pm = lambda o, k: str(pd.Period(o, "M") + k)
first = M.gt(0).idxmax(axis=1).where(M.gt(0).any(axis=1))
NUMF = [c for c in dict.fromkeys(CP + LL + WE + ["lfl_type4", "lfl_macro4", "own_sea4", "lvl"]) if c in T0.columns and T0[c].dtype.kind in "fiub"]
X0 = T0[["origin", "style_color", "level"] + NUMF].merge(B[["origin", "style_color", "ptype", "look", "style", "y"] + MEM], on=["origin", "style_color"], how="inner")
X0["type_look"] = X0.ptype + "|" + X0.look; X0["typ"] = X0.origin + "|" + X0.ptype
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
P = dict(n_estimators=400, learning_rate=0.04, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, n_jobs=8)
fit = lambda tr, te, X, y: np.mean([lgb.LGBMRegressor(objective=ob, **({"alpha": 0.5} if ob == "quantile" else {}), **P).fit(tr[X], tr[y]).predict(te[X]) for ob in ("quantile", "l2")], axis=0)
def cells(cell):
    rows = []
    for o, D in X0.groupby("origin"):
        w = D.level.fillna(0) + 1; g = D.groupby(cell)
        A = D[NUMF].astype(float).fillna(D[NUMF].astype(float).mean()).fillna(0).mul(w, axis=0).groupby(D[cell]).sum().div(w.groupby(D[cell]).sum(), axis=0)
        A["L"] = g["level"].sum(); A["Y"] = g["y"].sum(); A["n_items"] = g.size(); A["n_styles"] = g["style"].nunique()
        for m in MEM: A[f"bu_{m}"] = g[m].sum()
        mem = g.style_color.apply(list)
        hist = lambda ks: np.array([M.reindex(v)[[Pm(o, k) for k in ks if Pm(o, k) in M.columns]].sum().sum() for v in mem])
        for k in range(6): A[f"c_lag{k+1}"] = np.log((hist([-k]) + 1) / (A.L.values + 1))
        A["c_mom3"] = np.log((hist([0, -1, -2]) + 1) / (hist([-3, -4, -5]) + 1))
        A["c_ramp"] = np.log((hist([-11, -10, -9, -8]) + 1) / (hist([-12, -13, -14]) * 4 / 3 + 1))
        A["c_young"] = [np.mean([(pd.Period(o, "M") - pd.Period(first[s], "M")).n < 6 if isinstance(first.get(s), str) else 0 for s in v]) for v in mem]
        A["origin"] = o; A["o_ord"] = pd.Period(o).ordinal; rows.append(A.reset_index().rename(columns={cell: "cell"}))
    S = pd.concat(rows, ignore_index=True)
    for m in MEM: S[f"r_{m}"] = np.log((S[f"bu_{m}"] + 1) / (4 * S.L + 1))
    S["z"] = np.log((S.Y + 1) / (4 * S.L + 1)); S["zr"] = np.log((S.Y + 1) / (S.bu_F + 1)); return S
X1 = NUMF + ["L", "n_items", "n_styles"] + [f"c_lag{k}" for k in range(1, 7)] + ["c_mom3", "c_ramp", "c_young"]; X2 = X1 + [f"r_{m}" for m in MEM]
out_items = {}
for cell in ["type_look", "look", "ptype"]:
    S = cells(cell); res = []
    for o in sorted(S.origin.unique()):
        oo = pd.Period(o).ordinal; tr = S[S.o_ord + 4 <= oo]; te = S[S.origin == o].copy()
        if tr.origin.nunique() < 1: continue
        te["C1"] = (4 * te.L + 1) * np.exp(fit(tr, te, X1, "z")) - 1
        if tr.origin.nunique() >= 3: te["C2"] = (4 * te.L + 1) * np.exp(fit(tr, te, X2, "z")) - 1; te["C3"] = te.bu_F * np.exp(fit(tr, te, X2, "zr"))
        else: te["C2"] = te.bu_F; te["C3"] = te.bu_F
        res.append(te[["origin", "cell", "L", "Y", "bu_F", "C1", "C2", "C3"]])
    R = pd.concat(res); RR = R[R.origin >= "2025-06"]; yy = RR.Y.sum()
    print(f"\n== cells = {cell}: cell-level WAPE (recent origins; {RR.groupby('origin').size().mean():.0f} cells per origin)")
    print("  BU of F %.4f | C1 %.4f | C2 %.4f | C3 %.4f" % tuple((RR[c] - RR.Y).abs().sum() / yy for c in ["bu_F", "C1", "C2", "C3"]))
    # inverse-variance mix of BU and C2 by cell size (time-honest)
    R["eb"] = np.log((R.Y + 1) / (R.bu_F + 1)); R["et"] = np.log((R.Y + 1) / (R.C2 + 1)); R["bk"] = pd.qcut(np.log1p(R.bu_F), 4, labels=False, duplicates="drop"); R["wt"] = 0.0
    for o in R.origin.unique():
        past = R[pd.PeriodIndex(R.origin, freq="M").astype(int) + 4 <= pd.Period(o).ordinal]
        for b in R.bk.dropna().unique():
            pb = past[past.bk == b]; m = (R.origin == o) & (R.bk == b)
            R.loc[m, "wt"] = (1 / pb.et.var()) / (1 / pb.et.var() + 1 / pb.eb.var()) if pb.origin.nunique() >= 3 else 0.0
    R["Cm"] = np.exp(R.wt * np.log(R.C2.clip(lower=0.5)) + (1 - R.wt) * np.log(R.bu_F.clip(lower=0.5)))
    print("  mix of BU and C2: %.4f (mean weight on C2, recent: %.2f)" % ((R[R.origin >= "2025-06"].Cm - R[R.origin >= "2025-06"].Y).abs().sum() / yy, R[R.origin >= "2025-06"].wt.mean()))
    I = X0[["origin", "style_color", cell, "F"]].rename(columns={cell: "cell"}).merge(R[["origin", "cell", "bu_F", "C2", "C3", "Cm"]], on=["origin", "cell"], how="left")
    fac2 = (I.C2 / I.bu_F.clip(lower=0.5)).fillna(1); fac3 = (I.C3 / I.bu_F.clip(lower=0.5)).fillna(1); facm = (I.Cm / I.bu_F.clip(lower=0.5)).fillna(1)
    for nm, v in [("c2_full", I.F * fac2), ("c2_p3", I.F * fac2 ** 0.3), ("c2_p5", I.F * fac2 ** 0.5), ("c3_p5", I.F * fac3 ** 0.5), ("mix", I.F * facm)]:
        out_items[f"{cell}:{nm}"] = pd.Series(v.values, index=pd.MultiIndex.from_frame(I[["origin", "style_color"]]))
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); Fb = B.set_index(["origin", "style_color"]).F.reindex(key).values; T["F"] = Fb; COLS = ["F"]
for k, v in out_items.items(): T[k] = v.reindex(key).fillna(pd.Series(Fb, index=key)).values; COLS.append(k)
REC_ = [o for o in ORIGINS if o >= "2025-06"]
for nm, m in [("nine recent origins", T.origin.isin(REC_)), ("all 13 origins", np.ones(len(T), bool))]:
    print(f"\n######## {nm}"); print(sl.table(T[m], COLS, D).round(4).to_string())
F_ = T[T.origin.isin(REC_)]; print("\n== bootstrap vs F, recent"); [print(f"  {c:18s}", np.round(sl.boot(F_, c, "F"), 4)) for c in COLS[1:]]
T[["origin", "style_color"] + COLS].to_parquet("data/cells_preds.parquet", index=False)
