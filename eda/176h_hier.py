"""D185 (user: "go deeper into hierarchical reconciliation, anything hierarchical"). The first pass (D172) failed because the style model
was weaker than summing p2's items (style WAPE 0.292 vs 0.273). Here: a better style level, then several ways to combine levels.
Style-level features at each origin (data up to the origin): style units over its summed item level for each of the last 6 months,
style momentum (1 / 3 months), style trend, the style's own last-year ramp into the 4 target months (all colours), live colours,
colours launched in the last 2 months, style age, level-weighted means of the items' D159 features (type ramps, LLM, events,
temperature, cart ramp), product type. Bottom-up member totals (rules, D198 quantile, NN, TabPFN v2 and p2) as log ratios to 4 x L.
Style models (LightGBM median + L2, time-honest: origins o' + 4 <= o):
  S1  style features only, trained from 2024 origins
  S2  S1 + bottom-up member ratios (stacking at style level; member backtests exist from 2025-02, so it falls back to S1 until
      3 training origins exist)
  S3  residual model: target log((Y + 1) / (BU_p2 + 1)) on the S2 inputs; its prediction is applied as a style correction
Item-level combinations of p2 (each a version):
  s1_td / s2_td        top-down: items scaled to the style forecast (shares from p2)
  s2_mint              inverse-error-variance mix of BU and S2 in log space by size bucket (time-honest), then proportional
  s3_full / s3_half    p2 x exp(S3 correction) / x exp(0.5 x correction)
  share                learned within-style shares (LightGBM on the item's share of style units, last 3 months share,
                       its momentum vs the style, age, colour order, D159 item features) x the S2 style forecast
  mint4                MinT-WLS (variance scaling) over the whole hierarchy item -> style -> product type -> site, base forecasts
                       p2 (items), S2 (styles), a type model and a site model; weights = past mean squared errors per node level
Scored with the standard table (recent, recent from 2025-09, all 13, targets) and style-level WAPE; bootstrap against p2.
Writes data/hier_preds.parquet and eda/hier_output.txt (stdout)."""
import sys, contextlib, io, importlib.util, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
src = open("eda/164_q_weather.py", encoding="utf-8").read(); _a = sys.argv; sys.argv = sys.argv[:1]
exec(src[src.index("import sys"):src.index("for var in RUN:")]); sys.argv = _a
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
sty_of = pd.Series(M.index.str.split("_").str[0], index=M.index); MS = M.groupby(sty_of).sum()
first = M.gt(0).idxmax(axis=1).where(M.gt(0).any(axis=1)); Pm = lambda o, k: str(pd.Period(o, "M") + k)
T0["style"] = T0.style_color.str.split("_").str[0]
NUMF = [c for c in dict.fromkeys(CP + LL + WE + ["lfl_type4", "lfl_macro4", "own_sea4", "lvl"]) if c in T0.columns and T0[c].dtype.kind in "fiub"]
# ---- bottom-up member forecasts (backtest origins 2025-02 .. 2026-02)
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); G = lambda *xs: np.exp(np.mean([np.log(np.maximum(x, 0) + 0.5) for x in xs], axis=0)) - 0.5
qv = lambda f: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(key).q50.values
T["qd"] = qv("data/q_backtest_fkdyn_dyn.parquet"); T["t2"] = qv("data/q_backtest_fktab_tabpfn_v2.parquet")
T["p2"] = G(T.rules.values, T.qd.values, T.nn.values, T.t2.values); T["style"] = T.style_color.str.split("_").str[0]
MEM = ["rules", "qd", "nn", "t2", "p2"]
BU = T.groupby(["origin", "style"])[MEM].sum().add_prefix("bu_").reset_index()
# ---- style table
rows = []
for o in sorted(T0.origin.unique()):
    X = T0[T0.origin == o]; w = X.level.fillna(0) + 1
    A = X[NUMF].astype(float).fillna(X[NUMF].astype(float).mean()).fillna(0).mul(w, axis=0).groupby(X["style"]).sum().div(w.groupby(X["style"]).sum(), axis=0)
    A["L"] = X.groupby("style")["level"].sum(); A["Y"] = X.groupby("style").ytot.sum(min_count=1); A["n_items"] = X.groupby("style").size()
    A["ptype_c"] = X.groupby("style").ptype.agg(lambda v: v.astype(str).mode().iloc[0])
    S = A.index; L = A.L.values
    for k in range(6): A[f"s_lag{k+1}"] = np.log((MS.reindex(S)[Pm(o, -k)].values + 1) / (L + 1)) if Pm(o, -k) in MS.columns else np.nan
    l3 = MS.reindex(S)[[Pm(o, -k) for k in range(3) if Pm(o, -k) in MS.columns]].sum(1).values
    p3 = MS.reindex(S)[[Pm(o, -k) for k in range(3, 6) if Pm(o, -k) in MS.columns]].sum(1).values
    A["s_mom3"] = np.log((l3 + 1) / (p3 + 1))
    ly_t = [Pm(o, k - 12) for k in range(1, 5)]; ly_b = [Pm(o, -k - 12) for k in range(3)]
    if all(c in MS.columns for c in ly_t + ly_b):
        A["s_ramp"] = np.log((MS.reindex(S)[ly_t].sum(1).values + 1) / (MS.reindex(S)[ly_b].sum(1).values * 4 / 3 + 1))
    else: A["s_ramp"] = np.nan
    fs = first.groupby(sty_of).min().reindex(S)
    A["s_age"] = [(pd.Period(o, "M") - pd.Period(f, "M")).n if isinstance(f, str) else np.nan for f in fs]
    newc = first[(first > Pm(o, -2)) & (first <= o)].index; A["s_new2"] = pd.Series(sty_of[newc]).value_counts().reindex(S).fillna(0).values
    A["o_ord"] = pd.Period(o).ordinal; A["origin"] = o; rows.append(A.reset_index().rename(columns={"index": "style"}))
ST = pd.concat(rows, ignore_index=True); ST["ptype_c"] = ST.ptype_c.astype("category")
ST = ST.merge(BU, on=["origin", "style"], how="left")
for m in MEM: ST[f"r_{m}"] = np.log((ST[f"bu_{m}"] + 1) / (4 * ST.L + 1))
ST["z"] = np.log((ST.Y + 1) / (4 * ST.L + 1)); ST["zr"] = np.log((ST.Y + 1) / (ST.bu_p2 + 1))
X1 = NUMF + ["L", "n_items", "ptype_c"] + [f"s_lag{k}" for k in range(1, 7)] + ["s_mom3", "s_ramp", "s_age", "s_new2"]; X1 = [c if c != "L" else "L" for c in X1]
X2 = X1 + [f"r_{m}" for m in MEM]
P = dict(n_estimators=400, learning_rate=0.04, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, n_jobs=8)
fit = lambda tr, te, X, y: np.mean([lgb.LGBMRegressor(objective=ob, **({"alpha": 0.5} if ob == "quantile" else {}), **P).fit(tr[X], tr[y]).predict(te[X]) for ob in ("quantile", "l2")], axis=0)
out = []
for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
    oo = pd.Period(o).ordinal; tr = ST[(ST.o_ord + 4 <= oo) & ST.Y.notna()]; te = ST[ST.origin == o].copy()
    te["S1"] = (4 * te.L + 1) * np.exp(fit(tr, te, X1, "z")) - 1
    tr2 = tr[tr.bu_p2.notna()]
    if tr2.origin.nunique() >= 3:
        te["S2"] = (4 * te.L + 1) * np.exp(fit(tr2, te, X2, "z")) - 1; te["S3c"] = fit(tr2, te, X2, "zr")
    else: te["S2"] = te.S1; te["S3c"] = 0.0
    out.append(te[["origin", "style", "S1", "S2", "S3c", "bu_p2", "Y"]])
SF = pd.concat(out)
# ---- type and site models for MinT (from the style table, aggregated)
TT = ST.groupby(["origin", "ptype_c"], observed=True).agg(L=("L", "sum"), Y=("Y", lambda v: v.sum(min_count=1)), o_ord=("o_ord", "first"), s_mom3=("s_mom3", "mean"), s_ramp=("s_ramp", "mean"), lfl_type4=("lfl_type4", "mean"), bu_p2=("bu_p2", "sum")).reset_index()
TT["z"] = np.log((TT.Y + 1) / (4 * TT.L + 1)); TT["r_p2"] = np.log((TT.bu_p2 + 1) / (4 * TT.L + 1)); XT = ["s_mom3", "s_ramp", "lfl_type4", "r_p2", "ptype_c"]
outT = []
for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
    oo = pd.Period(o).ordinal; tr = TT[(TT.o_ord + 4 <= oo) & TT.Y.notna() & (TT.bu_p2 > 0)]; te = TT[TT.origin == o].copy()
    te["FT"] = (4 * te.L + 1) * np.exp(fit(tr, te, XT, "z")) - 1 if tr.origin.nunique() >= 3 else te.bu_p2
    outT.append(te[["origin", "ptype_c", "FT"]])
TF = pd.concat(outT)
# ---- item-level combinations
T = T.merge(SF, on=["origin", "style"], how="left"); B = T.groupby(["origin", "style"]).p2.transform("sum").clip(lower=0.5)
T["s1_td"] = T.p2 * (T.S1 / B).fillna(1); T["s2_td"] = T.p2 * (T.S2 / B).fillna(1)
T["s3_full"] = T.p2 * np.exp(T.S3c.fillna(0)); T["s3_half"] = T.p2 * np.exp(0.5 * T.S3c.fillna(0))
# inverse-variance BU / S2 mix (time-honest by size bucket)
S_ = SF.copy(); S_["eb"] = np.log((S_.Y + 1) / (S_.bu_p2 + 1)); S_["et"] = np.log((S_.Y + 1) / (S_.S2 + 1)); S_["bk"] = pd.qcut(np.log1p(S_.bu_p2), 4, labels=False, duplicates="drop"); S_["wt"] = 0.0
for o in S_.origin.unique():
    past = S_[pd.PeriodIndex(S_.origin, freq="M").astype(int) + 4 <= pd.Period(o).ordinal]
    for b in S_.bk.dropna().unique():
        pb = past[past.bk == b]; m = (S_.origin == o) & (S_.bk == b)
        S_.loc[m, "wt"] = (1 / pb.et.var()) / (1 / pb.et.var() + 1 / pb.eb.var()) if pb.origin.nunique() >= 3 else 0.0
S_["Fm"] = np.exp(S_.wt * np.log(S_.S2.clip(lower=0.5)) + (1 - S_.wt) * np.log(S_.bu_p2.clip(lower=0.5)))
T = T.merge(S_[["origin", "style", "Fm"]], on=["origin", "style"], how="left"); T["s2_mint"] = T.p2 * (T.Fm / B).fillna(1)
# learned within-style shares
T0s = T0[["origin", "style_color", "style", "level", "ytot", "lagr1", "lvl", "age_m", "color_num"] + [c for c in CP if c in T0.columns]].copy()
T0s["sty_L"] = T0s.groupby(["origin", "style"])["level"].transform("sum"); T0s["sh_lvl"] = T0s.level / T0s.sty_L.clip(lower=1e-9)
T0s["sh_y"] = T0s.ytot / T0s.groupby(["origin", "style"]).ytot.transform("sum").clip(lower=1e-9); T0s["n_sib"] = T0s.groupby(["origin", "style"])["level"].transform("size")
XS = ["sh_lvl", "lagr1", "lvl", "age_m", "color_num", "n_sib"] + [c for c in CP if c in T0s.columns]
T0s["o_ord"] = pd.PeriodIndex(T0s.origin, freq="M").astype(int); shares = []
for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
    oo = pd.Period(o).ordinal; tr = T0s[(T0s.o_ord + 4 <= oo) & T0s.sh_y.notna() & (T0s.n_sib > 1)]; te = T0s[T0s.origin == o].copy()
    te["sh_hat"] = lgb.LGBMRegressor(objective="l2", **P).fit(tr[XS], tr.sh_y).predict(te[XS]).clip(0.001)
    te["sh_hat"] = np.where(te.n_sib > 1, te.sh_hat / te.groupby("style").sh_hat.transform("sum"), 1.0); shares.append(te[["origin", "style_color", "sh_hat"]])
T = T.merge(pd.concat(shares), on=["origin", "style_color"], how="left")
T["share"] = (T.sh_hat * T.Fm).where(T.sh_hat.notna() & T.Fm.notna(), T.s2_mint)
# MinT-WLS over item / style / type / site (variance scaling by level, from past residual mean squares; time-honest)
T["ptype_c"] = it.product_type.reindex(T.style_color).astype(str).values if "it" in dir() else pd.read_parquet("data/items.parquet").set_index("style_color").product_type.reindex(T.style_color).astype(str).values
T = T.merge(TF.assign(ptype_c=TF.ptype_c.astype(str)), on=["origin", "ptype_c"], how="left")
T["mint4"] = T.p2.copy()
mse = {}
for o in sorted(T.origin.unique()):
    m = (T.origin == o).values; Ti = T[m]
    past = T[(pd.PeriodIndex(T.origin, freq="M").astype(int) + 4 <= pd.Period(o).ordinal)]
    if past.origin.nunique() < 3: continue
    ps = past.groupby(["origin", "style"]).agg(y=("y", "sum"), f=("Fm", "first")); pt = past.groupby(["origin", "ptype_c"]).agg(y=("y", "sum"), f=("FT", "first"))
    w_i = ((past.p2 - past.y) ** 2).mean(); w_s = ((ps.f - ps.y) ** 2).mean(); w_t = ((pt.f - pt.y) ** 2).mean()
    # closed form for a 3-level tree: reconcile each type subtree with diagonal W (WLS); site level omitted (its base is the sum)
    for t, Tt in Ti.groupby("ptype_c"):
        sty = Tt["style"].values; us = np.unique(sty); ni = len(Tt)
        Smat = np.vstack([np.eye(ni), np.array([[1.0 if s == u else 0.0 for s in sty] for u in us]), np.ones((1, ni))])
        yhat = np.concatenate([Tt.p2.values, Tt.groupby("style").Fm.first().reindex(us).values, [Tt.FT.iloc[0]]])
        ok = np.isfinite(yhat); Wd = np.concatenate([np.full(ni, w_i), np.full(len(us), w_s), [w_t]])[ok]
        Sm = Smat[ok]; Wi = 1 / Wd; A_ = Sm.T @ (Sm * Wi[:, None]); bvec = Sm.T @ (Wi * yhat[ok])
        T.loc[Tt.index, "mint4"] = np.clip(np.linalg.solve(A_ + 1e-9 * np.eye(ni), bvec), 0, None)
COLS = ["p2", "s1_td", "s2_td", "s2_mint", "s3_full", "s3_half", "share", "mint4"]
REC_ = [o for o in ORIGINS if o >= "2025-06"]; LATE = [o for o in ORIGINS if o >= "2025-09"]
for nm, m in [("nine recent origins", T.origin.isin(REC_)), ("recent from 2025-09 (style stacking has >= 3 origins)", T.origin.isin(LATE)), ("all 13 origins", np.ones(len(T), bool)), ("recent, 100 targets", T.origin.isin(REC_) & T.target)]:
    print(f"\n######## {nm}"); print(sl.table(T[m].reset_index(drop=True), COLS, {k: v[m.values if hasattr(m, 'values') else m] for k, v in D.items()}).round(4).to_string())
F_ = SF[SF.origin.isin(REC_)]; yy = F_.Y.sum()
print("\nstyle-level WAPE, recent: BU p2 %.4f | S1 %.4f | S2 %.4f | BU x S3 %.4f | mint mix %.4f" % ((F_.bu_p2 - F_.Y).abs().sum() / yy, (F_.S1 - F_.Y).abs().sum() / yy,
      (F_.S2 - F_.Y).abs().sum() / yy, (F_.bu_p2 * np.exp(F_.S3c) - F_.Y).abs().sum() / yy, (S_[S_.origin.isin(REC_)].Fm - S_[S_.origin.isin(REC_)].Y).abs().sum() / yy))
F_ = T[T.origin.isin(REC_)].reset_index(drop=True); print("\n== bootstrap vs p2, recent"); [print(f"  {c:10s}", np.round(sl.boot(F_, c, "p2"), 4)) for c in COLS[1:]]
T[["origin", "style_color"] + COLS].to_parquet("data/hier_preds.parquet", index=False)
