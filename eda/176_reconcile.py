"""D172 (overnight M5 goal, direction 1): hierarchical reconciliation item / style / type.
Base item forecast = p2 (D169: equal geometric mean of rules, the D198 quantile model, the NN and TabPFN v2), and the D159 three-way.
Upper-level models: LightGBM on style (and product-type) 4-month totals, target log((Y + 1) / (4 x L + 1)) with L = sum of item levels;
features = the item rows of the D159 quantile model (eda/164 T0) aggregated: level-weighted means of numeric features, sum of level,
number of items, mode of product type; median (quantile 0.5) and L2 objectives averaged. Time-honest: trained on origins o' + 4 <= o.
Versions (applied to each base):
  td        top-down: item forecasts scaled so each style sums to the style model's forecast
  part_a    partial: scaling factor ^ a, a in 0.3 / 0.5 / 0.7
  mint      log-space combination of the bottom-up style sum and the style model, weights = inverse error variance of each on the
            style's size bucket over past origins (time-honest; equal until 3 origins exist), then distributed proportionally
  td3       three levels: type model -> styles (mint within type) -> items
Scored with the standard R3-relative table (eda/score_lib) on recent origins, all 13 and the targets; style-level WAPE printed too.
Writes data/reconcile_preds.parquet and eda/reconcile_output.txt (stdout)."""
import sys, contextlib, io, importlib.util, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
src = open("eda/164_q_weather.py", encoding="utf-8").read(); _a = sys.argv; sys.argv = sys.argv[:1]
exec(src[src.index("import sys"):src.index("for var in RUN:")]); sys.argv = _a
FEATS = [c for c in dict.fromkeys(F0 + CP + LL + WE) if c in T0.columns]
T0["style"] = T0.style_color.str.split("_").str[0]
num = [c for c in FEATS if c not in ("ptype",) and T0[c].dtype.kind in "fiub"]
def agg(df, key):
    w = df.level.fillna(0).values + 1; X = df[num].astype(float); X = X.fillna(X.mean()).fillna(0)
    g = pd.DataFrame(X.values * w[:, None], columns=num); g["origin"] = df.origin.values; g[key] = df[key].values; g["_w"] = w
    s_ = g.groupby(["origin", key]).sum(); out = s_[num].div(s_["_w"], axis=0)
    e = df.groupby(["origin", key]).agg(L=("level", "sum"), Y=("ytot", lambda v: v.sum(min_count=1)), n_items=("level", "size"), o_ord=("o_ord", "first"),
                                         ptype_c=("ptype", lambda v: v.astype(str).mode().iloc[0]))
    out = out.join(e).reset_index(); out["ptype_c"] = out.ptype_c.astype("category"); return out
print("aggregating features...", flush=True)
SF = agg(T0, "style"); TF = agg(T0.assign(ptype_s=T0.ptype.astype(str)), "ptype_s")
P = dict(n_estimators=400, learning_rate=0.04, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, n_jobs=8)
def upper(DF, key):
    X = [c for c in num if c in DF.columns] + ["n_items", "ptype_c"]; DF = DF.copy(); DF["z"] = np.log((DF.Y + 1) / (4 * DF.L + 1)); out = []
    for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
        oo = pd.Period(o).ordinal; tr = DF[(DF.o_ord + 4 <= oo) & DF.Y.notna()]; te = DF[DF.origin == o]
        z = np.mean([lgb.LGBMRegressor(objective=ob, **({"alpha": 0.5} if ob == "quantile" else {}), **P).fit(tr[X], tr.z).predict(te[X]) for ob in ("quantile", "l2")], axis=0)
        out.append(pd.DataFrame({"origin": o, key: te[key].values, "F": (4 * te.L.values + 1) * np.exp(z) - 1}))
    return pd.concat(out)
print("style model...", flush=True); SFc = upper(SF, "style"); print("type model...", flush=True); TFc = upper(TF, "ptype_s")
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); G = lambda *xs: np.exp(np.mean([np.log(np.maximum(x, 0) + 0.5) for x in xs], axis=0)) - 0.5
qv = lambda f: pd.read_parquet(f).set_index(["origin", "style_color"]).reindex(key).q50.values
T["d159"] = G(T.rules.values, qv("data/q_backtest_wx_cpllm_we.parquet"), T.nn.values)
T["p2"] = G(T.rules.values, qv("data/q_backtest_fkdyn_dyn.parquet"), T.nn.values, qv("data/q_backtest_fktab_tabpfn_v2.parquet"))
T["style"] = T.style_color.str.split("_").str[0]; T["ptype_s"] = T0.drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).ptype.astype(str).reindex(key).values
T = T.merge(SFc, on=["origin", "style"], how="left").merge(TFc.rename(columns={"F": "FT"}), on=["origin", "ptype_s"], how="left")
for base in ["d159", "p2"]:
    B = T.groupby(["origin", "style"])[base].transform("sum"); fac = (T.F / B.clip(lower=0.5)).where(T.F.notna(), 1.0)
    T[f"{base}_td"] = T[base] * fac
    for a in (0.3, 0.5, 0.7): T[f"{base}_part{int(a * 10)}"] = T[base] * fac ** a
    # mint: inverse-variance weights from past origins, by style-size bucket (time-honest)
    S = T.groupby(["origin", "style"]).agg(B=(base, "sum"), F=("F", "first"), Y=("y", "sum")).reset_index()
    S["eb"] = np.log((S.Y + 1) / (S.B + 1)); S["et"] = np.log((S.Y + 1) / (S.F + 1)); S["bk"] = pd.qcut(np.log1p(S.B), 4, labels=False, duplicates="drop")
    S["wt"] = np.nan
    for o in sorted(S.origin.unique()):
        past = S[pd.PeriodIndex(S.origin, freq="M").astype(int) + 4 <= pd.Period(o).ordinal]
        for b in S.bk.unique():
            pb = past[past.bk == b]; m = (S.origin == o) & (S.bk == b)
            S.loc[m, "wt"] = (1 / pb.et.var()) / (1 / pb.et.var() + 1 / pb.eb.var()) if pb.origin.nunique() >= 3 else 0.5
    S["Fm"] = np.exp(S.wt * np.log(S.F.clip(lower=0.5)) + (1 - S.wt) * np.log(S.B.clip(lower=0.5)))
    T = T.merge(S[["origin", "style", "Fm", "wt"]].rename(columns={"Fm": f"Fm_{base}", "wt": f"wt_{base}"}), on=["origin", "style"], how="left")
    T[f"{base}_mint"] = T[base] * (T[f"Fm_{base}"] / B.clip(lower=0.5)).fillna(1.0)
    # three levels: type model rescales the mint style totals within each type (geometric half-way), then items follow
    St = T.groupby(["origin", "ptype_s"])[f"{base}_mint"].transform("sum"); ft = (T.FT / St.clip(lower=0.5)).where(T.FT.notna(), 1.0)
    T[f"{base}_td3"] = T[f"{base}_mint"] * ft ** 0.5
COLS = [c for c in T.columns if c.startswith(("d159", "p2")) and not c.startswith(("Fm_", "wt_"))]
REC_ = [o for o in ORIGINS if o >= "2025-06"]
for nm, m in [("nine recent origins", T.origin.isin(REC_)), ("all 13 origins", np.ones(len(T), bool)), ("recent, 100 targets", T.origin.isin(REC_) & T.target)]:
    F_ = T[m]; print(f"\n######## {nm}"); print(sl.table(F_, COLS, D).round(4).to_string())
F_ = T[T.origin.isin(REC_)]
print("\nstyle-level WAPE (recent): bottom-up p2", round((F_.groupby(["origin", "style"]).p2.sum() - F_.groupby(["origin", "style"]).y.sum()).abs().sum() / F_.y.sum(), 4),
      "| style model", round((F_.groupby(["origin", "style"]).F.first() - F_.groupby(["origin", "style"]).y.sum()).abs().sum() / F_.y.sum(), 4))
print("\n== bootstrap vs base, recent:"); [print(f"  {c:12s}", np.round(sl.boot(F_, c, c.split("_")[0]), 4)) for c in COLS if "_" in c]
T[["origin", "style_color"] + COLS].to_parquet("data/reconcile_preds.parquet", index=False)
