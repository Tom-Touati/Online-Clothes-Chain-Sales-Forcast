"""D247 (user: "what are the items we ain't good at, why and when"; D244: the largest misses are young launches that break out into the
autumn / holiday peak, all members under): can breakouts be spotted at the origin?
Base: final system F = p2c32a (data/dec_base.parquet), recent origins for diagnostics, all origins for time-honest fits.
Breakout = log((y + 1) / (F + 1)) > 0.5 and y >= 20.
Signals known at the origin (no future data):
  item curve    age_m, lagr1..6, f_r7 / f_r28, trend91, peak_r, sh_ret, sh_drop, disc_item, disc_rel, sh_tt
  weekly        last 4 complete weeks vs the 4 before (wk_mom), last week vs the mean of the last 8 (wk_last), weeks live, the best
                week over the mean (wk_peak), share of the item's life sales in the last 4 weeks (wk_recent)
  look peers    level-weighted mean trend91 of the other items in the item's look (pe_trend), the look's units growth over the last
                3 months vs the 3 before (pe_mom), number of look items launched in the last 3 months (pe_launch), the look's mean
                past log residual of F at origins with known outcomes o' + 4 <= o (pe_resid)
  window        bf_window, number of target months in Oct-Dec (n_q4)
1 Who breaks out: rate and error share of breakouts by age band x window.
2 AUC of each signal for breakout among young items (< 6 months) and among all items (y >= 20, recent).
3 Launch-stage x window bias: time-honest segment correction (as D245, shrunk n / (n + 2), strength 0.5) on age x window,
  age x window x volume, total-preserving and not.
4 Time-honest LightGBM (median + L2) on F's log residual with all signals, trained on earlier origins (o' + 4 <= o), applied as
  F x exp(lambda x prediction) to young items only and to all items; lambda 0.5 / 1, and a total-preserving variant.
Standard table + bootstrap against F. Writes eda/breakout_output.txt (stdout), data/breakout_preds.parquet."""
import sys, contextlib, io, importlib.util, warnings, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
import os; SUF = os.environ.get("DEC", ""); B = pd.read_parquet(f"data/dec_base{SUF}.parquet")   # D257: DEC=_<tag> refits on another base system
Fe = pd.read_parquet("data/features.parquet").drop_duplicates(["origin", "style_color"])
FC = ["lagr1", "lagr2", "lagr3", "lagr4", "lagr5", "lagr6", "f_r7", "peak_r", "sh_ret", "sh_drop", "disc_rel"]
B = B.merge(Fe[["origin", "style_color"] + FC], on=["origin", "style_color"], how="left")
B["acc7"] = np.log((B.f_r7 + 1e-3) / (B.f_r28 + 1e-3))
# weekly signals
W = pd.read_parquet("data/weekly_item.parquet"); W["ds"] = pd.to_datetime(W.ds)
Wp = W.pivot_table(index="unique_id", columns="ds", values="y", aggfunc="sum").fillna(0); wk = Wp.columns
rows = []
for o in sorted(B.origin.unique()):
    end = pd.Period(o, "M").end_time.normalize(); cols = [d for d in wk if d + pd.Timedelta(days=6) <= end]
    X = Wp[cols].values; ids = Wp.index
    l4 = X[:, -4:].sum(1); p4 = X[:, -8:-4].sum(1); m8 = X[:, -8:].mean(1); live = (X > 0).any(1)
    first = np.where(live, np.argmax(X > 0, 1), X.shape[1]); wl = X.shape[1] - first
    rows.append(pd.DataFrame({"origin": o, "style_color": ids, "wk_mom": np.log((l4 + 1) / (p4 + 1)), "wk_last": np.log((X[:, -1] + 1) / (m8 + 1)),
                              "wk_live": wl, "wk_peak": np.log((X.max(1) + 1) / (X.sum(1) / np.maximum(wl, 1) + 1)), "wk_recent": l4 / np.maximum(X.sum(1), 1)}))
B = B.merge(pd.concat(rows), on=["origin", "style_color"], how="left")
# look-peer signals
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
first_m = M.gt(0).idxmax(axis=1).where(M.gt(0).any(axis=1)); oo = pd.PeriodIndex(B.origin, freq="M").astype(int).values
B["pe_trend"] = np.nan; B["pe_mom"] = np.nan; B["pe_launch"] = np.nan; B["pe_resid"] = np.nan
for o in sorted(B.origin.unique()):
    m = (B.origin == o).values; D = B[m]; P = pd.Period(o, "M"); c3 = [str(P - k) for k in range(3)]; p3 = [str(P - k) for k in range(3, 6)]
    past = B[(oo + 4 <= P.ordinal)]
    pr = past.groupby("style_color").e.mean()   # the item's own past residuals are not used; peers' are, by look at origin o
    for lk, g in D.groupby("look"):
        ix = g.index; w = g.level.fillna(0) + 1; tr = (g.trend91.fillna(0) * w)
        B.loc[ix, "pe_trend"] = ((tr.sum() - tr) / (w.sum() - w).clip(lower=1e-9)).values if len(g) > 1 else np.nan
        mem = [s for s in g.style_color if s in M.index]
        B.loc[ix, "pe_mom"] = np.log((M.loc[mem, [c for c in c3 if c in M.columns]].sum().sum() + 1) / (M.loc[mem, [c for c in p3 if c in M.columns]].sum().sum() + 1))
        B.loc[ix, "pe_launch"] = sum(1 for s in mem if isinstance(first_m.get(s), str) and (P - pd.Period(first_m[s], "M")).n < 3)
        rr = pr.reindex(g.style_color); B.loc[ix, "pe_resid"] = [rr.drop(s).mean() if len(rr.dropna()) > 1 else np.nan for s in g.style_color]
B["n_q4"] = [sum(1 for h in range(1, 5) if (pd.Period(o, "M") + h).month in (10, 11, 12)) for o in B.origin]
B["brk"] = ((B.e > 0.5) & (B.y >= 20)).astype(int)
B["age_b"] = pd.cut(B.age_m, [-1, 3, 6, 12, 999], labels=["<3m", "3-6m", "6-12m", "12m+"])
SIG = ["age_m", "lagr1", "lagr2", "lagr3", "acc7", "trend91", "peak_r", "sh_ret", "sh_drop", "disc_item", "disc_rel", "sh_tt", "own_sea",
       "wk_mom", "wk_last", "wk_live", "wk_peak", "wk_recent", "pe_trend", "pe_mom", "pe_launch", "pe_resid", "spread", "n_q4"]
XS = os.environ.get("XSIG") == "1"   # D259: + falling-item signals (promotion sensitivity, recent spike, time since peak, launch curve, peak anniversary)
if XS:
    _x = pd.read_parquet("data/nf2_feats.parquet")[["origin", "style_color", "pk_28", "pk_365_days", "pk_anniv_dist", "curve_fwd", "curve_dev"]].merge(pd.read_parquet("data/nf3_feats.parquet")[["origin", "style_color", "promo_lift"]], on=["origin", "style_color"])
    B = B.merge(_x, on=["origin", "style_color"], how="left"); SIG = SIG + ["pk_28", "pk_365_days", "pk_anniv_dist", "curve_fwd", "curve_dev", "promo_lift"]; SUF = SUF + "_x"
    R = B[B.origin >= "2025-06"]
R = B[B.origin >= "2025-06"]; E = R.ae.sum()
print("== 1 breakouts (e > 0.5, y >= 20), recent origins: rate, share of all error, F bias, by age band x Q4 months in the window")
t = R.groupby(["age_b", "n_q4"], observed=True).apply(lambda d: pd.Series({"n": len(d), "brk_rate": d.brk.mean(), "brk_err_share": d.ae[d.brk == 1].sum() / E,
                                                                         "err_share": d.ae.sum() / E, "bias": d.F.sum() / d.y.sum() - 1}))
print(t.round(3).to_string())
print("\n== 2 AUC for breakout (recent, y >= 20 or F >= 20); > 0.5 = higher value, more breakouts")
for nm, d in [("young < 6 months", R[(R.age_m < 6) & ((R.y >= 20) | (R.F >= 20))]), ("all items", R[(R.y >= 20) | (R.F >= 20)])]:
    a = {c: roc_auc_score(d.brk, d[c].fillna(d[c].median())) for c in SIG if d[c].notna().any()}
    print(f"-- {nm}: {len(d)} item-origins, breakout rate {d.brk.mean():.3f}")
    print("  " + " | ".join(f"{k} {v:.2f}" for k, v in sorted(a.items(), key=lambda kv: -abs(kv[1] - 0.5))))
# 3 segment correction with the window
B["vol"] = B.groupby("origin").R3.transform(lambda v: pd.qcut(np.log1p(v), 5, labels=False, duplicates="drop"))
B["ab"] = B.age_b.cat.codes
SEGS = {"age_q4": ["ab", "n_q4"], "age_q4_vol": ["ab", "n_q4", "vol"], "age_bf": ["ab", "bf_window"]}
COLS = ["F"]
for nm, cols in SEGS.items():
    col = f"seg_{nm}"; B[col] = B.F
    for o in sorted(B.origin.unique()):
        tr = oo + 4 <= pd.Period(o).ordinal
        if B.origin[tr].nunique() < 2: continue
        g = B[tr].groupby(cols).apply(lambda d: pd.Series({"r": np.log((d.y.sum() + 1) / (d.F.sum() + 1)), "n": d.origin.nunique()}))
        g["r"] = g.r * g.n / (g.n + 2); m = (B.origin == o).values
        B.loc[m, col] = B.F[m] * np.exp(0.5 * B[m][cols].merge(g.reset_index(), on=cols, how="left").r.fillna(0).values)
    B[col + "n"] = B[col] * B.groupby("origin").F.transform("sum") / B.groupby("origin")[col].transform("sum"); COLS += [col, col + "n"]
# 4 time-honest residual model
P = dict(n_estimators=int(os.environ.get("BRK_ROUNDS", 300)), learning_rate=0.03, num_leaves=int(os.environ.get("BRK_LEAVES", 15)), min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10, verbose=-1, n_jobs=int(os.environ.get("NJOBS", 8)))
B["rhat"] = np.nan
for o in sorted(B.origin.unique()):
    tr = (oo + 4 <= pd.Period(o).ordinal); m = (B.origin == o).values
    if B.origin[tr].nunique() < 3: continue
    Xt = B.loc[tr, SIG].astype(float); yt = B.e[tr].clip(-2, 2); wt = np.sqrt(B.y[tr] + B.F[tr] + 1)
    B.loc[m, "rhat"] = np.mean([lgb.LGBMRegressor(objective=ob, **({"alpha": 0.5} if ob == "quantile" else {}), **P).fit(Xt, yt, sample_weight=wt).predict(B.loc[m, SIG].astype(float)) for ob in ("quantile", "l2")], axis=0)
B["rhat"] = B.rhat.fillna(0)
for lam in (0.5, 1.0):
    B[f"res_all_{lam}"] = B.F * np.exp(lam * B.rhat); B[f"res_young_{lam}"] = np.where(B.age_m < 6, B.F * np.exp(lam * B.rhat), B.F)
    B[f"res_all_{lam}n"] = B[f"res_all_{lam}"] * B.groupby("origin").F.transform("sum") / B.groupby("origin")[f"res_all_{lam}"].transform("sum")
    COLS += [f"res_all_{lam}", f"res_young_{lam}", f"res_all_{lam}n"]
d = B[(B.origin >= "2025-06") & ((B.y >= 20) | (B.F >= 20))]
print(f"\n== 4 residual model: Spearman(rhat, e) recent {d[['rhat', 'e']].corr(method='spearman').iloc[0, 1]:+.3f} | young {d[d.age_m < 6][['rhat', 'e']].corr(method='spearman').iloc[0, 1]:+.3f} | AUC breakout young {roc_auc_score(d[d.age_m < 6].brk, d[d.age_m < 6].rhat):.3f}")
if os.environ.get("NOSCORE") == "1": B[["origin", "style_color", "brk", "rhat"] + SIG + COLS].to_parquet("data/breakout_preds" + SUF + os.environ.get("OUTTAG", "") + ".parquet", index=False); print("saved without scoring (NOSCORE)"); sys.exit()   # D352: server runs
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); Bi = B.set_index(["origin", "style_color"]).reindex(key)
for c in COLS: T[c] = Bi[c].values
REC_ = [o for o in ORIGINS if o >= "2025-06"]
for nm, m in [("nine recent origins", T.origin.isin(REC_)), ("all 13 origins", np.ones(len(T), bool))]:
    print(f"\n######## {nm}"); print(sl.table(T[m], COLS, D).round(4).to_string())
F_ = T[T.origin.isin(REC_)]; print("\n== bootstrap vs F, recent"); [print(f"  {c:18s}", np.round(sl.boot(F_, c, "F"), 4)) for c in COLS[1:]]
B[["origin", "style_color", "brk", "rhat"] + SIG + COLS].to_parquet("data/breakout_preds" + SUF + os.environ.get("OUTTAG", "") + ".parquet", index=False)
