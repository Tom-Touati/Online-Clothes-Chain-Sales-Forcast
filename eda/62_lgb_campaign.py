"""Known-future campaign covariates in the LightGBM ensemble, same 13 rolling origins.
Site calendar per month from edikted.com-type ONLINE lines: deep-discount days (unit-weighted daily discount >= 58.5%, the
D016/D030 threshold) and the month's unit-weighted discount. Features: value in the target month and change versus the
3-month level window. 'deep' = deep-day features only (a calendar the business sets in advance); 'all' adds discount depth
(realised, so partly an outcome; reported as an upper bound). Writes data/camp_backtest_preds.parquet."""
import sys, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
F = pd.read_parquet("data/features.parquet")
for c in ["ptype", "macro"] + [c for c in F.columns if c.startswith("a_")]: F[c] = F[c].astype("category")
F["rlog"] = np.log((F.y + 1) / (F.level + 1))
OCC = [c for c in F.columns if c.startswith("occ_")]
BASE = [f"lagr{k}" for k in range(1, 7)] + ["lvl", "age_m", "ptype", "tmonth", "h", "color_num", "own_sea"] + OCC
FULL = BASE + ["f_r7", "f_r28", "trend91", "peak_r", "days_live", "sh_ret", "sh_drop", "sh_tt", "disc_item", "disc_rel", "lfl_type", "lfl_macro", "lfl_occ"] + [c for c in F.columns if c.startswith("a_")]
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=6)
def l1(tr, te, cols):
    m = lgb.LGBMRegressor(objective="l1", **P).fit(tr[cols], tr.rlog); return ((te.level + 1) * np.exp(m.predict(te[cols])) - 1).clip(lower=0)
d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[(~d.cancelled) & (d["class"] == "ONLINE")]; d["ny_date"] = pd.to_datetime(d.ny_date)
s = d.groupby("ny_date").agg(q=("qty", "sum"), dw=("disc_w", "sum")); s["disc"] = s.dw / s.q; del d
mon = pd.DataFrame({"disc": s.dw.resample("MS").sum() / s.q.resample("MS").sum(), "deep": s.disc.ge(0.585).resample("MS").sum()}); mon.index = mon.index.to_period("M")
op = pd.PeriodIndex(F.origin, freq="M"); tm = op + F.h.values
F["c_deep_t"] = mon.deep.reindex(tm).values; F["c_disc_t"] = mon.disc.reindex(tm).values
lw = lambda c: np.mean([mon[c].reindex(op - k).values for k in range(3)], axis=0)
F["c_deep_d"] = F.c_deep_t - lw("deep"); F["c_disc_d"] = F.c_disc_t - lw("disc")
F["c_deep_ly"] = mon.deep.reindex(tm - 12).values; F["c_deep_vly"] = F.c_deep_t - F.c_deep_ly     # versus the same month last year
DEEP = ["c_deep_t", "c_deep_d", "c_deep_vly"]; ALL = DEEP + ["c_disc_t", "c_disc_d"]
ORIGINS = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
out = []
for o in ORIGINS:
    oo = pd.Period(o).ordinal; tr = F[(F.o_ord + F.h <= oo) & F.y.notna()]; te = F[F.origin == o].copy()
    pb, pf = l1(tr, te, BASE), l1(tr, te, FULL)
    te["ens"] = (pb + pf) / 2
    te["ens_deep"] = (l1(tr, te, BASE + DEEP) + l1(tr, te, FULL + DEEP)) / 2
    te["ens_all"] = (l1(tr, te, BASE + ALL) + l1(tr, te, FULL + ALL)) / 2
    out.append(te[["origin", "style_color", "h", "ens", "ens_deep", "ens_all", "y"]])
    g = te.groupby("style_color")[["ens", "ens_deep", "ens_all", "y"]].sum()
    print(o, {c: round((g[c] - g.y).abs().sum() / g.y.sum(), 3) for c in ["ens", "ens_deep", "ens_all"]}, flush=True)
X = pd.concat(out); X.to_parquet("data/camp_backtest_preds.parquet")
T = set(pd.read_csv("data/targets.csv").style_color)
for seg, m in [("all", np.ones(len(X), bool)), ("targets", X.style_color.isin(T).values)]:
    R = X[m].groupby("origin").apply(lambda g: pd.Series({c: (lambda t: (t[c] - t.y).abs().sum() / t.y.sum())(g.groupby("style_color")[[c, "y"]].sum()) for c in ["ens", "ens_deep", "ens_all"]}))
    print(f"== {seg}: mean WAPE", R.mean().round(4).to_dict(), "| deep beats ens at", int((R.ens_deep < R.ens).sum()), "| all beats ens at", int((R.ens_all < R.ens).sum()), "of", len(R))
X["tmp"] = (pd.PeriodIndex(X.origin, freq="M") + X.h).astype(str)
A = X.groupby(["tmp", "h"])[["ens", "ens_deep", "ens_all", "y"]].sum()
print("month-level aggregate |log error| by h:"); print(np.log(A[["ens", "ens_deep", "ens_all"]].div(A.y, axis=0)).abs().groupby(level="h").mean().round(3).to_string())
