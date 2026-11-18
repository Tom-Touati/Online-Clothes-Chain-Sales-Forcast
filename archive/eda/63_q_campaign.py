"""Production quantile model (eda/20_quantiles.py, 4-month totals) with known-future campaign covariates, 13 rolling origins.
Covariates over the 4 target months: deep-discount days (sum), change vs 4/3 x the 3-month level window, change vs the same
4 months a year earlier; variant 'all' adds the unit-weighted discount (mean over the 4 months, and change vs level window).
Compared paired by origin with the stored baseline data/q_backtest.parquet on P50 WAPE / bias / within-20% (all, targets) and
on the representative origins 2025-06..2026-02 used for tau* (D050). Writes data/q_backtest_camp{_all}.parquet."""
import sys, numpy as np, pandas as pd, lightgbm as lgb, warnings, importlib.util, time
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
VAR = sys.argv[1] if len(sys.argv) > 1 else "deep"
spec = importlib.util.spec_from_file_location("q20", "eda/20_quantiles.py"); q20 = importlib.util.module_from_spec(spec); spec.loader.exec_module(q20)
T1 = q20.T1.copy(); q20.P["n_jobs"] = 6
d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[(~d.cancelled) & (d["class"] == "ONLINE")]; d["ny_date"] = pd.to_datetime(d.ny_date)
s = d.groupby("ny_date").agg(q=("qty", "sum"), dw=("disc_w", "sum")); s["disc"] = s.dw / s.q; del d
mon = pd.DataFrame({"disc": s.dw.resample("MS").sum() / s.q.resample("MS").sum(), "deep": s.disc.ge(0.585).resample("MS").sum()}); mon.index = mon.index.to_period("M")
op = pd.PeriodIndex(T1.origin, freq="M")
fut = lambda c, sh=0: np.sum([mon[c].reindex(op + h - sh).values for h in range(1, 5)], axis=0)
lev = lambda c: np.sum([mon[c].reindex(op - k).values for k in range(3)], axis=0)
T1["c_deep4"] = fut("deep"); T1["c_deep4_d"] = T1.c_deep4 - lev("deep") * 4 / 3; T1["c_deep4_vly"] = T1.c_deep4 - fut("deep", 12)
T1["c_disc4"] = fut("disc") / 4; T1["c_disc4_d"] = T1.c_disc4 - lev("disc") / 3
C = ["c_deep4", "c_deep4_d", "c_deep4_vly"] + (["c_disc4", "c_disc4_d"] if VAR == "all" else [])
q20.BASE = q20.BASE + C; q20.FULL = q20.FULL + C
if __name__ == "__main__":
    rows, t0 = [], time.time()
    for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
        oo = pd.Period(o).ordinal; tr = T1[(T1.o_ord + 4 <= oo) & T1.ytot.notna()]; te = T1[T1.origin == o]
        zq = q20.fit_q(tr, te); R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
        for j, q in enumerate(q20.QS): R[f"q{int(q*100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
        rows.append(R); print(o, round(time.time() - t0), "s", flush=True)
    Q = pd.concat(rows); Q.to_parquet(f"data/q_backtest_camp{'_all' if VAR == 'all' else ''}.parquet")
    B = pd.read_parquet("data/q_backtest.parquet")[["origin", "style_color", "q50"]].rename(columns={"q50": "base"})
    X = Q.merge(B, on=["origin", "style_color"]); X["camp"] = X.q50
    Tg = set(pd.read_csv("data/targets.csv").style_color)
    def met(g, c): ape = (g[c] - g.ytot).abs() / g.ytot.clip(lower=1); return pd.Series({"wape": (g[c] - g.ytot).abs().sum() / g.ytot.sum(), "bias": g[c].sum() / g.ytot.sum() - 1, "w20": (ape <= .2).mean()})
    for seg, m in [("all", np.ones(len(X), bool)), ("targets", X.style_color.isin(Tg).values)]:
        R = pd.concat({c: X[m].groupby("origin").apply(lambda g: met(g, c)) for c in ["base", "camp"]}, axis=1)
        print(f"\n== {seg}"); print(R.xs("wape", axis=1, level=1).round(3).T.to_string())
        print("mean:", R.mean().round(4).to_dict()); print("camp better WAPE at", int((R[("camp", "wape")] < R[("base", "wape")]).sum()), "of", len(R))
        rep = R.index >= "2025-06"; print("representative origins mean WAPE base/camp:", round(R[rep][("base", "wape")].mean(), 4), round(R[rep][("camp", "wape")].mean(), 4))
