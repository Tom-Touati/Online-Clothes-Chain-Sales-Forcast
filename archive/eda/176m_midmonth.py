"""D184 (overnight M5 goal, directions 5 and 8): mid-month origins as extra training rows for the D159 quantile model.
Rows from data/features_mid.parquet (eda/176m_features_mid.py: eda/06 on a half-month-shifted calendar, cutoffs on the 15th) are built
like the month-end rows (eda/20b T1: 4-shifted-month target, seasonal ratios averaged over the 4 target months). The D159 extras (cart
ramp, LLM scores, weather / event features, occasion timing) are taken from the preceding month-end origin (stale by half a month,
still past data only). A training row is used when its target is fully known: month-end rows o + 4 <= test origin, mid-month rows
o + 4.5 <= test origin. Testing on the month-end origins only, as always.
  mid       month-end + mid-month rows, equal weight
  midhalf   mid-month rows at weight 0.5
Writes data/q_backtest_mid_{mid,midhalf}.parquet; scored by eda/91_score_context.py mid data/q_backtest_wx_cpllm_we.parquet."""
import sys, numpy as np, pandas as pd, lightgbm as lgb, time
src = open("eda/164_q_weather.py", encoding="utf-8").read(); _a = sys.argv; sys.argv = sys.argv[:1]
exec(src[src.index("import sys"):src.index("for var in RUN:")]); sys.argv = _a
NEW = CP + LL + WE; qm.BASE = B0 + NEW; qm.FULL = F0 + NEW; COLS = list(dict.fromkeys(qm.FULL + qm.BASE))
Fm = pd.read_parquet("data/features_mid.parquet")
g = Fm.groupby(["origin", "style_color"])[["lfl_type", "lfl_macro", "lfl_occ", "own_sea"]].mean().add_suffix("4").reset_index()
Tm = Fm[Fm.h == 1].drop(columns=["y", "h"]).merge(g, on=["origin", "style_color"]); Tm["z"] = np.log((Tm.ytot + 1) / (4 * Tm.level + 1))
extra = T0[["origin", "style_color"] + [c for c in COLS if c not in Tm.columns]].rename(columns={"origin": "base_origin"})
Tm = Tm.merge(extra, on=["base_origin", "style_color"], how="left")
for c in COLS:
    if c in T0.columns and str(T0[c].dtype) == "category": Tm[c] = pd.Categorical(Tm[c], categories=T0[c].cat.categories)
T0["o_end"] = T0.o_ord + 4; T0["w"] = 1.0
print("month-end rows", len(T0), "| mid-month rows", len(Tm), "with target", int(Tm.ytot.notna().sum()), flush=True)
def fit_w(tr, te):
    out = np.zeros((len(te), len(qm.QS)))
    for cols in (qm.BASE, qm.FULL):
        for j, q in enumerate(qm.QS):
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, **qm.P).fit(tr[cols], tr.z, sample_weight=tr.w.values); out[:, j] += m.predict(te[cols]) / 2
    return np.sort(out, axis=1)
for var, wm in [(v, {"mid": 1.0, "midhalf": 0.5}[v]) for v in (sys.argv[1:] or ["mid", "midhalf"])]:
    ALL = pd.concat([T0, Tm.assign(w=wm)], ignore_index=True); out, t0 = [], time.time()
    for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
        oo = pd.Period(o).ordinal; tr = ALL[(ALL.o_end <= oo) & ALL.ytot.notna()]; te = T0[T0.origin == o]
        zq = fit_w(tr, te); R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
        for j, q in enumerate(qm.QS): R[f"q{int(q*100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
        out.append(R)
    pd.concat(out).to_parquet(f"data/q_backtest_mid_{var}.parquet"); print(var, "training rows at the last origin", len(tr), round(time.time() - t0), "s", flush=True)
