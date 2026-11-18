"""Can we predict which items will be forecast well? Uses out-of-sample predictions of the best backtested model
(average of two median-loss LightGBMs: base and full feature sets) over 13 origins.
1) error by segment; 2) an 'error model' trained only on folds whose actuals were known at the time, then scored on later folds."""
import pandas as pd, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
F0 = pd.read_parquet("data/features.parquet")
for c in ["ptype", "macro"] + [c for c in F0.columns if c.startswith("a_")]: F0[c] = F0[c].astype("category")
OCC = [c for c in F0.columns if c.startswith("occ_")]
BASE = [f"lagr{k}" for k in range(1, 7)] + ["lvl", "age_m", "ptype", "tmonth", "h", "color_num", "own_sea"] + OCC
FULL = BASE + ["f_r7", "f_r28", "trend91", "peak_r", "days_live", "sh_ret", "sh_drop", "sh_tt", "disc_item", "disc_rel", "lfl_type", "lfl_macro", "lfl_occ"] + [c for c in F0.columns if c.startswith("a_")]
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=4)
def l1(tr, te, cols):
    m = lgb.LGBMRegressor(objective="l1", **P).fit(tr[cols], np.log((tr.y + 1) / (tr.level + 1))); return ((te.level + 1) * np.exp(m.predict(te[cols])) - 1).clip(lower=0)
import os
parts = []
for o in ([] if os.path.exists("data/best_backtest_preds.parquet") else [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]):
    oo = pd.Period(o).ordinal; tr = F0[(F0.o_ord + F0.h <= oo) & F0.y.notna()]; te = F0[F0.origin == o]
    parts.append(pd.DataFrame({"origin": o, "style_color": te.style_color, "h": te.h, "pred": (l1(tr, te, BASE) + l1(tr, te, FULL)) / 2, "y": te.y}))
if parts: pd.concat(parts).to_parquet("data/best_backtest_preds.parquet")
BT = pd.read_parquet("data/best_backtest_preds.parquet")
E = BT.groupby(["origin", "style_color"], as_index=False)[["pred", "y"]].sum()
F = F0[F0.h == 1].drop(columns=["y", "h"])
E = E.merge(F, on=["origin", "style_color"])
E["lerr"] = np.log((E.pred + 1) / (E.y + 1)); E["ale"] = E.lerr.abs(); E["ape"] = (E.pred - E.y).abs() / E.y.clip(lower=1)
E["w20"] = E.ape <= .2; E["w50"] = E.ape <= .5
E["tagged"] = E[[c for c in E.columns if c.startswith("occ_")]].sum(axis=1) > 0
E["omonth"] = pd.PeriodIndex(E.origin, freq="M").month
E["vol"] = E[[f"lagr{k}" for k in range(1, 7)]].std(axis=1)          # how bumpy the last 6 months were
E["n_lags"] = E[[f"lagr{k}" for k in range(1, 7)]].notna().sum(axis=1)
tab = lambda by: E.groupby(by, observed=True).agg(n=("w20", "size"), within20=("w20", "mean"), within50=("w50", "mean"), med_ape=("ape", "median"), bias=("lerr", lambda s: np.expm1(s.median()))).round(3)
print("overall:", dict(n=len(E), within20=round(E.w20.mean(), 3), within50=round(E.w50.mean(), 3), off_by_more_than_50=round(1 - E.w50.mean(), 3)))
print("\n-- by age at origin (months)"); print(tab(pd.cut(E.age_m, [0, 3, 6, 12, 24, 200])))
print("\n-- by recent level (units/month)"); print(tab(pd.cut(E.level, [0, 100, 300, 1000, 3000, 1e6])))
print("\n-- by occasion-tagged"); print(tab("tagged"))
print("\n-- by macro category"); print(tab("macro"))
print("\n-- by origin month"); print(tab("omonth"))
print("\n-- by recent momentum (last 28d vs 91d run-rate)"); print(tab(pd.cut(E.f_r28, [-9, -0.5, -0.15, 0.15, 9])))

cols = ["lvl", "age_m", "days_live", "f_r7", "f_r28", "trend91", "peak_r", "vol", "n_lags", "sh_ret", "sh_drop", "sh_tt", "disc_rel", "tagged", "omonth", "color_num", "lfl_type", "own_sea", "macro"]
E["tagged"] = E.tagged.astype(int)
out = []
for o in sorted(E.origin.unique()):
    oo = pd.Period(o).ordinal; tr = E[E.o_ord + 4 <= oo]; te = E[E.origin == o]
    if tr.origin.nunique() < 3: continue
    m = lgb.LGBMRegressor(objective="l2", n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0).fit(tr[cols], tr.ale)
    te = te.assign(pred_ale=m.predict(te[cols]), q1=np.quantile(m.predict(tr[cols]), 0.2), q3=np.quantile(m.predict(tr[cols]), 0.7)); out.append(te)
O = pd.concat(out)
O["conf"] = np.select([O.pred_ale <= O.q1, O.pred_ale >= O.q3], [1, 3], 2)   # best 20% -> 1, worst 30% -> 3 (cut-offs from training folds only)
print(f"\n== error model, scored only on later folds ({O.origin.nunique()} origins, {len(O)} item-forecasts)")
print("rank corr(predicted abs log error, actual):", round(O[["pred_ale", "ale"]].corr(method="spearman").iloc[0, 1], 3))
print(O.groupby("conf").agg(n=("w20", "size"), within20=("w20", "mean"), within50=("w50", "mean"), med_ape=("ape", "median")).round(3))
print("by decile of predicted error:"); print(O.groupby(pd.qcut(O.pred_ale, 10, labels=False)).agg(n=("w20", "size"), within20=("w20", "mean"), within50=("w50", "mean")).round(3).T.to_string())
imp = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False); print("error-model drivers:", (imp / imp.sum()).round(3).head(8).to_dict())
J = O[O.origin == "2025-06"]
if len(J): print("fold 2025-06 only:"); print(J.groupby("conf").agg(n=("w20", "size"), within20=("w20", "mean"), within50=("w50", "mean")).round(3))
