"""Finest-grain bottom-up forecast (D126): item x location x month, summed to item-months and to the item's 4-month total.
Same locations, rows and features as eda/113 (location history and context + the item-level features), one row per horizon h = 1..4 with
h, the target calendar month and the item-level like-for-like indices for that month; target z = log((units at the location in month h + 1)
/ (location level + 1)). One pooled LightGBM median model per test origin, trained on rows whose month was complete by the origin.
Writes data/bum_items.parquet (origin, style_color, h, bum, y_cov) and data/bum_totals.parquet (origin, style_color, bum)."""
import sys, numpy as np, pandas as pd, lightgbm as lgb, warnings, importlib.util, duckdb, time
OBJ = sys.argv[1] if len(sys.argv) > 1 else "median"     # median | tweedie (expected value per location, so sums are unbiased; D125b)
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
spec = importlib.util.spec_from_file_location("q20", "eda/20_quantiles.py"); q20 = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(q20); sys.argv = _a
src = open("eda/113_bottom_up.py", encoding="utf-8").read().split('B["z"] = ')[0]; _OBJ = OBJ          # reuse the location table B (4-month rows)
exec(src.split("import sys, numpy")[1].split("\n", 1)[1])
F = pd.read_parquet("data/features.parquet", columns=["origin", "style_color", "h", "lfl_type", "lfl_macro", "lfl_occ", "own_sea", "tmonth"])
rows = []
for h in range(1, 5):
    X = B.copy(); X["h"] = h
    X["y_loc"] = [L.at[(s, l), P(o, h)] if P(o, h) in MON else np.nan for s, l, o in zip(X.style_color, X["loc"], X.origin)]
    X = X.merge(F[F.h == h].drop(columns="h"), on=["origin", "style_color"], how="left", suffixes=("", "_m"))
    rows.append(X)
BM = pd.concat(rows, ignore_index=True); BM["z"] = np.log((BM.y_loc + 1) / (BM.lv + 1))
for c in ["ltype", "loc"]: BM[c] = BM[c].astype("category")
FEAT = ["l_lvl"] + [f"llag{k}" for k in range(6)] + ["share", "il_age", "ltype", "loc", "loc_age", "loc_size", "loc_growth", "h", "tmonth_m",
        "lfl_type", "lfl_macro", "lfl_occ", "own_sea"] + [c for c in q20.FULL if c not in ("tmonth", "own_sea4", "lfl_type4", "lfl_macro4", "lfl_occ4")]
PRM = dict(objective="quantile", alpha=0.5, n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=200, subsample=0.7, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=4)
out, t0 = [], time.time()
for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
    oo = pd.Period(o).ordinal; tr = BM[(BM.o_ord + BM.h <= oo) & BM.y_loc.notna()]; te = BM[BM.origin == o].copy()
    if OBJ == "tweedie":
        PT = PRM | dict(objective="tweedie", tweedie_variance_power=1.3); PT.pop("alpha")
        m = lgb.LGBMRegressor(**PT).fit(tr[FEAT], tr.y_loc, init_score=np.log(tr.lv + 0.1))
        te["pred"] = np.exp(m.predict(te[FEAT], raw_score=True) + np.log(te.lv + 0.1))
    else:
        m = lgb.LGBMRegressor(**PRM).fit(tr[FEAT], tr.z)
        te["pred"] = np.clip((te.lv + 1) * np.exp(m.predict(te[FEAT])) - 1, 0, None)
    out.append(te.groupby(["origin", "style_color", "h"]).agg(bum=("pred", "sum"), y_cov=("y_loc", "sum")).reset_index())
    print(o, "train rows", len(tr), "test rows", len(te), f"{time.time() - t0:.0f}s", flush=True)
U = pd.concat(out, ignore_index=True); SUF = "_tw" if OBJ == "tweedie" else ""; U.to_parquet(f"data/bum_items{SUF}.parquet", index=False)
U.groupby(["origin", "style_color"]).bum.sum().reset_index().to_parquet(f"data/bum_totals{SUF}.parquet", index=False)
