"""Bottom-up forecast by location (D125): one LightGBM median model over item x location rows, summed to the item's 4-month total.
Locations: edikted.com ('web'), other online channels (TikTok, Flip -> 'online_other'), each dropship partner, each retail store.
Row = (item, location, origin) where the item is in the forecast set at the origin and sold at that location in the last 3 months.
Target z = log((units at the location in the next 4 months + 1) / (4 x location level + 1)), location level = mean monthly units o-2..o.
Features: location history (log level, lags 0-5 relative to level, share of the item's level, months since the item's first sale there),
location identity and context (type, id, months since the location's first sale, log size, 3-month growth), and the item-level
features of the production quantile model (eda/20 FULL set: seasonality, age, type, attributes, occasions, pace, channel shares...).
Training rows: origins whose window ended by the test origin. Also reports the 'rollout gap': actual window units at locations where
the item had no sales in the 3 months before the origin (a bottom-up model cannot see them).
Writes data/bu_items.parquet (origin, style_color, bu = summed forecast, bu_cov = forecast on covered locations only)."""
import sys, numpy as np, pandas as pd, lightgbm as lgb, warnings, importlib.util, duckdb, time
OBJ = sys.argv[1] if len(sys.argv) > 1 else "median"     # median | tweedie (expected value per location, so sums are unbiased; D125b)
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
spec = importlib.util.spec_from_file_location("q20", "eda/20_quantiles.py"); q20 = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(q20); sys.argv = _a
T1 = q20.T1
d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[~d.cancelled & (d["class"] != "RETAIL")]; d["ny_date"] = pd.to_datetime(d.ny_date)
d["loc"] = np.where(d["class"] == "DROP", "drop_" + d.chan.astype(str), np.where(d.chan == "edikted.com", "web", "online_other"))
d["m"] = d.ny_date.dt.to_period("M").astype(str)
A = d.groupby(["style_color", "loc", "m"]).qty.sum()
con = duckdb.connect(); con.execute("SET memory_limit='1500MB'; SET threads=2")
r = con.execute("SELECT style_color, 'store_' || store loc, strftime(ny_date::DATE, '%Y-%m') m, sum(qty_ordered) qty FROM read_parquet('Dataset.parquet') WHERE NOT cancelled AND class='RETAIL' GROUP BY ALL").fetchdf()
A = pd.concat([A, r.set_index(["style_color", "loc", "m"]).qty]).groupby(level=[0, 1, 2]).sum()
L = A.unstack("m").fillna(0); L = L[sorted(L.columns)]                        # (item, loc) x month
MON = list(L.columns); P = lambda o, k: str(pd.Period(o, "M") + k)
loc_tot = L.groupby(level="loc").sum(); loc_first = loc_tot.gt(0).idxmax(axis=1)
il_first = L.gt(0).idxmax(axis=1)
def ltype(l): return "web" if l == "web" else "online_other" if l == "online_other" else "drop" if l.startswith("drop_") else "store"
rows = []
for o in sorted(T1.origin.unique()):
    if P(o, -5) not in MON: continue
    lv = L[[P(o, -2), P(o, -1), P(o, 0)]].mean(axis=1); keep = lv > 0
    items = set(T1[T1.origin == o].style_color)
    X = L[keep][[]].reset_index(); X = X[X.style_color.isin(items)]
    idx = pd.MultiIndex.from_frame(X[["style_color", "loc"]])
    X["lv"] = lv.reindex(idx).values; X["l_lvl"] = np.log1p(X.lv)
    for k in range(6): X[f"llag{k}"] = np.log((L[P(o, -k)].reindex(idx).values + 1) / (X.lv + 1)) if P(o, -k) in MON else np.nan
    it_lv = X.groupby("style_color").lv.transform("sum"); X["share"] = X.lv / it_lv
    X["il_age"] = [(pd.Period(o, "M") - pd.Period(f, "M")).n for f in il_first.reindex(idx).values]
    X["ltype"] = [ltype(l) for l in X["loc"]]; X["loc_age"] = [(pd.Period(o, "M") - pd.Period(loc_first[l], "M")).n for l in X["loc"]]
    lt3 = loc_tot[[P(o, -2), P(o, -1), P(o, 0)]].sum(axis=1); lp3 = loc_tot[[P(o, -5), P(o, -4), P(o, -3)]].sum(axis=1)
    X["loc_size"] = np.log1p(lt3.reindex(X["loc"]).values); X["loc_growth"] = np.log((lt3 + 1) / (lp3 + 1)).reindex(X["loc"]).values
    fut = [P(o, h) for h in range(1, 5)]
    X["ytot_loc"] = L[fut].sum(axis=1).reindex(idx).values if all(f in MON for f in fut) else np.nan
    X["origin"] = o; rows.append(X)
B = pd.concat(rows, ignore_index=True)
ITEMF = [c for c in q20.FULL if c not in ("lvl",)] + ["lvl", "o_ord", "ytot"]
B = B.merge(T1[["origin", "style_color"] + ITEMF], on=["origin", "style_color"], how="left")
B["z"] = np.log((B.ytot_loc + 1) / (4 * B.lv + 1))
for c in ["ltype", "loc"]: B[c] = B[c].astype("category")
FEAT = ["l_lvl"] + [f"llag{k}" for k in range(6)] + ["share", "il_age", "ltype", "loc", "loc_age", "loc_size", "loc_growth"] + [c for c in q20.FULL]
PRM = dict(objective="quantile", alpha=0.5, n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=4)
out, t0, LOCOUT = [], time.time(), []
for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
    oo = pd.Period(o).ordinal; tr = B[(B.o_ord + 4 <= oo) & B.ytot_loc.notna()]; te = B[B.origin == o].copy()
    if OBJ == "tweedie":
        PT = PRM | dict(objective="tweedie", tweedie_variance_power=1.3); PT.pop("alpha")
        m = lgb.LGBMRegressor(**PT).fit(tr[FEAT], tr.ytot_loc, init_score=np.log(4 * tr.lv + 0.1))
        te["pred"] = np.exp(m.predict(te[FEAT], raw_score=True) + np.log(4 * te.lv + 0.1))
    else:
        m = lgb.LGBMRegressor(**PRM).fit(tr[FEAT], tr.z)
        te["pred"] = np.clip((4 * te.lv + 1) * np.exp(m.predict(te[FEAT])) - 1, 0, None)
    g = te.groupby("style_color").agg(bu=("pred", "sum"), cov_actual=("ytot_loc", "sum"), n_loc=("loc", "size")).reset_index()
    LOCOUT.append(te[["origin", "style_color", "loc", "ltype", "lv", "pred", "ytot_loc"]])
    g["origin"] = o; out.append(g)
    print(o, "train rows", len(tr), "test rows", len(te), f"{time.time() - t0:.0f}s", flush=True)
pd.concat(LOCOUT).to_parquet(f"data/bu_loc{'_tw' if OBJ == 'tweedie' else ''}.parquet", index=False)
U = pd.concat(out, ignore_index=True); U.to_parquet(f"data/bu_items{'_tw' if OBJ == 'tweedie' else ''}.parquet", index=False)
