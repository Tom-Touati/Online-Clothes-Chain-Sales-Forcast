"""D179 (overnight M5 goal, direction 5): weekly origins, i.e. many more training rows (M5 winners trained on daily rows).
A self-contained weekly pipeline (data/weekly_item.parquet, weekly_futr*.parquet, static_item.parquet from eda/176n): at a cutoff
week c (features from weeks up to c, horizon = the next 17 weeks), features:
  level (mean weekly units over 13 weeks), momentum (1 vs 4 weeks, 4 vs 4, 13 vs 13), year-on-year (13 weeks vs the same weeks a year
  earlier), own seasonal ramp (next 17 weeks last year / the 13 weeks before them last year), product-type ramp (same, summed over
  the type), age in weeks; known-future: event days in the next 17 weeks (16 events), mean normal temperature (site and where the
  item sells), site campaign intensity, week of year; static: product type, LLM scores.
Target z = log((units in the next 17 weeks + 1) / (17 x weekly level + 1)); LightGBM median + L2, averaged.
Versions differ only in the training cutoffs: month = the week ending nearest each month end (as the monthly models),
biweek = every 2nd week, week = every week. Time-honest: training cutoffs with c + 17 weeks <= the test cutoff.
Test at the 13 month-end origins: the last full week ending by the month end; the 17-week forecast is rescaled to the 4 calendar
months by days (flat rate). Writes data/q_backtest_wk_{month,biweek,week}.parquet (q50 in all quantile columns)."""
import sys, numpy as np, pandas as pd, lightgbm as lgb, time
sys.stdout.reconfigure(encoding="utf-8")
Y = pd.read_parquet("data/weekly_item.parquet"); FU = pd.read_parquet("data/weekly_futr.parquet").set_index("ds"); FI = pd.read_parquet("data/weekly_futr_item.parquet")
ST = pd.read_parquet("data/static_item.parquet").set_index("unique_id"); items = ST.index.values
weeks = pd.date_range(Y.ds.min(), FU.index.max(), freq="W-MON"); wi = {d: k for k, d in enumerate(weeks)}
U = np.zeros((len(items), len(weeks)), np.float32); ii = {s: k for k, s in enumerate(items)}
U[Y.unique_id.map(ii).values, Y.ds.map(wi).values] = Y.y.values
first = np.where(U.sum(1) > 0, np.argmax(U > 0, 1), len(weeks))
TI = FI.pivot(index="unique_id", columns="ds", values="t_norm_item").reindex(index=items, columns=weeks).values
ptype = pd.read_parquet("data/items.parquet").set_index("style_color").product_type.reindex(items).fillna("na").values
types = np.unique(ptype); tix = np.array([np.where(types == t)[0][0] for t in ptype])
UT = np.zeros((len(types), len(weeks))); np.add.at(UT, tix, U)
H = 17; EVC = [c for c in FU.columns if c.startswith("ev_")]
def feats(k):
    """Feature rows for cutoff week index k (weeks <= k observed)."""
    s = lambda a, b: U[:, max(a, 0):max(b, 0)].sum(1)
    lvl = s(k - 12, k + 1) / 13; live = first <= k
    f = {"lvl": np.log1p(lvl), "m1_4": np.log((U[:, k] + 1) / (s(k - 4, k) / 4 + 1)), "m4_4": np.log((s(k - 3, k + 1) + 1) / (s(k - 7, k - 3) + 1)),
         "m13": np.log((s(k - 12, k + 1) + 1) / (s(k - 25, k - 12) + 1)), "yoy": np.where(first <= k - 52, np.log((s(k - 12, k + 1) + 1) / (s(k - 64, k - 51) + 1)), np.nan),
         "own_ramp": np.where(first <= k - 64, np.log((s(k - 51, k - 51 + H) + 1) / (s(k - 64, k - 51) * H / 13 + 1)), np.nan),
         "type_ramp": (np.log((UT[:, max(k - 51, 0):max(k - 51 + H, 0)].sum(1) + 1) / (UT[:, max(k - 64, 0):max(k - 51, 0)].sum(1) * H / 13 + 1)))[tix] if k >= 64 else np.full(len(items), np.nan),
         "age_w": (k - first).astype(float), "ptype": tix}
    fw = FU.iloc[k + 1:k + 1 + H]
    for c in EVC: f[c] = np.full(len(items), fw[c].sum())
    for c in ["t_norm", "site_disc", "site_deep"]: f[c] = np.full(len(items), fw[c].mean())
    f["woy_sin"] = np.full(len(items), FU.woy_sin.iloc[k]); f["woy_cos"] = np.full(len(items), FU.woy_cos.iloc[k])
    f["t_item"] = np.nanmean(TI[:, k + 1:k + 1 + H], 1); f["t_item_d"] = f["t_item"] - np.nanmean(TI[:, max(k - 12, 0):k + 1], 1)
    F = pd.DataFrame(f); F = pd.concat([F, ST.reset_index(drop=True).drop(columns=[c for c in ST.columns if c.startswith("pt_")])], axis=1)
    F["k"] = k; F["style_color"] = items; F["lvl_raw"] = lvl; F["live"] = live
    F["yH"] = U[:, k + 1:k + 1 + H].sum(1) if k + H < len(weeks) else np.nan
    F["z"] = np.log((F.yH + 1) / (H * lvl + 1)); return F[live & (lvl > 0)]
month_end_k = {}
for o in [str(p) for p in pd.period_range("2024-03", "2026-06", freq="M")]:
    E = pd.Period(o, "M").to_timestamp(how="end").normalize(); month_end_k[o] = max(k for k, d in enumerate(weeks) if d + pd.Timedelta(days=6) <= E)
print("building features...", flush=True); t0 = time.time()
ALL = pd.concat([feats(k) for k in range(30, len(weeks) - 1)], ignore_index=True); print("rows", len(ALL), f"{time.time() - t0:.0f}s", flush=True)
X = [c for c in ALL.columns if c not in ("k", "style_color", "lvl_raw", "live", "yH", "z")]
P = dict(n_estimators=500, learning_rate=0.04, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, n_jobs=8)
mk = set(month_end_k.values())
for var in (sys.argv[1:] or ["month", "biweek", "week"]):
    out, t0 = [], time.time()
    for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
        kt = month_end_k[o]; tr = ALL[(ALL.k + H <= kt) & ALL.yH.notna()]
        if var == "month": tr = tr[tr.k.isin(mk)]
        elif var == "biweek": tr = tr[(tr.k % 2) == (kt % 2)]
        te = ALL[ALL.k == kt]
        z = np.mean([lgb.LGBMRegressor(objective=ob, **({"alpha": 0.5} if ob == "quantile" else {}), **P).fit(tr[X], tr.z).predict(te[X]) for ob in ("quantile", "l2")], axis=0)
        E = pd.Period(o, "M").to_timestamp(how="end").normalize(); E4 = (pd.Period(o, "M") + 4).to_timestamp(how="end").normalize()
        cut_end = weeks[kt] + pd.Timedelta(days=6); scale = (E4 - E).days / (H * 7)
        pred = ((H * te.lvl_raw.values + 1) * np.exp(z) - 1).clip(0) * scale
        R = pd.DataFrame({"origin": o, "style_color": te.style_color.values})
        for q in ["q10", "q25", "q40", "q50", "q60", "q75", "q90"]: R[q] = pred
        out.append(R)
    Q = pd.concat(out); Q.to_parquet(f"data/q_backtest_wk_{var}.parquet", index=False)
    print(var, "training rows at the last origin", len(tr), f"{time.time() - t0:.0f}s", flush=True)
