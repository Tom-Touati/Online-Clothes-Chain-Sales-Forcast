"""Partly recursive tree ensemble (D131), 13 origins, monthly item forecasts (as the D009-D013 LightGBM ensemble: two median-loss models on
the BASE and FULL feature sets, log ratio of the month to the level, pooled over h).
  direct     : the production design (h is a feature; no information about the earlier months of the window)
  recursive  : the same plus fut1..fut3 = log ratio to level of months o+1..o+h-1; trained on the actual values (teacher forcing),
               predicted step by step at the origin, feeding each month's prediction into the next
  partly     : geometric mean of direct and recursive (as the M5 winner averaged recursive and non-recursive models)
Writes data/rec_backtest_preds.parquet (origin, style_color, h, direct, recursive, partly, y)."""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
F = pd.read_parquet("data/features.parquet")
for c in ["ptype", "macro"] + [c for c in F.columns if c.startswith("a_")]: F[c] = F[c].astype("category")
OCC = [c for c in F.columns if c.startswith("occ_")]
BASE = [f"lagr{k}" for k in range(1, 7)] + ["lvl", "age_m", "ptype", "tmonth", "h", "color_num", "own_sea"] + OCC
FULL = BASE + ["f_r7", "f_r28", "trend91", "peak_r", "days_live", "sh_ret", "sh_drop", "sh_tt", "disc_item", "disc_rel", "lfl_type", "lfl_macro", "lfl_occ"] + [c for c in F.columns if c.startswith("a_")]
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=4)
F["rlog"] = np.log((F.y + 1) / (F.level + 1))
W = F.pivot_table(index=["origin", "style_color"], columns="h", values="rlog")              # actual log ratios of the window months
for k in (1, 2, 3):
    v = W[k].reindex(pd.MultiIndex.from_arrays([F.origin, F.style_color])).values
    F[f"fut{k}"] = np.where(F.h > k, v, np.nan)
FUT = ["fut1", "fut2", "fut3"]
out, t0 = [], time.time()
for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
    oo = pd.Period(o).ordinal; tr = F[(F.o_ord + F.h <= oo) & F.y.notna()]
    tr_rec = tr[tr.o_ord + 4 <= oo]                                   # recursive training needs the window's earlier months known too
    te = F[F.origin == o].copy(); te[FUT] = np.nan
    dr = np.zeros(len(te)); rc = {h: None for h in range(1, 5)}
    for cols in (BASE, FULL):
        md = lgb.LGBMRegressor(objective="l1", **P).fit(tr[cols], tr.rlog); dr += md.predict(te[cols]) / 2
        mr = lgb.LGBMRegressor(objective="l1", **P).fit(tr_rec[cols + FUT], tr_rec.rlog)
        X = te.copy()
        preds = {}
        for h in range(1, 5):
            sel = X.h.values == h
            for k in range(1, h): X.loc[sel, f"fut{k}"] = preds[k].reindex(X.loc[sel, "style_color"]).values
            p = pd.Series(mr.predict(X.loc[sel, cols + FUT]), index=X.loc[sel, "style_color"].values); preds[h] = p
            rc[h] = p / 2 if rc[h] is None else rc[h] + p / 2
    te["direct"] = np.clip((te.level + 1) * np.exp(dr) - 1, 0, None)
    te["recursive"] = np.clip((te.level + 1) * np.exp(np.concatenate([rc[h].reindex(te.style_color[te.h == h]).values for h in range(1, 5)])[np.argsort(np.argsort(te.h.values, kind="stable"), kind="stable")]) - 1, 0, None)
    te["partly"] = np.sqrt((te.direct + 0.5) * (te.recursive + 0.5)) - 0.5
    out.append(te[["origin", "style_color", "h", "direct", "recursive", "partly", "y"]])
    x = te; print(o, {c: round((x[c] - x.y).abs().sum() / x.y.sum(), 3) for c in ["direct", "recursive", "partly"]}, f"{time.time() - t0:.0f}s", flush=True)
pd.concat(out).to_parquet("data/rec_backtest_preds.parquet", index=False)
