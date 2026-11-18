"""D322 (fork; proposal #6 of the error study, assigned in D314: "Re-forecast after month 1"). The month-1 update: after the first month of a
4-month window has sold, re-forecast months 2-4. Shared by the backtest (eda/198af_month1_update.py) and the 2026 script (eda/month1_update_2026.py).
  split     the 4-month forecast F is split into month 1 and months 2-4 by the product type's share of month 1 in the same window last year
            (0.25 when last year is missing); in 2026 the workbook's own July / Aug-Oct split is used
  inputs    known at the end of month 1: month-1 sales against the month-1 forecast and against the recency level; own stores, partners and
            the web discount in month 1 against the origin month; plus the origin's level, age, forecast against R3, model disagreement, trend,
            product group, TikTok share
  target    log((actual months 2-4 + 1) / (forecast months 2-4 + 1)); LightGBM median, trained on origins whose windows have closed"""
import numpy as np, pandas as pd, lightgbm as lgb
P_ = lambda o, k: str(pd.Period(o, "M") + k)
FEATS = ["m1r", "m1lvl", "st1", "pa1", "disc1", "llv", "age_m", "lf", "spread", "trend91", "macro_c", "sh_tiktok"]
PR = dict(objective="quantile", alpha=0.5, n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=3)
_M = pd.read_parquet("data/monthly_item.parquet"); _M.columns = [str(c) for c in _M.columns]; _A = pd.read_parquet("data/fk_axes_month.parquet").set_index(["style_color", "m"])
_PT = pd.read_parquet("data/items.parquet").set_index("style_color").product_type.reindex(_M.index).fillna("na"); _MT = _M.groupby(_PT.values).sum()


def ax(col, items, m):
    return _A[col].reindex(pd.MultiIndex.from_arrays([items, [m] * len(items)])).values.astype(float)


def type_share_m1(ptypes, o):
    """share of month o+1 in the window o+1..o+4, for each product type, the same window last year"""
    cols = [P_(o, k - 12) for k in (1, 2, 3, 4)]
    if not all(c in _MT.columns for c in cols): return np.full(len(ptypes), 0.25)
    w = _MT[cols]; s = (w[cols[0]] / w.sum(1).replace(0, np.nan)).clip(0.1, 0.5); return s.reindex(ptypes).fillna(0.25).values


def features(Z, o, m1, f_m1):
    """Z: rows at origin o (style_color, level, F, R3, age_m, spread, trend91, macro, sh_tiktok); m1: month-1 actuals; f_m1: month-1 forecast"""
    s = Z.style_color.values; X = pd.DataFrame(index=Z.index)
    X["m1r"] = np.log((m1 + 1) / (f_m1 + 1)); X["m1lvl"] = np.log((m1 + 1) / (Z.level.values + 1))
    with np.errstate(all="ignore"):
        X["st1"] = np.nan_to_num(ax("n_stores", s, P_(o, 1))) - np.nan_to_num(ax("n_stores", s, o)); X["pa1"] = np.nan_to_num(ax("n_partners", s, P_(o, 1))) - np.nan_to_num(ax("n_partners", s, o))
        X["disc1"] = ax("disc_web", s, P_(o, 1)) - ax("disc_web", s, o)
    X["llv"] = np.log1p(Z.level.values); X["age_m"] = Z.age_m.values; X["lf"] = np.log((Z.F.values + 1) / (Z.R3.values + 1)); X["spread"] = Z.spread.values; X["trend91"] = Z.trend91.values
    X["macro_c"] = pd.Categorical(Z.macro.astype(str), categories=["bottom", "dress", "other", "swim", "top", "warm"]).codes; X["sh_tiktok"] = Z.sh_tiktok.values; return X


def backtest_table():
    D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); D["oo"] = D.origin.map(lambda o: pd.Period(o, "M").ordinal); parts = []
    for o, Z in D.groupby("origin"):
        s = Z.style_color.values; m1 = _M.reindex(s)[P_(o, 1)].fillna(0).values if P_(o, 1) in _M else np.zeros(len(Z)); s1 = type_share_m1(Z.ptype.astype(str).values, o)
        f_m1 = Z.F.values * s1; X = features(Z, o, m1, f_m1); X["m1"] = m1; X["f_m1"] = f_m1; X["f_rest"] = Z.F.values - f_m1; X["y_rest"] = Z.y.values - m1; X["r3_rest"] = Z.R3.values * 0.75; parts.append(X)
    X = pd.concat(parts).reindex(D.index); return pd.concat([D[["origin", "oo", "style_color", "y", "F", "R3"]], X], axis=1)


def train(rows):
    t = np.log((rows.y_rest.clip(lower=0) + 1) / (rows.f_rest + 1)); return lgb.LGBMRegressor(**PR).fit(rows[FEATS], t)


def update(model, rows, shrink=1.0):
    return np.clip((rows.f_rest.values + 1) * np.exp(shrink * model.predict(rows[FEATS])) - 1, 0, None)
