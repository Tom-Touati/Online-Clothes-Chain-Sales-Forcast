"""Rule-based monthly unit forecast per style_color (version 1).

Forecast for item i, target month m (h = 1..4 months after the last data month B):
    pred = base_rate_i  x days(m)  x  level_i(m)  x  season_seg(m)  x  ramp_i(h)

  base_rate_i   recent units/day (last month blended with last 14 and last 91 days)
  level_i(m)    channel-weighted like-for-like (LFL) change from month B to month m,
                taken from the same calendar months one year earlier
                (web: LFL chain; retail: flagship store per day deflated for panel growth;
                 dropship / TikTok / other: flat)
  season_seg(m) product-type (x sleeve) seasonal profile relative to the whole panel,
                same months one year earlier, shrunk to a WINTER / SUMMER / CORE group
  ramp_i(h)     young items keep ramping for ~2 months after launch

Only pandas/numpy, simple aggregations. Run:  python rules.py
"""
import numpy as np
import pandas as pd

DATA = "data/"
W14, WLONG = 0.4, 0.2          # base blend weights (last 14 days, last 91 days); rest = last month
RAMP1, RAMP2 = 1.25, 1.1       # launched in month before base: x1.25 (then x1.1 more); one month earlier: x1.1
SHRINK_K = 8                   # pseudo-items pulling a segment profile towards its group profile
CH_K = 20                      # pseudo-units pulling an item's channel mix towards the panel mix
ANCHOR_PRE = 0.4               # weight of the "month before base" anchor for the web level
OTHER_LEVEL = 1.0              # dropship + TikTok etc.: last year was channel onboarding, assume flat
CAP = (0.15, 4.0)              # bounds on the seasonal relative factor

WINTER_TYPES = {"Hoodie", "Sweater", "Sweatpants", "Cardigan", "Jacket", "Coat", "Sweatshirt",
                "Leggings", "Scarf", "Leg Warmers", "Tights", "Vest", "Knitted Top", "Beanie", "Gloves"}
SUMMER_TYPES = {"Shorts", "Short Jeans", "Skort", "Crop Top", "Tank Top", "Swimwear Top",
                "Swimwear Bottom", "Romper", "Bikini", "Anklet", "Belly Chain", "Bracelet"}
TOPS_FAMILY = {"Top", "T-Shirt", "Bodysuit", "Dress", "Crop Top", "Shirt", "Corset", "Strapless"}


def load():
    it = pd.read_csv(DATA + "items.csv", parse_dates=["published_date", "first_sale", "last_sale"])
    it = it.set_index("style_color")
    at = pd.read_csv(DATA + "item_attributes.csv").set_index("style_color")
    di = pd.read_parquet(DATA + "daily_item.parquet")
    st = pd.read_parquet(DATA + "daily_store.parquet")
    # launch = earliest of publish date / first sale (first_sale is censored at 2024-01-01)
    it["launch"] = it[["published_date", "first_sale"]].min(axis=1)
    sleeve = at["sleeve"].reindex(it.index).fillna("").str.lower()
    long_sl = sleeve.str.contains("long") & it.product_type.isin(TOPS_FAMILY)
    sleeveless = sleeve.str.contains("sleeveless|strap") & it.product_type.isin(TOPS_FAMILY)
    # segment = product type, long-sleeve tops split off; group = coarse season group
    it["seg"] = it.product_type.fillna("NA") + np.where(long_sl, "|LS", "")
    it["grp"] = np.select([it.product_type.isin(WINTER_TYPES) | long_sl,
                           it.product_type.isin(SUMMER_TYPES) | sleeveless], ["WINTER", "SUMMER"], "CORE")
    di["m"] = di.date.dt.to_period("M")
    di["other"] = di["units"] - di["web_units"] - di["retail"]          # drop + TikTok + other online
    return it, di, st


def lfl_chain(X, launch, months, key):
    """Like-for-like chained index per key: month-over-month ratio of units over items that were
    live >30 days before the start of the earlier month and sold in it; chained from months[0]."""
    idx, n_min = {}, None
    cum = None
    for a, b in zip(months[:-1], months[1:]):
        ok = (launch.reindex(X.index) < a.start_time - pd.Timedelta(days=30)) & (X[a] > 0)
        g = X[ok].groupby(key.reindex(X.index)[ok])
        r = g[b].sum() / g[a].sum()
        cum = r if cum is None else cum.reindex(r.index.union(cum.index)) * r
        idx[b] = cum.copy()
        n = g.size()
        n_min = n if n_min is None else n.reindex(n.index.union(n_min.index)).combine(n_min, min)
    return pd.DataFrame(idx), n_min.fillna(0)


def seasonal_factors(it, di, st, B, targets, items_for_factors=None, anchor_pre=None):
    """Factors for targets relative to base month B, measured on the same months one year earlier."""
    anchor_pre = ANCHOR_PRE if anchor_pre is None else anchor_pre
    lyB = B - 12
    ly = [lyB] + [t - 12 for t in targets]
    d = di[di.m.isin(ly)]
    if items_for_factors is not None:
        d = d[d.style_color.isin(items_for_factors)]
    U = d.pivot_table(index="style_color", columns="m", values="units", aggfunc="sum").reindex(columns=ly).fillna(0)
    Wb = d.pivot_table(index="style_color", columns="m", values="web_units", aggfunc="sum").reindex(columns=ly).fillna(0)
    one = pd.Series("ALL", index=U.index)
    all_u, _ = lfl_chain(U, it.launch, ly, one)
    seg_u, seg_n = lfl_chain(U, it.launch, ly, it.seg)
    grp_u, grp_n = lfl_chain(U, it.launch, ly, it.grp)
    base = all_u.loc["ALL"]
    seg_rel = seg_u.div(base, axis=1)
    grp_rel = grp_u.div(base, axis=1)
    # web level: LFL chain of edikted.com units, converted to a per-day factor (months differ in length).
    # Anchor A: last year's base month.  Anchor B: last year's month before the base month, corrected by
    # this year's (B-1 -> B) per-day LFL move.  Blending both protects against a base month that was
    # promo-distorted in one of the two years (Feb-24 web jumped +53%/day on Jan-24; Feb-25 only +10%).
    pre = B - 1
    have = set(di.m.unique())
    if anchor_pre == 0 or not {pre - 12, pre, B} <= have:              # anchor B needs 13+ months of data
        anchor_pre, chain_months = 0.0, ly
    else:
        chain_months = [pre - 12] + ly
    Wp = di[di.m.isin(chain_months)].pivot_table(index="style_color", columns="m", values="web_units",
                                                 aggfunc="sum").reindex(columns=chain_months).fillna(0)
    if items_for_factors is not None:
        Wp = Wp[Wp.index.isin(items_for_factors)]
    ch, _ = lfl_chain(Wp, it.launch, chain_months, pd.Series("ALL", index=Wp.index))
    ch = ch.loc["ALL"]
    ch[chain_months[0]] = 1.0
    web_lfl = ch / ch[lyB]                                             # monthly-total ratio vs lyB (for retail)
    a_feb = {t: ch[t] / ch[lyB] * lyB.days_in_month / t.days_in_month for t in ly[1:]}
    a_pre, cur_pd = dict(a_feb), 1.0
    if anchor_pre > 0:
        a_pre = {t: ch[t] * (pre - 12).days_in_month / t.days_in_month for t in ly[1:]}
        Wc = di[di.m.isin([pre, B])].pivot_table(index="style_color", columns="m", values="web_units",
                                                 aggfunc="sum").reindex(columns=[pre, B]).fillna(0)
        cur, _ = lfl_chain(Wc, it.launch, [pre, B], pd.Series("ALL", index=Wc.index))
        cur_pd = cur.loc["ALL", B] * pre.days_in_month / B.days_in_month   # this year's per-day move
    web = pd.Series({t: (1 - anchor_pre) * a_feb[t] + anchor_pre * a_pre[t] / cur_pd for t in ly[1:]})
    # retail level: flagship (oldest) store units/day, deflated by panel growth (web total / web LFL)
    st = st.assign(m=st.date.dt.to_period("M"))
    flag = st.groupby("store").date.min().idxmin()
    sd = st[st.store == flag].groupby("m").units.sum() / st[st.store == flag].groupby("m").date.nunique()
    web_tot = di[di.m.isin(ly)].groupby("m").web_units.sum()
    ret = pd.Series({t: (sd.get(t, np.nan) / sd.get(lyB, np.nan)) / ((web_tot[t] / web_tot[lyB]) / web_lfl[t])
                     for t in ly[1:]}).fillna(1.0)
    return dict(seg_rel=seg_rel, seg_n=seg_n, grp_rel=grp_rel, web=web, ret=ret, ly=ly,
                a_feb=a_feb, a_pre=a_pre, cur_pd=cur_pd)


def forecast(origin, items, it, di, st, factors=None):
    """Forecast 4 months after origin (last date of data) for the given items."""
    o = pd.Timestamp(origin)
    B = o.to_period("M")
    targets = [B + h for h in range(1, 5)]
    d = di[di.date <= o]
    if factors is None:
        factors = seasonal_factors(it, d, st[st.date <= o], B, targets)
    launch = it.launch.reindex(items).fillna(o)
    live = ((o - launch).dt.days + 1).clip(lower=1)

    # ---- 1. base rate (units/day) -------------------------------------------------------------
    D = d[d.date > o - pd.Timedelta(days=91)].pivot_table(index="style_color", columns="date",
                                                           values="units", aggfunc="sum")
    D = D.reindex(index=items).fillna(0)
    def rate(n):
        return D.iloc[:, -n:].sum(axis=1) / np.minimum(live, n)
    r_month = rate(B.days_in_month)
    b = (1 - W14) * r_month + W14 * rate(14)
    b = b * r_month.sum() / max(b.sum(), 1e-9)                       # blend is level-neutral
    r91 = rate(91)
    b = (1 - WLONG) * b + WLONG * r91 * r_month.sum() / max(r91.sum(), 1e-9)
    b[D.iloc[:, -28:].sum(axis=1) == 0] = 0.0                        # stopped selling: nothing in 4 weeks

    # ---- 2. channel mix of the last month, shrunk to the panel mix ------------------------------
    lm = d[d.m == B].groupby("style_color")[["units", "web_units", "retail", "other"]].sum().reindex(items).fillna(0)
    tot = lm[["web_units", "retail", "other"]].sum()
    pan = tot / tot.sum()
    sh = (lm[["web_units", "retail", "other"]] + CH_K * pan) .div(lm["units"] + CH_K, axis=0)

    # ---- 3. seasonal profile per segment, shrunk to group ---------------------------------------
    seg = it.seg.reindex(items).fillna("NA")
    grp = it.grp.reindex(items).fillna("CORE")
    out = {}
    age = (B.start_time - launch).dt.days / 30.44                     # age at start of base month
    for h, t in enumerate(targets, start=1):
        lyt = t - 12
        level = (sh["web_units"] * factors["web"][lyt] + sh["retail"] * factors["ret"][lyt]
                 + sh["other"] * OTHER_LEVEL)
        g_rel = grp.map(factors["grp_rel"][lyt]).fillna(1.0)
        s_rel = seg.map(factors["seg_rel"][lyt])
        n = seg.map(factors["seg_n"]).fillna(0).where(s_rel.notna(), 0)   # no profile -> group only
        rel = ((n * s_rel.fillna(0) + SHRINK_K * g_rel) / (n + SHRINK_K)).clip(*CAP)
        # ---- 4. lifecycle ramp for young items ----------------------------------------------------
        ramp = np.where(age < 1, RAMP1 * (RAMP2 if h >= 2 else 1.0), np.where(age < 2, RAMP2, 1.0))
        out[str(t)] = (b * t.days_in_month * level * rel * ramp).clip(lower=0).fillna(0)
    return pd.DataFrame(out, index=items)


if __name__ == "__main__":
    it, di, st = load()
    items = pd.read_csv(DATA + "to_forecast.csv").style_color.tolist()
    origin = di.date.max()
    P = forecast(origin, items, it, di, st)
    pred = P.stack().rename("pred").reset_index()
    pred.columns = ["style_color", "month", "pred"]
    pred["pred"] = pred.pred.astype(float).round(2)
    assert len(pred) == 4 * len(items) and np.isfinite(pred.pred).all() and (pred.pred >= 0).all()
    pred.to_csv("predictions.csv", index=False)
    print(f"origin {origin.date()}  items {len(items)}  rows {len(pred)}")
    print(pred.groupby("month").pred.sum().round(0).to_string())
