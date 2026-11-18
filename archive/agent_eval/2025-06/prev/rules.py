"""Rule-based monthly unit forecast per style_color (version 4, snapshot 2025-05-31).

Forecast for item i, target month m (h = 1..4 months after the last data month B):
    pred = base_rate_i x days(m) x level_i(m) x season_seg(m) x ramp_i(h) x occasion_i(m)

  base_rate_i   recent units/day (last month blended with last 14 and last 91 days)
  level_i(m)    channel-weighted change from month B to month m, from the same months one year earlier
                (web: 75% "deflated" = LY like-for-like (LFL) move / LY whole-site web move, i.e. the
                 panel's drift relative to the site with this year's site held flat; 25% the LY LFL
                 chain itself (July-24 regime jump removed, pre-anchor may only lower it);
                 retail: flagship store per day deflated for panel growth, damped f**0.7, plus a
                 full-month uplift for stores that opened inside the base month;
                 dropship / TikTok / other: x0.93 per month; site web path by calendar month)
  season_seg(m) product-type (x sleeve) seasonal profile relative to the whole panel,
                same months one year earlier, shrunk to a WINTER / SUMMER / CORE group
  ramp_i(h)     young items keep ramping for ~2 months after launch
  occasion_i(m) occasion-tag calendar index, target month / base month (4th of July, Graduation,
                Homecoming, Halloween)

Only pandas/numpy, simple aggregations. Run:  python rules.py
"""
import numpy as np
import pandas as pd

DATA = "data/"
P = dict(
    W14=0.4, WLONG=0.2,        # base blend weights (last 14 days, last 91 days); rest = last month
    RAMP0=1.6,                 # launched inside the base month (partial-month base)
    RAMP1=1.4,                 # launched in the month before the base month
    RAMP2=1.15,                # launched two months before the base month
    RAMP_H2=1.1,               # extra growth from h=2 on for the RAMP0 / RAMP1 classes
    POW=0.0,                   # level-neutral regression to the mean of base rates: b * (b/mean)^-POW
    SHRINK_K=8,                # pseudo-items pulling a segment profile towards its group profile
    CH_K=20,                   # pseudo-units pulling an item's channel mix towards the panel mix
    ANCHOR_PRE=0.4,            # web level: weight of the "month before base" anchor
    RET_ANCHOR_PRE=0.25,       # retail level: weight of the "month before base" anchor
    OTHER_LEVEL=1.0,           # dropship + TikTok etc.: last year was channel onboarding, assume flat
    CAP=(0.15, 4.0),           # bounds on the seasonal relative factor
    NEW_STORE=True,            # stores opened inside the base month count at their run-rate
    PRE_ASYM=True,             # pre-anchor may only LOWER the web/retail factor (protects against a
                               # spiked base month, but does not assume a dip in the base month reverts)
    W_DEFL=0.75,               # web level: weight of the "deflated" factor = LY LFL / LY site web (per day)
    SITE_WEB=1.0,              # flat level of the assumed edikted.com site units/day (x SITE_WEB_IDX)
    OTHER_MONTHLY=0.93,        # dropship + TikTok etc.: per-month factor, compounded over h
    RET_POW=0.6,               # retail seasonal factor damped as f**RET_POW (2024 flagship was still ramping)
    SEAS_ANCHOR=0.0,           # season-group level anchoring: rel x (LY 2-month group move / TY move)**a
)
# Assumed edikted.com site web units/day by calendar month (index, default 1.0); the deflated web factor
# uses idx[target] / idx[base].  2024 post-regime months: Aug 9.5k, Sep 8.6k, Oct 8.0k per day (back-to-
# school lift, post-Labor-Day fade); May and June both contain a 3-day sale (Memorial Day, mid-June).
SITE_WEB_IDX = {5: 1.0, 6: 1.0, 7: 1.0, 8: 1.03, 9: 0.93}
# Occasion tags with a calendar peak: demand index by calendar month (default 1.0); the multiplier is
# idx[target month] / idx[base month], so a peak already inside the base month is not counted twice.
# Last year's tagged items vs items of the same segment, May-24 base (research/r7_occasion.py), shrunk
# about halfway; this year's scoreboard (tags vs total, May-25): 4th of July 1.20, Graduation 1.20,
# Homecoming 1.20 under-forecast.
#   4th of July (n=4): Jun 1.28 Jul 1.13 Aug 1.18 Sep 0.95 -> rises into June, holds   -> May->Jun x1.13
#   Graduation  (n=8, all dresses): Jun 0.77 Jul 0.80 Aug 0.67 Sep 0.64 -> prom/grad season over
#   Homecoming  (n=8, all dresses): Jun 0.61 Jul 0.87 Aug 1.21 Sep 2.68 -> Sep homecoming dances
#   Halloween   (n=22): Sep 1.37, Oct 2.0
OCCASION = {"4th of July": {5: 1.15, 6: 1.3, 7: 1.2, 8: 1.15, 9: 1.1},
            "Graduation": {4: 1.1, 5: 1.2, 6: 1.0, 7: 1.0, 8: 0.95, 9: 0.95},
            "Homecoming": {5: 1.2, 6: 0.95, 7: 1.05, 8: 1.3, 9: 1.9, 10: 1.9},
            "Halloween": {9: 1.15, 10: 1.6}}
# Late July 2024 edikted.com moved into a permanent deep-discount regime (web discount 0.45 -> 0.57,
# LFL web/day x2.3 vs June).  2025 is already in that regime, so last year's Jun->Jul web move is
# replaced by the discount-matched part of July (Jul 8-21: discount 0.45 vs June 0.46).
REGIME_DATE = pd.Timestamp("2024-07-22")
CLEAN_JUL = ("2024-07-08", "2024-07-21")

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


def web_regime_fix(di, it, ch, months):
    """If the chain crosses the July-2024 regime jump, replace the Jun->Jul-24 monthly ratio by the
    discount-matched one: LFL web units/day on Jul 8-21 vs June, times 31/30 days."""
    jun, jul = pd.Period("2024-06"), pd.Period("2024-07")
    if jun not in months or jul not in months:
        return ch
    d = di[(di.date >= jun.start_time) & (di.date <= pd.Timestamp(CLEAN_JUL[1]))]
    ok = d[d.m == jun].groupby("style_color").units.sum()
    ok = ok[(ok > 0) & (it.launch.reindex(ok.index) < jun.start_time - pd.Timedelta(days=30))].index
    w = d[d.style_color.isin(ok)].groupby("date").web_units.sum()
    lift = w[CLEAN_JUL[0]:CLEAN_JUL[1]].mean() / w[w.index < jul.start_time].mean()
    new = ch.copy()
    new[jul] = ch[jun] * lift * jul.days_in_month / jun.days_in_month
    for t in months[months.index(jul) + 1:]:                           # later months keep their own moves
        new[t] = ch[t] / ch[jul] * new[jul]
    return new


def seasonal_factors(it, di, st, B, targets, items_for_factors=None, anchor_pre=None, ret_anchor_pre=None):
    """Factors for targets relative to base month B, measured on the same months one year earlier."""
    anchor_pre = P["ANCHOR_PRE"] if anchor_pre is None else anchor_pre
    ret_anchor_pre = P["RET_ANCHOR_PRE"] if ret_anchor_pre is None else ret_anchor_pre
    lyB = B - 12
    ly = [lyB] + [t - 12 for t in targets]
    d = di[di.m.isin(ly)]
    if items_for_factors is not None:
        d = d[d.style_color.isin(items_for_factors)]
    U = d.pivot_table(index="style_color", columns="m", values="units", aggfunc="sum").reindex(columns=ly).fillna(0)
    one = pd.Series("ALL", index=U.index)
    all_u, _ = lfl_chain(U, it.launch, ly, one)
    seg_u, seg_n = lfl_chain(U, it.launch, ly, it.seg)
    grp_u, grp_n = lfl_chain(U, it.launch, ly, it.grp)
    base = all_u.loc["ALL"]
    seg_rel = seg_u.div(base, axis=1)
    grp_rel = grp_u.div(base, axis=1)
    # season-group level anchoring: if a group moved more (or less) relative to the panel over the last two
    # months this year than over the same two months last year, part of that gap closes again later
    anc = pd.Series(1.0, index=grp_rel.index)
    tyw = [B - 2, B - 1, B]
    lyw = [t - 12 for t in tyw]
    if P["SEAS_ANCHOR"] > 0 and set(lyw) <= set(di.m.unique()):
        def cum_rel(ms):
            Uq = di[di.m.isin(ms)].pivot_table(index="style_color", columns="m", values="units",
                                               aggfunc="sum").reindex(columns=ms).fillna(0)
            g, _ = lfl_chain(Uq, it.launch, ms, it.grp)
            a, _ = lfl_chain(Uq, it.launch, ms, pd.Series("ALL", index=Uq.index))
            return g[ms[-1]] / a.loc["ALL", ms[-1]]
        anc = ((cum_rel(lyw) / cum_rel(tyw)) ** P["SEAS_ANCHOR"]).clip(0.7, 1.4).reindex(grp_rel.index).fillna(1.0)
    # web level: LFL chain of edikted.com units, converted to a per-day factor (months differ in length).
    # Anchor A: last year's base month.  Anchor B: last year's month before the base month, corrected by
    # this year's (B-1 -> B) per-day LFL move.  Blending both protects against a base month that was
    # promo-distorted in one of the two years (Feb-24 web jumped +53%/day on Jan-24; Feb-25 only +10%).
    pre = B - 1
    have = set(di.m.unique())
    if not {pre - 12, pre, B} <= have:                                 # anchor B needs 13+ months of data
        anchor_pre, ret_anchor_pre, chain_months = 0.0, 0.0, ly
    else:
        chain_months = [pre - 12] + ly
    Wp = di[di.m.isin(chain_months)].pivot_table(index="style_color", columns="m", values="web_units",
                                                 aggfunc="sum").reindex(columns=chain_months).fillna(0)
    if items_for_factors is not None:
        Wp = Wp[Wp.index.isin(items_for_factors)]
    ch, _ = lfl_chain(Wp, it.launch, chain_months, pd.Series("ALL", index=Wp.index))
    ch = ch.loc["ALL"]
    ch[chain_months[0]] = 1.0
    ch_raw = ch.copy()                                                 # raw chain: used to deflate retail
    ch = web_regime_fix(di, it, ch, chain_months)
    a_feb = {t: ch[t] / ch[lyB] * lyB.days_in_month / t.days_in_month for t in ly[1:]}
    a_pre, cur_pd, cur_ret = dict(a_feb), 1.0, 1.0
    if chain_months[0] != lyB:
        a_pre = {t: ch[t] * (pre - 12).days_in_month / t.days_in_month for t in ly[1:]}
        Xc = di[di.m.isin([pre, B])]
        cur = {}
        for col in ["web_units", "retail"]:                            # this year's per-day (B-1 -> B) LFL moves
            Xp = Xc.pivot_table(index="style_color", columns="m", values=col, aggfunc="sum").reindex(columns=[pre, B]).fillna(0)
            c, _ = lfl_chain(Xp, it.launch, [pre, B], pd.Series("ALL", index=Xp.index))
            cur[col] = c.loc["ALL", B] * pre.days_in_month / B.days_in_month
        cur_pd, cur_ret = cur["web_units"], cur["retail"]
    pre_w = {t: a_pre[t] / cur_pd for t in ly[1:]}
    if P["PRE_ASYM"]:
        pre_w = {t: min(pre_w[t], a_feb[t]) for t in ly[1:]}
    web = pd.Series({t: (1 - anchor_pre) * a_feb[t] + anchor_pre * pre_w[t] for t in ly[1:]})
    # deflated web factor: last year's LFL move divided by last year's whole-site web move (removes 2024's
    # growth and the July-24 discount-regime jump), times an assumed site trend for this year
    site_pd = di[di.m.isin(ly)].groupby("m").web_units.sum() / di[di.m.isin(ly)].groupby("m").date.nunique()
    defl_w = pd.Series({t: ch_raw[t] / ch_raw[lyB] * lyB.days_in_month / t.days_in_month
                        / (site_pd[t] / site_pd[lyB]) * P["SITE_WEB"]
                        * SITE_WEB_IDX.get(t.month, 1.0) / SITE_WEB_IDX.get(B.month, 1.0) for t in ly[1:]})
    web = (1 - P["W_DEFL"]) * web + P["W_DEFL"] * defl_w
    # retail level: flagship (oldest) store units/day, deflated by panel growth (web total / web LFL);
    # same two anchors as web (Feb-24 was a retail promo month: retail discount 0.38 vs 0.18 in Mar-24)
    st = st.assign(m=st.date.dt.to_period("M"))
    flag = st.groupby("store").date.min().idxmin()
    sd = st[st.store == flag].groupby("m").units.sum() / st[st.store == flag].groupby("m").date.nunique()
    web_tot = di[di.m.isin(chain_months)].groupby("m").web_units.sum()
    def defl(t, a):                                                    # flagship per-day move a -> t, deflated
        return (sd.get(t, np.nan) / sd.get(a, np.nan)) / ((web_tot[t] / web_tot[a]) / (ch_raw[t] / ch_raw[a]))
    def ret_pre(t):
        v = defl(t, pre - 12) / cur_ret
        return min(v, defl(t, lyB)) if P["PRE_ASYM"] else v
    ret = pd.Series({t: (1 - ret_anchor_pre) * defl(t, lyB) +
                     (ret_anchor_pre * ret_pre(t) if ret_anchor_pre > 0 else 0)
                     for t in ly[1:]}).fillna(1.0)
    # stores that opened inside the base month: count them at their run-rate (median day of the last
    # 14 days, robust to the opening-day spike) for a full month instead of their partial-month total
    ret_adj = 1.0
    sB = st[st.m == B]
    opened = st.groupby("store").date.min()
    new = opened[opened >= B.start_time].index
    if P["NEW_STORE"] and len(new) and sB.units.sum() > 0:
        last = sB[sB.date > sB.date.max() - pd.Timedelta(days=14)]
        run = sum(last[last.store == s].groupby("date").units.sum().median() * B.days_in_month for s in new)
        ret_adj = (sB.units.sum() - sB[sB.store.isin(new)].units.sum() + run) / sB.units.sum()
    return dict(anc=anc, defl_w=defl_w, seg_rel=seg_rel, seg_n=seg_n, grp_rel=grp_rel, web=web,
                ret=(ret ** P["RET_POW"]) * ret_adj, ly=ly,
                a_feb=a_feb, a_pre=a_pre, cur_pd=cur_pd, cur_ret=cur_ret, ret_adj=ret_adj)


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
    b = (1 - P["W14"]) * r_month + P["W14"] * rate(14)
    b = b * r_month.sum() / max(b.sum(), 1e-9)                       # blend is level-neutral
    r91 = rate(91)
    b = (1 - P["WLONG"]) * b + P["WLONG"] * r91 * r_month.sum() / max(r91.sum(), 1e-9)
    b[D.iloc[:, -28:].sum(axis=1) == 0] = 0.0                        # stopped selling: nothing in 4 weeks
    if P["POW"] > 0:                                                 # big sellers regress, small ones rise
        pos = b > 0
        tot_b = b.sum()
        b[pos] = b[pos] * (b[pos] / b[pos].mean()) ** (-P["POW"])
        b = b * tot_b / max(b.sum(), 1e-9)

    # ---- 2. channel mix of the last month, shrunk to the panel mix ------------------------------
    lm = d[d.m == B].groupby("style_color")[["units", "web_units", "retail", "other"]].sum().reindex(items).fillna(0)
    tot = lm[["web_units", "retail", "other"]].sum()
    pan = tot / tot.sum()
    sh = (lm[["web_units", "retail", "other"]] + P["CH_K"] * pan).div(lm["units"] + P["CH_K"], axis=0)

    # ---- 3. seasonal profile per segment, shrunk to group ---------------------------------------
    seg = it.seg.reindex(items).fillna("NA")
    grp = it.grp.reindex(items).fillna("CORE")
    occ = it.occasion.reindex(items).fillna("")
    out = {}
    age = (B.start_time - launch).dt.days / 30.44                     # age at start of base month
    K = P["SHRINK_K"]
    for h, t in enumerate(targets, start=1):
        lyt = t - 12
        level = (sh["web_units"] * factors["web"][lyt] + sh["retail"] * factors["ret"][lyt]
                 + sh["other"] * P["OTHER_LEVEL"] * P["OTHER_MONTHLY"] ** h)
        g_rel = grp.map(factors["grp_rel"][lyt]).fillna(1.0)
        s_rel = seg.map(factors["seg_rel"][lyt])
        n = seg.map(factors["seg_n"]).fillna(0).where(s_rel.notna(), 0)   # no profile -> group only
        rel = ((n * s_rel.fillna(0) + K * g_rel) / (n + K)).clip(*P["CAP"]) * grp.map(factors["anc"]).fillna(1.0)
        # ---- 4. lifecycle ramp for young items ----------------------------------------------------
        h2 = P["RAMP_H2"] if h >= 2 else 1.0
        ramp = np.select([age < 0, age < 1, age < 2], [P["RAMP0"] * h2, P["RAMP1"] * h2, P["RAMP2"]], 1.0)
        # ---- 5. occasion-tag calendar peaks -------------------------------------------------------
        occ_f = pd.Series(1.0, index=items)
        for tag, mult in OCCASION.items():
            occ_f[occ.str.contains(tag, regex=False)] *= mult.get(t.month, 1.0) / mult.get(B.month, 1.0)
        out[str(t)] = (b * t.days_in_month * level * rel * ramp * occ_f).clip(lower=0).fillna(0)
    return pd.DataFrame(out, index=items)


if __name__ == "__main__":
    it, di, st = load()
    items = pd.read_csv(DATA + "to_forecast.csv").style_color.tolist()
    origin = di.date.max()
    P_ = forecast(origin, items, it, di, st)
    pred = P_.stack().rename("pred").reset_index()
    pred.columns = ["style_color", "month", "pred"]
    pred["pred"] = pred.pred.astype(float).round(2)
    assert len(pred) == 4 * len(items) and np.isfinite(pred.pred).all() and (pred.pred >= 0).all()
    pred.to_csv("predictions.csv", index=False)
    print(f"origin {origin.date()}  items {len(items)}  rows {len(pred)}")
    print(pred.groupby("month").pred.sum().round(0).to_string())
