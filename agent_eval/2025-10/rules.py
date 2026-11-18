"""Rule-based monthly unit forecast per style_color (version 9, snapshot 2025-10-31).
v9 vs v8: site plan re-based on October (web 12.75k/day actual vs 11.2k planned) and extended to Feb-26;
DRIFT_ADJ 1.01 -> 1.0; flash-day cap on item base rates (Oct 14 one-day event); occasion indices re-checked from
an October base with February added (Halloween post-peak drop, softer outside party/evening wear); Party-tier
post-October step down.  Seasonal amplitude still measured (ADAPT_AMP: 0.67 now, Sep->Oct moved at LY's full
amplitude).  See notes.md.

Forecast for item i, target month m (h = 1..4 months after the last data month B):
    pred = base_rate_i x days(m) x level_i(m) x season_seg(m) x ramp_i(h) x occasion_i(m)

  base_rate_i   recent units/day (last month blended with last 14 and last 91 days)
  level_i(m)    channel-weighted change from month B to month m: web = LY panel drift (LY like-for-like
                move / LY whole-site web move) x SITE_WEB_PD plan; other = same drift x OTHER_PD plan;
                retail = flagship store per day deflated for panel growth, damped f**0.6, new / closed
                store adjustments
  season_seg(m) product-type (x sleeve) seasonal profile relative to the whole panel,
                same months one year earlier, shrunk to a WINTER / SUMMER / CORE group
  ramp_i(h)     young items keep ramping for ~2 months after launch
  occasion_i(m) occasion-tag calendar index, target month / base month (4th of July, Graduation,
                Homecoming, Halloween, New Year's Eve)

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
    SHRINK_K=16,               # pseudo-items pulling a segment profile towards its group profile
    CH_K=20,                   # pseudo-units pulling an item's channel mix towards the panel mix
    ANCHOR_PRE=0.4,            # web level: weight of the "month before base" anchor
    RET_ANCHOR_PRE=0.25,       # retail level: weight of the "month before base" anchor
    OTHER_LEVEL=1.0,           # dropship + TikTok etc.: last year was channel onboarding, assume flat
    CAP=(0.15, 4.0),           # bounds on the seasonal relative factor
    NEW_STORE=True,            # stores opened inside the base month count at their run-rate
    PRE_ASYM=True,             # pre-anchor may only LOWER the web/retail factor (protects against a
                               # spiked base month, but does not assume a dip in the base month reverts)
    W_DEFL=1.0,                # web level: weight of the "deflated" factor = LY LFL / LY site web (per day)
    OTHER_MONTHLY=0.9,         # dropship + TikTok etc.: per-month factor, compounded over h (OTHER_PATH off)
    OTHER_PATH=True,           # v6: other = panel drift x assumed other-channel site path (OTHER_PD)
    RET_POW=0.8,               # retail seasonal factor damped as f**RET_POW (2024 flagship was still ramping;
                               # v8 0.6 -> 0.8: Aug->Sep-25 stores fell as much as LY, LFL retail 0.45 vs 0.69 assumed)
    SEAS_ANCHOR=0.5,           # season-group level anchoring: rel x (LY group move / TY move)**a
    ANCHOR_WIN=2,              # months in the anchoring window (B-ANCHOR_WIN -> B)
    ANCHOR_LONG=True,          # v7: geometric mean of the short-window gap and the gap since January
    ADAPT_AMP=True,            # v8: SEAS_POW measured from the last AMP_WIN monthly WINTER/SUMMER moves,
    AMP_WIN=3,                 #     TY log-move / LY log-move (clipped AMP_CLIP); falls back to SEAS_POW
    AMP_CLIP=(0.45, 1.0),
    SEAS_POW=0.8,              # fallback when ADAPT_AMP has no LY swing: rel**SEAS_POW (2025's seasonal
                               # swings are milder: Jun-Aug-25 group moves ~ LY's**0.5; v7 0.9 -> 0.8)
    DRIFT_ADJ=1.0,            # 2025 panel-vs-site drift is milder than 2024's (0.934 vs 0.906/month): LY drift
                               # x DRIFT_ADJ**h for web and other (v7 1.0 -> 1.01)
    SPIKE_X=2.5,               # v9: site "flash day" = base-month day with site units > SPIKE_X x the month's median
    SPIKE_CAP=2.0,             #     day; an item's units on it are capped at SPIKE_CAP x site ratio x its normal day
                               #     (Oct 14-25: 45k web units, 145k cancelled; S12749 13x its normal day)
    TIER_IDX={9: 1.1, 10: 1.1},  # v9: 'Party' occasion tier (not Halloween-tagged) calendar index
    HAL_SOFT=0.5,              # v9: Halloween index applied as idx**HAL_SOFT to items outside Party/Evening tier
    STORE_CLOSED_DAYS=5,       # store with no sale in the last N days of the base month = closed / pop-up
                               # (0 = off): its base-month units are removed from the retail level
)
# SITE PLAN: edikted.com web units/day and other-channel (drop + TikTok + wholesale) units/day of the WHOLE
# site by month.  Months <= 2025-09 are actuals (used when backtesting older origins), later months are the
# planner's assumptions.  Web factor = panel drift x SITE_WEB_PD[target] / site web/day of the base month.
SITE_WEB_PD = {"2025-01": 9593, "2025-02": 10831, "2025-03": 9660, "2025-04": 7906, "2025-05": 9102,
               "2025-06": 8832, "2025-07": 13902, "2025-08": 15638, "2025-09": 12204, "2025-10": 12751,
               "2025-11": 26500, "2025-12": 23300, "2026-01": 11500, "2026-02": 12000}
# v9 plan (research/r1_site.py), from Oct-25 12.75k/day actual (v8 plan 11.2k).  Oct contains a one-day flash
# event on Oct 14 (45.0k web units kept, 145k cancelled); ex-spike Oct ~11.9k/day.  Weekly web/day Oct 15.1 /
# 19.6 / 11.4 / 7.05k (last week = the same pre-November lull as 2024: 4.1k/day).  YoY web/day Jun 2.0, Jul 1.80,
# Aug 1.65, Sep 1.42, Oct 1.60 (1.50 ex-spike); sale weeks 1.43-1.75x LY, quiet weeks 1.47-1.72x.
#   Nov 26.5k  YoY 1.36 on Nov-24 19.5k (Black Friday Nov 28 vs Nov 29; BF week LY 54k/day); v8 had 24.5k
#   Dec 23.3k  YoY 1.32 on Dec-24 17.7k (Cyber Monday Dec 1 vs Dec 2 -- both in December)
#   Jan 11.5k  YoY 1.20 on Jan-25 9.6k (Jan-Feb-25 ran at disc 0.60-0.61, already in the deep-discount regime)
#   Feb 12.0k  YoY 1.11 on Feb-25 10.8k (no Valentine's lift seen in item data)
# Other (ex-Flip 2024 per day: Oct 2.43k, Nov 3.85, Dec 5.94, Jan 2.28, Feb 2.05; YoY Oct 1.39 -- Pacsun 1.52x,
# TikTok 1.38x, Macy's 1.9x), YoY decaying 1.17 -> 1.06 -> 1.18 -> 1.17: Nov 4.5k, Dec 6.3k, Jan 2.7k, Feb 2.4k.
OTHER_PD = {"2025-01": 2472, "2025-02": 2213, "2025-03": 1957, "2025-04": 1770, "2025-05": 1985,
            "2025-06": 1812, "2025-07": 3099, "2025-08": 3127, "2025-09": 2951, "2025-10": 3390,
            "2025-11": 4500, "2025-12": 6300, "2026-01": 2700, "2026-02": 2400}
# Occasion tags with a calendar peak: demand index by calendar month (default 1.0); the multiplier is
# idx[target month] / idx[base month], so a peak already inside the base month is not counted twice.
OCCASION = {"4th of July": {5: 1.15, 6: 1.3, 7: 1.2, 8: 1.1, 9: 0.97, 10: 0.9, 11: 0.95, 12: 0.95, 1: 0.95, 2: 0.95},
            "Graduation": {4: 1.1, 5: 1.2, 6: 1.0, 7: 1.0, 8: 0.9, 9: 0.85, 10: 0.8, 11: 0.72, 12: 0.7, 1: 0.8, 2: 0.9},
            "Homecoming": {5: 1.2, 6: 0.95, 7: 1.05, 8: 1.3, 9: 2.6, 10: 1.75, 11: 1.4, 12: 1.31, 1: 1.75, 2: 1.58},
            "Halloween": {8: 1.1, 9: 1.4, 10: 2.4, 11: 0.9, 12: 0.92, 1: 1.0, 2: 1.15},
            "New Year's Eve": {9: 1.1, 10: 1.25, 11: 1.125, 12: 1.25, 1: 1.19, 2: 1.25}}
# v9 re-check from an OCTOBER-24 base (research/r3_factors.py, r5_attr.py, r6_seg.py), tag vs untagged, Nov/Dec/Jan/Feb:
#   Halloween (n=38) same-group 0.35 0.36 0.39 0.45, same-segment 0.39 0.43 0.47 0.49; TY Sep->Oct peak 1.83 vs
#     LY 2.09 (Oct-25 scoreboard rel 0.89 -> Oct index 2.6 -> 2.4); Halloween r14/r31 0.94 vs 0.75 for the rest
#     (the 14-day blend lifts their base ~8%) -> from Oct x0.375 / 0.383 / 0.417 / 0.48.  Party/Evening-tier
#     Halloween items fall to ~0.4, other-tier ones only to ~0.75 (n=18 / 20) -> HAL_SOFT.
#   Homecoming (n=12) 0.65-0.86 / 0.58-0.80 / 1.26-1.35 / 0.85-0.96, shrunk -> x0.80 / 0.75 / 1.0 / 0.9
#   New Year's Eve w/o Homecoming (n=7) 0.54-0.79 / 0.70-1.13 / 0.87-0.94 / 1.19-1.31 (no Dec lift; Oct-25
#     scoreboard rel 0.81) -> x0.90 / 1.0 / 0.95 / 1.0
#   Graduation 0.78 0.79 1.1 1.33 -> Feb x1.125;  4th of July 1.21 1.24 1.13 1.25 -> x1.06;  Valentine's
#     (n=26) 1.07 1.02 1.02 0.90: no pre-Valentine lift in Jan/Feb -> still not used.
OCC_EXCLUDE = {"New Year's Eve": "Homecoming"}   # tag not applied when the item also has this tag
# Late July 2024 edikted.com moved into a permanent deep-discount regime (web discount 0.45 -> 0.57,
# LFL web/day x2.3 vs June).  2025 is already in that regime, so last year's Jun->Jul web move is
# replaced by the discount-matched part of July (Jul 8-21: discount 0.45 vs June 0.46).
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
    it["tier"] = at["occasion_tier"].reindex(it.index).fillna("")
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
    tyw = [B - k for k in range(P["ANCHOR_WIN"], -1, -1)]
    lyw = [t - 12 for t in tyw]
    def cum_rel(ms):                                                    # group LFL move rel. to panel
        Uq = di[di.m.isin(ms)].pivot_table(index="style_color", columns="m", values="units",
                                           aggfunc="sum").reindex(columns=ms).fillna(0)
        g, _ = lfl_chain(Uq, it.launch, ms, it.grp)
        a, _ = lfl_chain(Uq, it.launch, ms, pd.Series("ALL", index=Uq.index))
        return g[ms[-1]] / a.loc["ALL", ms[-1]]
    # v8 seasonal amplitude: how large this year's recent monthly WINTER / SUMMER moves (relative to the
    # panel) were compared with the same months last year, as a ratio of log-moves:
    #   amp = sum(log TY move x sign(log LY move)) / sum(|log LY move|)  over the last AMP_WIN months.
    # Jul->Oct-25 WINTER 1.13 / 1.21 / 1.24 vs LY 1.34 / 1.40 / 1.27, SUMMER 0.90 / 0.99 / 0.88 vs 0.835 / 0.99 /
    # 0.90 -> amp 0.67 at this origin (0.51 a month ago; Sep->Oct alone ~1.0).
    seas_pow = P["SEAS_POW"]
    if P["ADAPT_AMP"]:
        num = den = 0.0
        for k in range(P["AMP_WIN"]):
            a, b = B - k - 1, B - k
            if not {a - 12, b - 12, a, b} <= set(di.m.unique()):
                continue
            ty, ly_ = np.log(cum_rel([a, b])), np.log(cum_rel([a - 12, b - 12]))
            for g in ["WINTER", "SUMMER"]:
                if g in ty.index and g in ly_.index:
                    num += ty[g] * np.sign(ly_[g])
                    den += abs(ly_[g])
        if den > 0.15:                                                  # LY swing too small -> keep default
            seas_pow = float(np.clip(num / den, *P["AMP_CLIP"]))
    if P["SEAS_ANCHOR"] != 0 and set(lyw) <= set(di.m.unique()):
        gap = cum_rel(lyw) / cum_rel(tyw)
        # v7: also the LEVEL gap since January (same calendar window both years); short-window moves alone
        # read a gap that is already closing (spring: winter fell less; summer: it then rose less) as new
        jan = pd.Period(f"{B.year}-01", "M")
        if P["ANCHOR_LONG"] and B.month - 1 > P["ANCHOR_WIN"]:
            tyl = [jan + k for k in range(B.month)]
            gap = np.sqrt(gap * cum_rel([t - 12 for t in tyl]) / cum_rel(tyl))
        anc = (gap ** P["SEAS_ANCHOR"]).clip(0.7, 1.4).reindex(grp_rel.index).fillna(1.0)
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
    drift = pd.Series({t: ch_raw[t] / ch_raw[lyB] * lyB.days_in_month / t.days_in_month
                       / (site_pd[t] / site_pd[lyB]) * P["DRIFT_ADJ"] ** h
                       for h, t in enumerate(ly[1:], start=1)})        # panel vs site, per day, LY (adj.)
    dB = di[di.m == B]
    nB = dB.date.nunique()
    web_B, oth_B = dB.web_units.sum() / nB, dB.other.sum() / nB         # this year's base-month site per day
    defl_w = pd.Series({t: drift[t] * SITE_WEB_PD.get(str(t + 12), web_B) / web_B for t in ly[1:]})
    other = pd.Series({t: drift[t] * OTHER_PD.get(str(t + 12), oth_B) / oth_B for t in ly[1:]})
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
    # pop-up / closed stores (e.g. NYR sold only Jun 19-22, 2025): no sale in the last N days of the base
    # month -> they contribute nothing in the target months, so their base-month units are removed
    closed = []
    if P["STORE_CLOSED_DAYS"] > 0 and len(sB):
        lastsale = sB.groupby("store").date.max()
        closed = list(lastsale[lastsale <= sB.date.max() - pd.Timedelta(days=P["STORE_CLOSED_DAYS"])].index)
        new = [s for s in new if s not in closed]
    if P["NEW_STORE"] and len(new) and sB.units.sum() > 0:
        last = sB[sB.date > sB.date.max() - pd.Timedelta(days=14)]
        run = sum(last[last.store == s].groupby("date").units.sum().median() * B.days_in_month for s in new)
        ret_adj = (sB.units.sum() - sB[sB.store.isin(new)].units.sum() + run) / sB.units.sum()
    if closed and sB.units.sum() > 0:
        ret_adj *= 1 - sB[sB.store.isin(closed)].units.sum() / sB.units.sum()
    return dict(seas_pow=seas_pow, anc=anc, defl_w=defl_w, drift=drift, other=other, seg_rel=seg_rel, seg_n=seg_n, grp_rel=grp_rel, web=web,
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
    raw_month = (D.iloc[:, -B.days_in_month:].sum(axis=1) / np.minimum(live, B.days_in_month)).sum()
    # v9 flash-day cap: on a base-month day where the whole site sold > SPIKE_X x its median day, an item may
    # sell at most SPIKE_CAP x (site ratio) x its own normal day; the excess is redistributed level-neutrally
    # (the site level web_B still contains the flash day, so totals stay consistent with the site plan)
    sday = d[d.m == B].groupby("date").units.sum()
    flash = sday[sday > P["SPIKE_X"] * sday.median()]
    if len(flash):
        cols = [c for c in D.columns[-B.days_in_month:] if c not in flash.index]
        normal = D[cols].mean(axis=1)
        live_ok = (live >= 14)
        for day, v in flash.items():
            if day in D.columns:
                cap = P["SPIKE_CAP"] * v / sday.median() * normal
                D.loc[live_ok, day] = np.minimum(D.loc[live_ok, day], cap[live_ok])
    def rate(n):
        return D.iloc[:, -n:].sum(axis=1) / np.minimum(live, n)
    r_month = rate(B.days_in_month)
    b = (1 - P["W14"]) * r_month + P["W14"] * rate(14)
    b = b * r_month.sum() / max(b.sum(), 1e-9)                       # blend is level-neutral
    r91 = rate(91)
    b = (1 - P["WLONG"]) * b + P["WLONG"] * r91 * r_month.sum() / max(r91.sum(), 1e-9)
    b = b * raw_month / max(b.sum(), 1e-9)                           # keep the pre-cap level (no-op w/o flash)
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
    tier = it.tier.reindex(items).fillna("")
    party_ev = tier.str.contains("Party|Evening")
    out = {}
    age = (B.start_time - launch).dt.days / 30.44                     # age at start of base month
    K = P["SHRINK_K"]
    for h, t in enumerate(targets, start=1):
        lyt = t - 12
        level = (sh["web_units"] * factors["web"][lyt] + sh["retail"] * factors["ret"][lyt]
                 + sh["other"] * P["OTHER_LEVEL"] * (factors["other"][lyt] if P["OTHER_PATH"]
                                                     else P["OTHER_MONTHLY"] ** h))
        g_rel = grp.map(factors["grp_rel"][lyt]).fillna(1.0)
        s_rel = seg.map(factors["seg_rel"][lyt])
        n = seg.map(factors["seg_n"]).fillna(0).where(s_rel.notna(), 0)   # no profile -> group only
        rel = ((n * s_rel.fillna(0) + K * g_rel) / (n + K)).clip(*P["CAP"]) ** factors["seas_pow"]
        rel = rel * grp.map(factors["anc"]).fillna(1.0)
        # ---- 4. lifecycle ramp for young items ----------------------------------------------------
        h2 = P["RAMP_H2"] if h >= 2 else 1.0
        ramp = np.select([age < 0, age < 1, age < 2], [P["RAMP0"] * h2, P["RAMP1"] * h2, P["RAMP2"]], 1.0)
        # ---- 5. occasion-tag calendar peaks -------------------------------------------------------
        occ_f = pd.Series(1.0, index=items)
        for tag, mult in OCCASION.items():
            has = occ.str.contains(tag, regex=False)
            if tag in OCC_EXCLUDE:
                has &= ~occ.str.contains(OCC_EXCLUDE[tag], regex=False)
            f_occ = mult.get(t.month, 1.0) / mult.get(B.month, 1.0)
            if tag == "Halloween":                                  # v9: softer swing outside party/evening wear
                occ_f[has & party_ev] *= f_occ
                occ_f[has & ~party_ev] *= f_occ ** P["HAL_SOFT"]
            else:
                occ_f[has] *= f_occ
        # v9: Party-tier items (not Halloween-tagged) are elevated in the Sep-Oct party season: LY from an Oct
        # base they fell to 0.78 / 0.76 / 0.82 / 0.82 of same-segment items in Nov-Feb (n=77); shrunk to 0.91
        pt = tier.str.contains("Party") & ~occ.str.contains("Halloween", regex=False)
        occ_f[pt] *= P["TIER_IDX"].get(t.month, 1.0) / P["TIER_IDX"].get(B.month, 1.0)
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
