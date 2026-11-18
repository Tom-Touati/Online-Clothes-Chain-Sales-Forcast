"""Rule-based monthly unit forecast per style_color (version 13, snapshot 2026-02-28).
v13 vs v12: site plan Mar-Jun-26 re-based on February (web YoY 1.30-1.38, other at the Feb level: TikTok still
rising); June entries for every occasion tag / tier and Mar-Jun re-measured from a FEBRUARY-25 base (v12 read June
as a reversal to 1.0); W14 0.4 -> 0.5 (oracle-site backtest 0.3124 -> 0.3114, 8 of 9 origins better).  notes.md.

Forecast for item i, target month m (h = 1..4 months after the last data month B):
    pred = base_rate_i x days(m) x level_i(m) x season_seg(m) x ramp_i(h) x occasion_i(m)

  base_rate_i   recent units/day (last month blended with last 14 and last 91 days)
  level_i(m)    channel-weighted change from month B to month m: web = LY panel drift (LY like-for-like
                move / LY whole-site web move) x SITE_WEB_PD plan; other = same drift x OTHER_PD plan;
                retail = mature stores LFL move x LY retail panel drift, damped f**0.8, new / closed
                store adjustments
  season_seg(m) product-type (x sleeve) seasonal profile relative to the whole panel,
                same months one year earlier, shrunk to a WINTER / SUMMER / CORE group
  ramp_i(h)     young items keep ramping for ~2 months after launch
  occasion_i(m) occasion-tag and occasion-tier calendar indices, target month / base month

Only pandas/numpy, simple aggregations. Run:  python rules.py
"""
import numpy as np
import pandas as pd

DATA = "data/"
P = dict(
    W14=0.5, WLONG=0.2,        # base blend weights (last 14 days, last 91 days); rest = last month (v13 W14 0.4 -> 0.5: backtest
                               #  0.3124 -> 0.3114, 8 of 9 origins; 0.6 0.3109 but worse on the Oct-Jan holiday bases)
    RAMP0=1.6,                 # launched inside the base month (partial-month base); v11 see RAMP0_NEW
    RAMP1=1.4,                 # launched in the month before the base month
    RAMP2=1.15,                # launched two months before the base month
    RAMP_H2=1.1,               # extra growth from h=2 on for the RAMP0 / RAMP1 classes
    POW=0.0,                   # level-neutral regression to the mean of base rates: b * (b/mean)^-POW
    SHRINK_K=16,               # pseudo-items pulling a segment profile towards its group profile
    CH_K=20,                   # pseudo-units pulling an item's channel mix towards the panel mix
    RET_ANCHOR_PRE=0.25,       # retail level: weight of the "month before base" anchor
    OTHER_LEVEL=1.0,           # dropship + TikTok etc.: last year was channel onboarding, assume flat
    CAP=(0.15, 4.0),           # bounds on the seasonal relative factor
    NEW_STORE=True,            # stores opened inside the base month count at their run-rate
    PRE_ASYM=True,             # pre-anchor may only LOWER the retail factor (protects against a
                               # spiked base month, but does not assume a dip in the base month reverts)
    RET_POW=0.8,               # retail seasonal factor damped as f**RET_POW (2024 flagship was still ramping;
                               # v8 0.6 -> 0.8: Aug->Sep-25 stores fell as much as LY, LFL retail 0.45 vs 0.69 assumed)
    SEAS_ANCHOR=0.75,          # season-group level anchoring: rel x (LY group move / TY move)**a
                               # (v12 0.5 -> 0.75: oracle-site backtest 11 origins 0.3161 -> 0.3148, 7 of 11
                               #  origins better, Dec->Jan-26 h=1 0.249 -> 0.242; 0 / 0.25 worse)
    ANCHOR_WIN=2,              # months in the anchoring window (B-ANCHOR_WIN -> B)
    ANCHOR_LONG=True,          # v7: geometric mean of the short-window gap and the gap since January
    ADAPT_AMP=True,            # v8: SEAS_POW measured from the last AMP_WIN monthly WINTER/SUMMER moves,
    AMP_WIN=3,                 #     TY log-move / LY log-move (clipped AMP_CLIP); falls back to SEAS_POW
    AMP_CLIP=(0.7, 1.0),       # v12 floor 0.45 -> 0.7 (backtest 0.3218 -> 0.3161; floors 0.6 / 0.8 0.3176 / 0.3173,
                               #  0.9 0.3214; gains in the Jul-Sep-25 origins where amp read 0.45-0.6)
    SEAS_POW=0.8,              # fallback when ADAPT_AMP has no LY swing: rel**SEAS_POW (2025's seasonal
                               # swings are milder: Jun-Aug-25 group moves ~ LY's**0.5; v7 0.9 -> 0.8)
    DRIFT_ADJ=1.0,            # 2025 panel-vs-site drift is milder than 2024's (0.934 vs 0.906/month): LY drift
                               # x DRIFT_ADJ**h for web and other (v7 1.0 -> 1.01)
    SPIKE_X=2.5,               # v9: site "flash day" = base-month day with site units > SPIKE_X x the month's median
    SPIKE_CAP=1.0,             #     day; an item's units on it are capped at SPIKE_CAP x site ratio x its normal day
                               #     (v10 2.0 -> 1.0: Nov-24 BF-base allocation test WAPE 0.2274 -> 0.2219)
    RAMP_NOV=(1.35, 1.25, 1.1),  # v10: RAMP0/1/2 when the base month is November (BF lifts new arrivals; Nov-24
                               #     base: young items 0.84 of oracle at 1.6/1.4; test 0.2274 -> 0.2178 at 1.3/1.2/1.1)
    RAMP0_NEW=2.0,             # v11: RAMP0 for a non-November base (scoreboard: base-month launches act/pred
                               #     1.44 at h=1, ~1.7 at h=2-4 with 1.6; Dec-24 test 1.51 over Jan-Apr)
    AGE_FADE=0.93,             # v11: items 1-4 months old at the base month: x AGE_FADE**(h-1) (scoreboard act/pred
    AGE_FADE_RANGE=(1.0, 4.0), #     for that age band 0.99 / 0.96 / 0.87 / 0.77 at h=1..4; backtest 0.3186 -> 0.3152)
    HAL_SOFT=0.5,              # v9: Halloween index applied as idx**HAL_SOFT to items outside Party/Evening tier
    STORE_CLOSED_DAYS=5,      # store with no sale in the last N days of the base month = closed / pop-up
                               # (0 = off): its base-month units are removed from the retail level
)
# SITE PLAN: edikted.com web units/day and other-channel (drop + TikTok + wholesale) units/day of the WHOLE
# site by month.  Months <= 2026-02 are actuals (used when backtesting older origins), later months are the
# planner's assumptions.  Web factor = panel drift x SITE_WEB_PD[target] / site web/day of the base month.
SITE_WEB_PD = {"2025-01": 9593, "2025-02": 10831, "2025-03": 9660, "2025-04": 7906, "2025-05": 9102,
               "2025-06": 8832, "2025-07": 13902, "2025-08": 15638, "2025-09": 12204, "2025-10": 12751,
               "2025-11": 32130, "2025-12": 24203, "2026-01": 14050, "2026-02": 13827, "2026-03": 12800,
               "2026-04": 10900, "2026-05": 12000, "2026-06": 11500}
# v13 plan (research/r2_site.py).  Feb-26 actual web 13.83k/day (v12 plan 14.5k, YoY 1.28 vs LY's deep-promo Feb;
# disc 0.608 vs 0.606); 2-week aligned YoY Feb 15-28 1.41 (LY's early-March dip), Feb 1-21 1.20-1.22.  Monthly YoY
# Sep-Feb 1.42 / 1.60 / 1.65 / 1.37 / 1.47 / 1.28 -> Mar 12.8k (YoY 1.33 on LY 9.66k), Apr 10.9k (1.38: LY April weak,
# Easter Apr 20 LY vs Apr 5 TY), May 12.0k (1.32, Memorial Day in both), Jun 11.5k (1.30).  Other Feb 4.18k/day (v12
# plan 3.3k, second month +27% over plan): TikTok 3.05k/day (Sep->Feb 1.27/1.50/2.30/2.38/2.55/3.05k), wholesale 1.13k
# (LY Mar-Jun wholesale flat 1.06-1.25k); Feb 15-28 other 4.41k/day -> Mar 4.2k, Apr 3.95k, May 3.95k, Jun 3.8k (a
# mild decline only; LY TikTok halved Feb->Apr but TY has risen against LY's seasonality for 5 months).
# (v12 plan Feb-May from January: web YoY 1.34-1.42, other 3.3 -> 3.15k with LY ex-Flip seasonality; Jan-26 actual
#  web 14.05k at disc 0.611, other 3.85k.  Details in notes.md History.)
OTHER_PD = {"2025-01": 2472, "2025-02": 2213, "2025-03": 1957, "2025-04": 1770, "2025-05": 1985,
            "2025-06": 1812, "2025-07": 3099, "2025-08": 3127, "2025-09": 2951, "2025-10": 3390,
            "2025-11": 6146, "2025-12": 7367, "2026-01": 3849, "2026-02": 4184, "2026-03": 4200, "2026-04": 3950,
            "2026-05": 3950, "2026-06": 3800}
# Occasion tags with a calendar peak: demand index by calendar month (default 1.0); the multiplier is
# idx[target month] / idx[base month], so a peak already inside the base month is not counted twice.
OCCASION = {"4th of July": {5: 1.035, 6: 1.10, 7: 1.2, 8: 1.1, 9: 0.97, 10: 0.9, 11: 0.95, 12: 1.02, 1: 1.02, 2: 1.02, 3: 1.015, 4: 0.995},
            "Graduation": {4: 0.86, 5: 0.826, 6: 0.75, 7: 0.75, 8: 0.9, 9: 0.85, 10: 0.8, 11: 0.72, 12: 0.75, 1: 0.75, 2: 0.86, 3: 0.873},
            "Homecoming": {4: 1.08, 5: 1.056, 6: 0.906, 7: 1.05, 8: 1.3, 9: 2.6, 10: 1.75, 11: 1.4, 12: 1.25, 1: 1.4, 2: 1.2, 3: 1.158},
            "Halloween": {8: 1.1, 9: 1.4, 10: 2.4, 11: 0.9, 12: 0.88, 1: 0.862, 2: 0.862, 3: 0.819, 4: 0.81, 5: 0.806, 6: 0.815},
            "New Year's Eve": {9: 1.1, 10: 1.25, 11: 1.125, 12: 1.18, 1: 1.18, 2: 1.18, 3: 1.121, 4: 1.103, 5: 1.08, 6: 1.05},
            "Valentine's": {12: 0.96, 1: 0.96, 2: 0.94, 3: 0.893, 4: 0.902, 5: 0.879, 6: 0.884},
            "St. Patrick": {2: 1.1, 3: 1.265, 4: 1.067, 5: 0.957, 6: 0.979}}
# v13 Mar-Jun values re-measured from a FEBRUARY-25 base (research/r5_tags.py): items launched > 30 days before the
# base month vs a segment x month oracle, shrunk halfway to 1 and chained onto the Feb index (raw Mar / Apr / May /
# Jun-25):  4th of July (13) 0.99 0.95 1.03 1.14 (Jun 1.3 -> 1.10: LY's June lift was ~1.14 vs segment, Jan base
# 1.14 too);  Graduation (13) 1.03 1.00 0.92 0.74 (v12 Jun 1.0 = x1.16 from Feb was a default, LY fell);
# Homecoming (15) 0.93 0.80 0.76 0.51;  Halloween (55) 0.90 0.88 0.87 0.89;  NYE (22) 0.90 0.87 0.83 0.78;
# Valentine's (50) 0.90 0.92 0.87 0.88;  St. Patrick (6) 1.30 0.94 0.74 0.78.  v12 had no June entries -> June
# read as a reversal to 1.0 (Halloween x1.16, Valentine's x1.06, NYE x0.85 from Feb).  TY Jan->Feb-26 vs segment
# oracle (research/r5b_tags_ty.py): Graduation 1.17 (LY 1.28), Halloween 0.98 (1.00), Valentine's 0.95 (0.98),
# NYE 1.04 (1.02), 4th of July 1.13, St. Patrick 1.43 -> LY shapes still hold.
# (v12 Jan-25-base raw values for Feb-May and v11 Dec-24-base values are in notes.md History.)
OCC_EXCLUDE = {"New Year's Eve": "Homecoming"}   # tag not applied when the item also has this tag
# Occasion-tier calendar indices (tier from item_attributes; multiplier idx[target] / idx[base]); Party Sep/Oct
#   step from v9; Dec/Jan Evening and Resort-Festival values from the v11 Dec-24-base measurement.
TIER_IDX = {"Party": {9: 1.1, 10: 1.1, 3: 0.99, 4: 0.96, 5: 0.915, 6: 0.89},
            "Evening": {12: 1.12, 1: 1.12, 2: 1.159, 3: 1.14, 4: 1.08, 5: 1.06, 6: 1.0},
            "Resort-Festival": {12: 1.05, 1: 1.155, 2: 1.236, 3: 1.23, 4: 1.13, 5: 1.14, 6: 1.21},
            "Loungewear": {1: 0.98, 2: 0.92, 3: 0.892, 4: 0.883, 5: 0.879, 6: 0.837},
            "Basics": {2: 1.03, 3: 1.03, 4: 1.04, 5: 1.13, 6: 1.15}}
# v13 Mar-Jun tiers from the Feb-25 base (untagged items vs segment oracle, shrunk halfway, chained onto Feb; raw
#   Mar / Apr / May / Jun): Party (58) 0.98 0.92 0.83 0.78;  Resort-Festival (10) 0.99 0.83 0.84 0.96;  Loungewear
#   (9) 0.94 0.92 0.91 0.82;  Basics (27) 1.00 1.02 1.26 1.30;  Evening (26) 0.86 0.79 0.76 0.60 but TY Jan->Feb-26
#   Evening held (0.98 vs LY 0.85) and v12's Jan-base read was 1.07-1.12 -> Evening set between (Mar 1.14 vs v12
#   1.187).  June entries are new (v12 none -> Party x1.0, Loungewear x1.09, Resort-Festival x0.81 from Feb).
# (v12 Jan-25-base tier values: notes.md History.  Basics re-added in v12: TY under-forecast Nov-Jan.)
TIER_SKIP_TAGS = {"Party": "Halloween|Valentine's", "Evening": "Halloween|New Year's Eve|Valentine's",
                  "Resort-Festival": "Halloween|New Year's Eve|Valentine's"}   # tagged items keep their tag index
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


def seasonal_factors(it, di, st, B, targets, items_for_factors=None, ret_anchor_pre=None):
    """Factors for targets relative to base month B, measured on the same months one year earlier."""
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
    # v8 seasonal amplitude: this year's recent monthly WINTER / SUMMER moves (rel. to panel) vs the same months
    # last year: amp = sum(log TY move x sign(log LY move)) / sum(|log LY move|) over the last AMP_WIN months
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
    # web / other level = LY panel drift (panel items' LFL per-day move / whole-site per-day move; removes 2024's
    # growth and the July-24 discount-regime jump) x this year's explicit site plan (SITE_WEB_PD / OTHER_PD).
    # (The v1-v5 LFL web anchors and the July-24 regime fix were retired: W_DEFL had been 1.0 since v5.)
    pre = B - 1
    have = set(di.m.unique())
    if not {pre - 12, pre, B} <= have:                                 # pre-anchor needs 13+ months of data
        ret_anchor_pre, chain_months = 0.0, ly
    else:
        chain_months = [pre - 12] + ly
    def chain(col):                                                    # LFL chain of one channel's units
        X = di[di.m.isin(chain_months)].pivot_table(index="style_color", columns="m", values=col,
                                                    aggfunc="sum").reindex(columns=chain_months).fillna(0)
        if items_for_factors is not None:
            X = X[X.index.isin(items_for_factors)]
        c, _ = lfl_chain(X, it.launch, chain_months, pd.Series("ALL", index=X.index))
        c = c.loc["ALL"]
        c[chain_months[0]] = 1.0
        return c
    ch_raw, rch = chain("web_units"), chain("retail")
    cur_ret = 1.0
    if chain_months[0] != lyB:                                         # this year's per-day (B-1 -> B) retail LFL
        Xp = di[di.m.isin([pre, B])].pivot_table(index="style_color", columns="m", values="retail",
                                                 aggfunc="sum").reindex(columns=[pre, B]).fillna(0)
        c, _ = lfl_chain(Xp, it.launch, [pre, B], pd.Series("ALL", index=Xp.index))
        cur_ret = c.loc["ALL", B] * pre.days_in_month / B.days_in_month
    site_pd = di[di.m.isin(ly)].groupby("m").web_units.sum() / di[di.m.isin(ly)].groupby("m").date.nunique()
    drift = pd.Series({t: ch_raw[t] / ch_raw[lyB] * lyB.days_in_month / t.days_in_month
                       / (site_pd[t] / site_pd[lyB]) * P["DRIFT_ADJ"] ** h
                       for h, t in enumerate(ly[1:], start=1)})        # panel vs site, per day, LY (adj.)
    dB = di[di.m == B]
    nB = dB.date.nunique()
    web_B, oth_B = dB.web_units.sum() / nB, dB.other.sum() / nB         # this year's base-month site per day
    web = pd.Series({t: drift[t] * SITE_WEB_PD.get(str(t + 12), web_B) / web_B for t in ly[1:]})
    other = pd.Series({t: drift[t] * OTHER_PD.get(str(t + 12), oth_B) / oth_B for t in ly[1:]})
    # retail level (v11): MATURE stores (open > 60 days before LY's base month), like-for-like units/day, x LY
    # retail panel drift (panel items' retail LFL / all-store retail).  v1-v10 used the flagship deflated by web
    # growth: Nov->Dec-25 flagship LY 1.30 vs mature stores 1.12 (TY 1.10); v10 factor 1.32 vs panel actual 0.91.
    # Pre-anchor as before (Feb-24 was a retail promo month: retail discount 0.38 vs 0.18 in Mar-24).
    st = st.assign(m=st.date.dt.to_period("M"))
    opened = st.groupby("store").date.min()
    mature = opened[opened < lyB.start_time - pd.Timedelta(days=60)].index
    nd = st.groupby("m").date.nunique()
    sm = st[st.store.isin(mature)].groupby("m").units.sum() / nd
    sa = st.groupby("m").units.sum() / nd
    def defl(t, a):                                                    # mature-store move x retail panel drift
        pan = rch[t] / rch[a] * a.days_in_month / t.days_in_month
        return (sm.get(t, np.nan) / sm.get(a, np.nan)) * pan / (sa.get(t, np.nan) / sa.get(a, np.nan))
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
    return dict(seas_pow=seas_pow, anc=anc, drift=drift, other=other, seg_rel=seg_rel, seg_n=seg_n, grp_rel=grp_rel,
                web=web, ret=(ret ** P["RET_POW"]) * ret_adj, ly=ly, cur_ret=cur_ret, ret_adj=ret_adj)


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
        pos, tot_b = b > 0, b.sum()
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
                 + sh["other"] * P["OTHER_LEVEL"] * factors["other"][lyt])
        g_rel = grp.map(factors["grp_rel"][lyt]).fillna(1.0)
        s_rel = seg.map(factors["seg_rel"][lyt])
        n = seg.map(factors["seg_n"]).fillna(0).where(s_rel.notna(), 0)   # no profile -> group only
        rel = ((n * s_rel.fillna(0) + K * g_rel) / (n + K)).clip(*P["CAP"]) ** factors["seas_pow"]
        rel = rel * grp.map(factors["anc"]).fillna(1.0)
        # ---- 4. lifecycle ramp for young items ----------------------------------------------------
        h2 = P["RAMP_H2"] if h >= 2 else 1.0
        r0, r1, r2 = P["RAMP_NOV"] if B.month == 11 else (P["RAMP0_NEW"], P["RAMP1"], P["RAMP2"])
        ramp = np.select([age < 0, age < 1, age < 2], [r0 * h2, r1 * h2, r2], 1.0)
        # v11 lifecycle fade: items 1-4 months old lose AGE_FADE per month beyond h=1
        lo, hi = P["AGE_FADE_RANGE"]
        fade = np.where((age >= lo) & (age < hi), P["AGE_FADE"] ** (h - 1), 1.0)
        ramp = ramp * fade
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
        # occasion-tier calendar indices (TIER_IDX: Party Sep-Oct step, v10 Evening / Basics / Resort-Festival)
        for tr, mult in TIER_IDX.items():
            has = tier.str.contains(tr, regex=False)
            if tr in TIER_SKIP_TAGS:
                has &= ~occ.str.contains(TIER_SKIP_TAGS[tr])
            occ_f[has] *= mult.get(t.month, 1.0) / mult.get(B.month, 1.0)
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
