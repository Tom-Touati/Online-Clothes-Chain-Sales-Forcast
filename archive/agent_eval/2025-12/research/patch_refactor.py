"""One-off refactor of rules.py: remove dead code paths (W_DEFL=1 web anchors, July-24 regime fix, OTHER_PATH
switch, flagship retail path).  Output must be unchanged (checked against pred_before_refactor.csv)."""
s = open("rules.py", encoding="utf-8").read()


def rep(old, new):
    global s
    assert old in s, old[:60]
    s = s.replace(old, new)


def cut(a_txt, b_txt, new):
    global s
    a, b = s.index(a_txt), s.index(b_txt)
    s = s[:a] + new + s[b:]


cut("    # web level: LFL chain of edikted.com units", "    def ret_pre(t):", '''    # web / other level = LY panel drift (panel items' LFL per-day move / whole-site per-day move; removes 2024's
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
''')
rep('''    ret_adj = 1.0
    sB = st[st.m == B]
    opened = st.groupby("store").date.min()
    new = opened''', '''    ret_adj = 1.0
    sB = st[st.m == B]
    new = opened''')
rep('''    return dict(seas_pow=seas_pow, anc=anc, defl_w=defl_w, drift=drift, other=other, seg_rel=seg_rel, seg_n=seg_n, grp_rel=grp_rel, web=web,
                ret=(ret ** P["RET_POW"]) * ret_adj, ly=ly,
                a_feb=a_feb, a_pre=a_pre, cur_pd=cur_pd, cur_ret=cur_ret, ret_adj=ret_adj)''',
    '''    return dict(seas_pow=seas_pow, anc=anc, drift=drift, other=other, seg_rel=seg_rel, seg_n=seg_n, grp_rel=grp_rel,
                web=web, ret=(ret ** P["RET_POW"]) * ret_adj, ly=ly, cur_ret=cur_ret, ret_adj=ret_adj)''')
rep('''def seasonal_factors(it, di, st, B, targets, items_for_factors=None, anchor_pre=None, ret_anchor_pre=None):
    """Factors for targets relative to base month B, measured on the same months one year earlier."""
    anchor_pre = P["ANCHOR_PRE"] if anchor_pre is None else anchor_pre
''', '''def seasonal_factors(it, di, st, B, targets, items_for_factors=None, ret_anchor_pre=None):
    """Factors for targets relative to base month B, measured on the same months one year earlier."""
''')
cut("def web_regime_fix", "def seasonal_factors", "")
cut("# Late July 2024 edikted.com moved", "WINTER_TYPES = ", "")
cut("    # v8 seasonal amplitude: how large", '    seas_pow = P["SEAS_POW"]',
    '''    # v8 seasonal amplitude: this year's recent monthly WINTER / SUMMER moves (rel. to panel) vs the same months
    # last year: amp = sum(log TY move x sign(log LY move)) / sum(|log LY move|) over the last AMP_WIN months
''')
rep('''    ANCHOR_PRE=0.4,            # web level: weight of the "month before base" anchor
''', "")
rep('''    W_DEFL=1.0,                # web level: weight of the "deflated" factor = LY LFL / LY site web (per day)
    OTHER_MONTHLY=0.9,         # dropship + TikTok etc.: per-month factor, compounded over h (OTHER_PATH off)
    OTHER_PATH=True,           # v6: other = panel drift x assumed other-channel site path (OTHER_PD)
''', "")
rep('''    RET_MATURE=True,           # v11: retail seasonal move from stores open > 60 days before LY base month (LFL),
                               #     x LY retail panel drift (panel retail LFL / all-store retail), not flagship
''', "")
rep("    PRE_ASYM=True,             # pre-anchor may only LOWER the web/retail factor",
    "    PRE_ASYM=True,             # pre-anchor may only LOWER the retail factor")
rep('''                 + sh["other"] * P["OTHER_LEVEL"] * (factors["other"][lyt] if P["OTHER_PATH"]
                                                     else P["OTHER_MONTHLY"] ** h))''',
    '''                 + sh["other"] * P["OTHER_LEVEL"] * factors["other"][lyt])''')
rep('''                retail = flagship store per day deflated for panel growth, damped f**0.6, new / closed
                store adjustments''', '''                retail = mature stores LFL move x LY retail panel drift, damped f**0.8, new / closed
                store adjustments''')
rep('''1-4-month-old items (AGE_FADE) from the 2025 scoreboard;
bigger ramp''', '''1-4-month-old items (AGE_FADE) from the 2025 scoreboard; bigger ramp''')
open("rules.py", "w", encoding="utf-8").write(s)
print("patched")
