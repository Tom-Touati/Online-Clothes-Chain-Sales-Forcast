"""D173 (overnight M5 goal, direction 2): the planned campaign calendar as known-future features (M5: price / promotion features).
Site campaign intensity per month from online order lines: units-weighted mean discount and deep-discount days (daily site discount
>= 58.5%, as eda/57). In the backtest the realised calendar of the target months stands in for the plan (Edikted plans campaigns
in advance; the Jul-Oct 2026 plan is given in the workbook).
  site    c_disc_t, c_deep_t (target 4 months), and their change against the origin window (c_disc_d, c_deep_d)
  elast   + the item's discount response: slope of its monthly log units on the site discount over its history up to the origin
          (linear trend controlled, >= 6 live months), and slope x c_disc_d, slope x c_deep_d
  llm     + LLM-score interactions: statement x deep days, party x deep days, staple x discount change
Base = the D159 quantile model (eda/164 cpllm_we: lean + cart ramp + LLM + temperature + events). Writes data/q_backtest_cmp_{site,elast,llm}.parquet;
scored by eda/91_score_context.py cmp data/q_backtest_wx_cpllm_we.parquet."""
import sys, numpy as np, pandas as pd, time
src = open("eda/164_q_weather.py", encoding="utf-8").read(); _a = sys.argv; sys.argv = sys.argv[:1]
exec(src[src.index("import sys"):src.index("for var in RUN:")]); sys.argv = _a
RUN = sys.argv[1:] or ["site", "elast", "llm"]
d = pd.read_parquet("data/daily_item_chan.parquet", columns=["ny_date", "class", "cancelled", "qty", "disc_w"]); d = d[(~d.cancelled) & (d["class"] == "ONLINE")]
d["ny_date"] = pd.to_datetime(d.ny_date); s = d.groupby("ny_date").agg(q=("qty", "sum"), dw=("disc_w", "sum")); s["disc"] = s.dw / s.q
mon = pd.DataFrame({"disc": s.dw.resample("MS").sum() / s.q.resample("MS").sum(), "deep": s.disc.ge(0.585).resample("MS").sum()}); mon.index = mon.index.to_period("M").astype(str)
Pm = lambda o, k: str(pd.Period(o, "M") + k)
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
items = M.index.values; rows = []
for o in sorted(T0.origin.unique()):
    tg = [Pm(o, k) for k in (1, 2, 3, 4)]; og = [Pm(o, -k) for k in range(3)]
    cdt = mon.disc.reindex(tg).mean(); cdp = mon.deep.reindex(tg).sum() / 30; cdo = mon.disc.reindex(og).mean(); cpo = mon.deep.reindex(og).sum() / 30
    hm = [m for m in M.columns if m <= o and m >= Pm(o, -23)]; U = np.log1p(M[hm].values); live = np.cumsum(M[hm].values, 1) > 0; x = mon.disc.reindex(hm).values
    sl = np.full(len(items), np.nan)
    for i in range(len(items)):
        m_ = live[i] & np.isfinite(x)
        if m_.sum() >= 6:
            X = np.column_stack([np.ones(m_.sum()), np.arange(m_.sum()), x[m_]]); sl[i] = np.linalg.lstsq(X, U[i, m_], rcond=None)[0][2]
    rows.append(pd.DataFrame({"origin": o, "style_color": items, "c_disc_t": cdt, "c_deep_t": cdp, "c_disc_d": cdt - cdo, "c_deep_d": cdp - cpo, "disc_slope": sl}))
CF2 = pd.concat(rows); CF2["x_slope_disc"] = CF2.disc_slope * CF2.c_disc_d; CF2["x_slope_deep"] = CF2.disc_slope * CF2.c_deep_d
T0 = T0.merge(CF2, on=key, how="left")
T0["x_stmt_deep"] = T0.statement / 10 * T0.c_deep_t; T0["x_party_deep"] = T0.party / 10 * T0.c_deep_t; T0["x_staple_disc"] = T0.staple / 10 * T0.c_disc_d
SITE = ["c_disc_t", "c_deep_t", "c_disc_d", "c_deep_d"]; EL = ["disc_slope", "x_slope_disc", "x_slope_deep"]; LX = ["x_stmt_deep", "x_party_deep", "x_staple_disc"]
V = {"site": SITE, "elast": SITE + EL, "llm": SITE + EL + LX}
print("site discount by month (2025):", mon.loc["2025-01":"2025-12"].round(3).to_dict(), flush=True)
for var in RUN:
    NEW = CP + LL + WE + V[var]; qm.BASE = B0 + NEW; qm.FULL = F0 + NEW; out, t0 = [], time.time()
    for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
        oo = pd.Period(o).ordinal; tr = T0[(T0.o_ord + 4 <= oo) & T0.ytot.notna()]; te = T0[T0.origin == o]
        zq = qm.fit_q(tr, te); R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
        for j, q in enumerate(qm.QS): R[f"q{int(q*100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
        out.append(R)
    pd.concat(out).to_parquet(f"data/q_backtest_cmp_{var}.parquet"); print(var, round(time.time() - t0), "s", flush=True)
