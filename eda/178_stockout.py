"""D174 (overnight M5 goal, direction 3): stock-out-corrected inputs and targets (M5 / Favorita: censored demand).
From daily sales (all channels; sizes sold per day from the online channel), at each origin, for the 91 days up to it:
  a avail   oos_share: share of live days with zero sales while the trailing 28-day rate was >= 2 a day (likely out of stock);
            oos28: the same over the last 28 days; size_breadth: sizes selling in the last 28 days / the item's max sizes in a day so far;
            days_since_sale; restock: sales in the last 7 days / sales in the 21 days before (recovery after a gap)
  b level   + lvl_adj = log1p(units in the 91 days / in-stock days x 30.44) and its ratio to the plain level (the demand-corrected level)
  c weight  + training rows weighted by 1 - the target window's likely-out-of-stock share (only training targets, which are in the past),
            floor 0.3: censored targets count less
Base = the D159 quantile model (eda/164 cpllm_we). Writes data/q_backtest_oos_{avail,level,weight}.parquet;
scored by eda/91_score_context.py oos data/q_backtest_wx_cpllm_we.parquet."""
import sys, numpy as np, pandas as pd, time, lightgbm as lgb
src = open("eda/164_q_weather.py", encoding="utf-8").read(); _a = sys.argv; sys.argv = sys.argv[:1]
exec(src[src.index("import sys"):src.index("for var in RUN:")]); sys.argv = _a
RUN = sys.argv[1:] or ["avail", "level", "weight"]
d = pd.read_parquet("data/daily_item_chan.parquet", columns=["style_color", "ny_date", "class", "cancelled", "qty", "n_sizes"]); d = d[~d.cancelled]; d["ny_date"] = pd.to_datetime(d.ny_date)
items = np.sort(pd.read_parquet("data/items.parquet").style_color.values); ci = {s: i for i, s in enumerate(items)}; DAYS = pd.date_range("2024-01-01", "2026-10-31"); di = {x: k for k, x in enumerate(DAYS)}
Q = np.zeros((len(items), len(DAYS)), np.float32); SZ = np.zeros_like(Q)
g = d.groupby(["style_color", "ny_date"]).qty.sum().reset_index(); g = g[g.style_color.isin(ci)]; Q[g.style_color.map(ci), g.ny_date.map(di)] = g.qty.values
g = d[d["class"] == "ONLINE"].groupby(["style_color", "ny_date"]).n_sizes.max().reset_index(); g = g[g.style_color.isin(ci)]; SZ[g.style_color.map(ci), g.ny_date.map(di)] = g.n_sizes.values
csum = np.cumsum(np.hstack([np.zeros((len(items), 1)), Q]), 1); trail = lambda k, w: (csum[:, k] - csum[:, max(k - w, 0)]) / w      # mean over days [k-w, k)
first = np.argmax(Q > 0, 1); first[Q.sum(1) == 0] = len(DAYS)
likely = np.zeros_like(Q, bool)
for k in range(28, len(DAYS)): likely[:, k] = (Q[:, k] == 0) & (trail(k, 28) >= 2) & (k > first)
maxsz = np.maximum.accumulate(SZ, axis=1)
rows = []
for o in sorted(T0.origin.unique()):
    e = di[pd.Period(o, "M").to_timestamp(how="end").normalize()] + 1; s91 = slice(e - 91, e); live = np.arange(e - 91, e)[None, :] >= first[:, None]
    nlive = live.sum(1); oos = (likely[:, s91] & live).sum(1); oos28 = likely[:, e - 28:e].sum(1)
    sz28 = (SZ[:, e - 28:e] > 0).any(1) * np.max(SZ[:, e - 28:e], 1); breadth = np.where(maxsz[:, e - 1] > 0, np.max(SZ[:, e - 28:e], 1) / np.maximum(maxsz[:, e - 1], 1), np.nan)
    last = np.where(Q[:, :e].sum(1) > 0, e - 1 - np.argmax(Q[:, :e][:, ::-1] > 0, 1), np.nan)
    u91 = Q[:, s91].sum(1); instock = np.maximum(nlive - oos, 1)
    tg = slice(e, min(e + 123, len(DAYS))); oos_t = (likely[:, tg]).sum(1) / max(tg.stop - tg.start, 1)
    rows.append(pd.DataFrame({"origin": o, "style_color": items, "oos_share": np.where(nlive > 0, oos / np.maximum(nlive, 1), np.nan), "oos28": oos28 / 28,
                              "size_breadth": breadth, "days_since_sale": (e - 1) - last, "restock": np.log((Q[:, e - 7:e].sum(1) + 1) / (Q[:, e - 28:e - 7].sum(1) / 3 + 1)),
                              "lvl_adj": np.log1p(u91 / instock * 30.44), "oos_target_share": oos_t}))
OF = pd.concat(rows); T0 = T0.merge(OF, on=key, how="left"); T0["lvl_adj_ratio"] = T0.lvl_adj - T0.lvl
print("likely out-of-stock share of live days (91 days to origin), mean by origin:", OF.groupby("origin").oos_share.mean().round(3).tail(6).to_dict(), flush=True)
A = ["oos_share", "oos28", "size_breadth", "days_since_sale", "restock"]; LV = ["lvl_adj", "lvl_adj_ratio"]
V = {"avail": A, "level": A + LV, "weight": A + LV}
fit0 = qm.fit_q
def fit_w(tr, te):          # version c: weighted training rows
    out = np.zeros((len(te), len(qm.QS))); w = np.clip(1 - tr.oos_target_share.fillna(0).values, 0.3, 1)
    for cols in (qm.BASE, qm.FULL):
        for j, q in enumerate(qm.QS):
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, **qm.P).fit(tr[cols], tr.z, sample_weight=w); out[:, j] += m.predict(te[cols]) / 2
    return np.sort(out, axis=1)
for var in RUN:
    NEW = CP + LL + WE + V[var]; qm.BASE = B0 + NEW; qm.FULL = F0 + NEW; fit = fit_w if var == "weight" else fit0; out, t0 = [], time.time()
    for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
        oo = pd.Period(o).ordinal; tr = T0[(T0.o_ord + 4 <= oo) & T0.ytot.notna()]; te = T0[T0.origin == o]
        zq = fit(tr, te); R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
        for j, q in enumerate(qm.QS): R[f"q{int(q*100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
        out.append(R)
    pd.concat(out).to_parquet(f"data/q_backtest_oos_{var}.parquet"); print(var, round(time.time() - t0), "s", flush=True)
