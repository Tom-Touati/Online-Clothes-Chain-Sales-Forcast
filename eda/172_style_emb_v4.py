"""D166 (user goal): v4 of the content style embedding, with stronger temperature and event-timing inputs, on the D164 gate
(eda/170: held-out styles, 3 origins x 5 folds, cart PPMI truth past 12 months and next 3 months, style bootstrap).
New inputs at each origin, from the item's own sales up to the origin (24 months):
  TEMP2  temperature at which the item sells: units-weighted mean of the item-mix-weighted state temperature over its live months;
         slope of log monthly units on that temperature with a linear time trend (live months >= 6); correlation of detrended log
         units with the national temperature anomaly
  EVT2   for each of 16 events (key dates: Valentine's Feb 14, spring break Mar 15, Easter, festival Apr 15, prom May 1, Mother's Day,
         Memorial Day, graduation Jun 1, July 4, Prime Day, back-to-school Aug 15, Labor Day, Halloween Oct 31, Black Friday,
         Christmas Dec 25, New Year's Eve): log lift of daily units in the 30 days before the event over the item's own daily rate,
         and the units-weighted mean days before the event of its sales in the 60 days before it
v4 = v3 (eda/170) with temp = [TEMP, TEMP2] and events = [EVENTS, EVT2]; ablations without region / temperature / events.
Reference on the future target only: the fork's dynamic co-occurrence network (data/fk_cooc_dyn.npz, D198/D199). It is trained on
every item's past carts, held-out ones included, so it is shown as "uses the item's own past carts", not as a held-out result.
Writes eda/style_emb_v4_output.txt (stdout) and data/style_emb_v4_results.csv."""
import os, sys
os.environ.setdefault("GATE_STEPS", "500")
src = open("eda/170_style_emb_gate.py", encoding="utf-8").read()
_argv = sys.argv; sys.argv = sys.argv[:1] + ["v3"]
exec(src[:src.index("CFG = {")])
sys.argv = _argv
from datetime import date, timedelta
RUN4 = sys.argv[1:] or ["v3", "v4", "v4_noreg", "v4_notemp", "v4_noevt"]
DL = pd.read_parquet("data/daily_item_chan.parquet", columns=["style_color", "ny_date", "qty", "cancelled"]); DL = DL[~DL.cancelled]
DL["ny_date"] = pd.to_datetime(DL.ny_date); DAYS = pd.date_range("2024-01-01", "2026-06-30"); di = {d: k for k, d in enumerate(DAYS)}; ci = {s: k for k, s in enumerate(items)}
DAILY = np.zeros((n, len(DAYS)), np.float32)
g = DL.groupby(["style_color", "ny_date"]).qty.sum().reset_index(); g = g[g.style_color.isin(ci)]
DAILY[g.style_color.map(ci).values, g.ny_date.map(di).values] = g.qty.values
EASTER = {2024: date(2024, 3, 31), 2025: date(2025, 4, 20), 2026: date(2026, 4, 5)}
PRIME = {2024: date(2024, 7, 16), 2025: date(2025, 7, 8), 2026: date(2026, 7, 14)}
def nth_wd(y, m, wd, k):
    d = date(y, m, 1)
    if k > 0: return d + timedelta((wd - d.weekday()) % 7 + 7 * (k - 1))
    d = date(y, m + 1, 1) - timedelta(1); return d - timedelta((d.weekday() - wd) % 7)
KEY = {"valentine": lambda y: date(y, 2, 14), "spring_break": lambda y: date(y, 3, 15), "easter": lambda y: EASTER[y], "festival": lambda y: date(y, 4, 15),
       "prom": lambda y: date(y, 5, 1), "mothers": lambda y: nth_wd(y, 5, 6, 2), "memorial": lambda y: nth_wd(y, 5, 0, -1), "graduation": lambda y: date(y, 6, 1),
       "july4": lambda y: date(y, 7, 4), "prime_day": lambda y: PRIME[y], "bts": lambda y: date(y, 8, 15), "labor_day": lambda y: nth_wd(y, 9, 0, 1),
       "halloween": lambda y: date(y, 10, 31), "bfcm": lambda y: nth_wd(y, 11, 3, 4) + timedelta(1), "holiday": lambda y: date(y, 12, 25), "nye": lambda y: date(y, 12, 31)}
TMS = W.pivot(index="m", columns="state", values="tmean").reindex(columns=states); TMS = TMS.T.fillna(TMS.mean(axis=1)).T
TAN = W.groupby("m").tanom.mean()
FS = pd.to_datetime(it.first_sale)


def inputs2(o):
    od = pd.Period(o, "M").to_timestamp(how="end").normalize(); end = di[od] + 1; start = max(0, end - 730)
    fsi = np.array([di.get(f.normalize(), 0) if pd.notna(f) else end for f in FS])
    Dw = DAILY[:, start:end]; live = (np.arange(start, end)[None, :] >= fsi[:, None])
    rate = np.where(live.sum(1) > 0, (Dw * live).sum(1) / np.maximum(live.sum(1), 1), np.nan)
    lift, mdays = [], []
    for e, f in KEY.items():
        num = np.zeros(n); den = np.zeros(n); wsum = np.zeros(n); dsum = np.zeros(n)
        for y in (2024, 2025, 2026):
            d = pd.Timestamp(f(y)); k = di.get(d)
            if k is None or k > end - 1 or k - 60 < start: continue
            a30 = slice(k - 30, k); ok30 = fsi <= k - 30
            num += np.where(ok30, DAILY[:, a30].sum(1), 0); den += np.where(ok30, 30, 0)
            a60 = DAILY[:, k - 60:k]; dd = np.arange(60, 0, -1)[None, :]
            wsum += a60.sum(1); dsum += (a60 * dd).sum(1)
        lift.append(np.where(den > 0, np.log((num + 1) / (np.nan_to_num(rate) * den + 1)), np.nan))
        mdays.append(np.where(wsum > 0, dsum / np.maximum(wsum, 1e-9), np.nan))
    EVT2 = np.column_stack(lift + mdays)
    ms = [Pm(o, -k) for k in range(23, -1, -1)]; U = M.reindex(columns=ms).fillna(0).values; lv = np.cumsum(U, 1) > 0
    w12 = [mi[m] for m in mons if Pm(o, -11) <= m <= o]; u12 = UST[w12].sum(0)
    mix = (u12 + 20 * (u12.sum(0) / max(u12.sum(), 1))) / (u12.sum(1, keepdims=True) + 20)
    Tm = mix @ TMS.reindex(ms).values.T                                  # items x months: temperature where the item sells
    tsell = np.where(lv.any(1), (U * Tm).sum(1) / np.maximum(U.sum(1), 1e-9), np.nan)
    an = TAN.reindex(ms).values; slope = np.full(n, np.nan); acor = np.full(n, np.nan)
    for i in range(n):
        m_ = lv[i]
        if m_.sum() < 6: continue
        y_ = np.log1p(U[i, m_]); t_ = np.arange(m_.sum()); X = np.column_stack([np.ones(m_.sum()), t_, Tm[i, m_]])
        b = np.linalg.lstsq(X, y_, rcond=None)[0]; slope[i] = b[2]
        r_ = y_ - X[:, :2] @ np.linalg.lstsq(X[:, :2], y_, rcond=None)[0]
        if np.std(r_) > 0 and np.std(an[m_]) > 0: acor[i] = np.corrcoef(r_, an[m_])[0, 1]
    TEMP2 = np.column_stack([tsell, slope, acor])
    return np.nan_to_num(zs(TEMP2)), np.nan_to_num(zs(EVT2))


CFG4 = {"v3": ("nce", ["prod", "reg", "temp", "evt"]), "v4": ("nce", ["prod", "reg", "temp4", "evt4"]), "v4_noreg": ("nce", ["prod", "temp4", "evt4"]),
        "v4_notemp": ("nce", ["prod", "reg", "evt4"]), "v4_noevt": ("nce", ["prod", "reg", "temp4"]), "v4_noevt_notemp": ("nce", ["prod", "reg"])}
DYN = np.load("data/fk_cooc_dyn.npz", allow_pickle=True); dyn_items = DYN["items"].astype(str); dyn_ix = pd.Series(range(len(dyn_items)), index=dyn_items).reindex(items).values
res = []; t0 = time.time()
for o in ORIG:
    REGION, TEMP, EVENTS = inputs(o); TEMP2, EVT2 = inputs2(o); TR = truth(o)
    B = {"prod": np.hstack([ATT, COLL, NAME, LLM]), "reg": REGION, "temp": TEMP, "evt": EVENTS, "temp4": np.hstack([TEMP, TEMP2]), "evt4": np.hstack([EVENTS, EVT2])}
    raw_prod = rown(zs(PROD_RAW)); raw_all = rown(np.hstack([zs(PROD_RAW), LLM, REGION, TEMP, EVENTS, TEMP2, EVT2]))
    dynv = rown(np.nan_to_num(DYN[o].astype(float)[dyn_ix])) if o in DYN.files else None
    alive = M.reindex(columns=[Pm(o, -k) for k in range(12)]).fillna(0).values.sum(1) > 0
    styles = np.unique(sty); fold = dict(zip(styles, np.random.default_rng(0).permutation(len(styles)) % 5)); f = np.array([fold[s] for s in sty])
    for k in range(NF):
        tr = np.where(alive & (f != k))[0]; te = np.where(alive & (f == k))[0]
        Hs = {"raw product": raw_prod, "raw everything": raw_all}
        for v in RUN4:
            obj, blk = CFG4[v]; Hs[v] = train([B[b] for b in blk], tr, *TR["past"], obj)
        for tgt, (P_, ok_) in TR.items():
            H2 = dict(Hs)
            if tgt == "future" and dynv is not None: H2["dyn net (uses own past carts)"] = dynv
            for nm, H in H2.items():
                d = score(H, te, tr, P_, ok_); d["model"] = nm; d["target"] = tgt; d["origin"] = o; d["fold"] = k; res.append(d)
        print(o, "fold", k, f"{time.time() - t0:.0f}s", flush=True)
R = pd.concat(res); R.to_csv(os.environ.get("GATE_OUT", "data/style_emb_v4_results.csv"), index=False)
print("\n== mean over held-out items: Spearman with cart PPMI | top-20 precision | chance")
print(R.groupby(["target", "model"])[["rho", "prec", "chance"]].mean().round(4).to_string())
rng = np.random.default_rng(0)
for ref in [r for r in ["raw product", "v3"] if r in R.model.unique()]:
    print(f"\n== paired bootstrap over held-out styles, model minus {ref} (95% interval)")
    for tgt in ["past", "future"]:
        base = R[(R.target == tgt) & (R.model == ref)].set_index(["origin", "style"])[["rho", "prec"]].groupby(level=[0, 1]).mean()
        for m in [x for x in R.model.unique() if x != ref]:
            x = R[(R.target == tgt) & (R.model == m)].set_index(["origin", "style"])[["rho", "prec"]].groupby(level=[0, 1]).mean()
            dlt = (x - base.reindex(x.index)).dropna()
            if not len(dlt): continue
            bs = np.array([dlt.values[rng.integers(0, len(dlt), len(dlt))].mean(0) for _ in range(1000)]); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            print(f"  {tgt:6s} {m:30s} rho {dlt.rho.mean():+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}] | prec {dlt.prec.mean():+.4f} [{lo[1]:+.4f}, {hi[1]:+.4f}]")
