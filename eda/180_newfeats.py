"""D252 (user goal, 18 Nov: "clear all the fixable errors"): new feature blocks for the D198 tree member, one row per (origin, item) of
data/fk_explain_table.parquet. Everything from data up to the origin's month-end, except the calendar (known future) and the
launch-plan block (PLAN, marked: it assumes the merchandising launch calendar is known at the origin).
AGE    days_live (true launch date, not clipped), age_m, launch day-of-year (sin, cos), launch-anniversary phase days_live mod 365
       (sin, cos), distance in days from the target window's middle to the item's next or last launch anniversary (anniv_dist).
EV2    15 more event windows (data/events_month2.parquet; days in the 4 target months _t and the 3 origin months _o): super_bowl,
       st_patricks, fathers_day, pride, prime_fall (October Prime days), singles_day (Nov 8-11), xmas_ship (Dec 10-18 shipping
       rush), after_xmas (Dec 26-31 sales), new_year_sale (Jan 1-7), mlk, presidents (long-weekend sales), tax_refund (Feb 15 -
       Mar 31), back_to_college (Aug 10 - Sep 5), nyfw (New York fashion weeks, Feb and Sep), cyber_week (Thanksgiving Monday ..
       Cyber Monday + 1). Calendar of the target window: weekend days, paydays (1st and 15th, and Fridays), days.
TREND  which kinds of items are rising or falling: for the item's product type, macro group, silhouette, print, length and fabric
       feel, the group's units (other items only) last 3 months vs the 3 before for items alive in both (like-for-like momentum,
       _mom) and the group's share of site units last 3 months vs the same 3 months a year earlier (_shr); the type's launch rate
       (launches in 3 months / live items); the item's trend91 minus its type's level-weighted mean (item_vs_type).
CURVE  launch curves from earlier launches (2024+, 30-day buckets since launch, only buckets observed by the origin): expected
       log ratio of the next 4 buckets' units to the last complete bucket at the item's age (by type, shrunk to all types, n / (n + 30);
       age >= 12 buckets pooled) = curve_fwd; the item's growth from bucket 1 to its last complete bucket against the same for
       peers at that age (curve_dev); log1p first-30-day units (first30); bucket of the item's best so far (best_bucket).
PEAK   daily, all channels: best 7-day run over life and over the last 365 days (log ratio to level), days since each, peak
       anniversary distance to the target window's middle, worst 28-day run over the last 365 live days (log ratio to level) and
       days since it, number of 7-day runs above 2x level in the last year (spikes), best 7-day run in the last 28 days.
WDAY   weekday mix from the last 91 days: the item's Fri-Sun share minus the site's (wk_end_rel); calendar-composition effect =
       log of the item-weekday-weighted day count of the target window over that of the last 91 days, per day (cal_eff).
PLAN   (known only with the launch plan) launches of the same type / macro group inside the 4 target months, over live items.
Writes data/nf2_feats.parquet and data/events_month2.parquet; prints coverage."""
import sys, json, numpy as np, pandas as pd
from datetime import date, timedelta
sys.stdout.reconfigure(encoding="utf-8")
T = pd.read_parquet("data/fk_explain_table.parquet")[["origin", "style_color", "level", "trend91", "ptype"]]
it = pd.read_parquet("data/items.parquet").set_index("style_color")
items = np.sort(it.index.values); ix = {s: i for i, s in enumerate(items)}; n = len(items)
it = it.reindex(items)
pub = pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
start = pd.concat([pub, pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
F6 = pd.read_parquet("data/features.parquet"); F6 = F6[F6.h == 1].drop_duplicates(["origin", "style_color"])
attr = F6.drop_duplicates("style_color").set_index("style_color").reindex(items)[["ptype", "macro", "a_silhouette", "a_print_pattern", "a_length", "a_fabric_feel"]].astype(str).fillna("na")
# ---------- daily matrix ----------
d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[~d.cancelled]; d["ny_date"] = pd.to_datetime(d.ny_date)
D0 = pd.Timestamp("2024-01-01"); days = pd.date_range(D0, "2026-06-30"); nd = len(days)
Dm = d.groupby(["style_color", "ny_date"]).qty.sum().unstack(fill_value=0).reindex(index=items, columns=days, fill_value=0).values.astype(np.float32); del d
import os
CUT = os.environ.get("CUT")   # D254 leakage test: wipe every sale after the cut month (daily and monthly) and rebuild that origin only
if CUT: Dm[:, days > pd.Period(CUT, "M").end_time] = 0
site_d = Dm.sum(0); wd = days.weekday.values
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; M = M.reindex(items).fillna(0); months = list(M.columns)
if CUT: M.loc[:, [m for m in months if m > CUT]] = 0; T = T[T.origin == CUT]
first_m = np.array([str(pd.Period(s, "M")) if pd.notna(s) else "9999-12" for s in start])
# ---------- events 2 ----------
def nth_wd(y, m, wdy, k): d0 = date(y, m, 1); off = (wdy - d0.weekday()) % 7; return d0 + timedelta(off + 7 * (k - 1))
def rng(a, b): return pd.date_range(a, b)
EVD = {}
for y in range(2023, 2028):
    thanks = nth_wd(y, 11, 3, 4)
    ev = {"super_bowl": rng(nth_wd(y, 2, 6, 2) - timedelta(7), nth_wd(y, 2, 6, 2)), "st_patricks": rng(date(y, 3, 10), date(y, 3, 17)),
          "fathers_day": rng(nth_wd(y, 6, 6, 3) - timedelta(9), nth_wd(y, 6, 6, 3)), "pride": rng(date(y, 6, 1), date(y, 6, 30)),
          "prime_fall": rng(nth_wd(y, 10, 1, 2) - timedelta(5), nth_wd(y, 10, 1, 2) + timedelta(1)), "singles_day": rng(date(y, 11, 8), date(y, 11, 11)),
          "xmas_ship": rng(date(y, 12, 10), date(y, 12, 18)), "after_xmas": rng(date(y, 12, 26), date(y, 12, 31)), "new_year_sale": rng(date(y, 1, 1), date(y, 1, 7)),
          "mlk": rng(nth_wd(y, 1, 0, 3) - timedelta(3), nth_wd(y, 1, 0, 3)), "presidents": rng(nth_wd(y, 2, 0, 3) - timedelta(3), nth_wd(y, 2, 0, 3)),
          "tax_refund": rng(date(y, 2, 15), date(y, 3, 31)), "back_to_college": rng(date(y, 8, 10), date(y, 9, 5)),
          "nyfw": rng(date(y, 2, 7), date(y, 2, 14)).append(rng(date(y, 9, 6), date(y, 9, 13))), "cyber_week": rng(thanks - timedelta(3), thanks + timedelta(5))}
    for k, v in ev.items(): EVD.setdefault(k, []).append(v)
allm = [str(p) for p in pd.period_range("2023-01", "2027-12", freq="M")]
E2 = pd.DataFrame({"m": allm})
for k, vs in EVD.items():
    dd = pd.DatetimeIndex(np.concatenate([v.values for v in vs])); E2[k] = [int(((dd.to_period("M").astype(str)) == m).sum()) for m in allm]
cal = pd.DataFrame({"d": pd.date_range("2023-01-01", "2027-12-31")}); cal["m"] = cal.d.dt.to_period("M").astype(str)
cal["wkend"] = cal.d.dt.weekday >= 5; cal["pay"] = cal.d.dt.day.isin([1, 15]); cal["fri"] = cal.d.dt.weekday == 4
E2 = E2.merge(cal.groupby("m").agg(c_wkend=("wkend", "sum"), c_pay=("pay", "sum"), c_fri=("fri", "sum"), c_days=("d", "size")).reset_index(), on="m")
(None if CUT else E2.to_parquet("data/events_month2.parquet")); E2i = E2.set_index("m"); EVN = list(EVD)
# ---------- launch curves (30-day buckets) ----------
st_idx = np.array([(s - D0).days if pd.notna(s) else -10 ** 6 for s in start]); NB = 22
Bk = np.full((n, NB), np.nan, np.float32); Bend = np.full((n, NB), 10 ** 9)          # bucket units and the day index its last day falls on
for i in range(n):
    if st_idx[i] < 0: continue
    for k in range(NB):
        a, b = st_idx[i] + 30 * k, st_idx[i] + 30 * k + 30
        if b > nd: break
        Bk[i, k] = Dm[i, a:b].sum(); Bend[i, k] = b - 1
ptc = attr.ptype.values
rows = []
for o in sorted(T.origin.unique()):
    P = pd.Period(o, "M"); e = (P.end_time.normalize() - D0).days; mid = P.end_time.normalize() + pd.Timedelta(days=61)
    sub = T[T.origin == o]; idx = np.array([ix[s] for s in sub.style_color]); lev = sub.level.values
    R = pd.DataFrame({"origin": o, "style_color": sub.style_color.values})
    # AGE
    st = start.values[idx]; live = (P.end_time.normalize() - pd.to_datetime(st)).days.values.astype(float)
    R["days_live"] = live; R["age_m"] = live / 30.44
    ldoy = pd.to_datetime(st).dayofyear.values; R["launch_doy_s"] = np.sin(2 * np.pi * ldoy / 365.25); R["launch_doy_c"] = np.cos(2 * np.pi * ldoy / 365.25)
    ph = np.mod(live, 365.25); R["anniv_s"] = np.sin(2 * np.pi * ph / 365.25); R["anniv_c"] = np.cos(2 * np.pi * ph / 365.25)
    x = np.mod(live + 61, 365.25); R["anniv_dist"] = np.where(live + 61 >= 300, np.minimum(x, 365.25 - x), np.nan)
    # EV2 + calendar
    tg = [str(P + k) for k in range(1, 5)]; og = [str(P - k) for k in range(3)]
    for k in EVN: R[f"e2_{k}_t"] = E2i.loc[tg, k].sum(); R[f"e2_{k}_o"] = E2i.loc[og, k].sum()
    for c in ["c_wkend", "c_pay", "c_fri", "c_days"]: R[c] = E2i.loc[tg, c].sum()
    # TREND
    ko = months.index(o); U3 = M.iloc[:, ko - 2:ko + 1].sum(1).values; U3p = M.iloc[:, ko - 5:ko - 2].sum(1).values
    U3y = M.iloc[:, ko - 14:ko - 11].sum(1).values if ko >= 14 else np.full(n, np.nan); site3 = U3.sum(); site3y = np.nansum(U3y) if ko >= 14 else np.nan
    lfl = (U3p > 0) & (U3 > 0)
    for g in ["ptype", "macro", "a_silhouette", "a_print_pattern", "a_length", "a_fabric_feel"]:
        lab = attr[g].values; G = pd.DataFrame({"g": lab, "u": U3 * lfl, "p": U3p * lfl, "a": U3, "y": U3y}).groupby("g").sum()
        gi = lab[idx]; gu, gp = G.u.reindex(gi).values - (U3 * lfl)[idx], G.p.reindex(gi).values - (U3p * lfl)[idx]
        R[f"tr_{g}_mom"] = np.log((gu + 10) / (gp + 10))
        R[f"tr_{g}_shr"] = np.log(((G.a.reindex(gi).values + 10) / site3) / ((G.y.reindex(gi).values + 10) / site3y)) if ko >= 14 else np.nan
    born = np.array([fm > str(P - 3) and fm <= o for fm in first_m]); alive = U3 > 0
    lr = pd.Series(born).groupby(ptc).sum() / pd.Series(alive).groupby(ptc).sum().clip(lower=1); R["tr_type_launch"] = lr.reindex(ptc[idx]).values
    w = lev + 1; tt = pd.Series(sub.trend91.fillna(0).values * w).groupby(sub.ptype.values).sum() / pd.Series(w).groupby(sub.ptype.values).sum()
    R["item_vs_type"] = sub.trend91.values - tt.reindex(sub.ptype.values).values
    # CURVE (earlier launches, buckets observed by e)
    obs = Bend <= e; Bo = np.where(obs, Bk, np.nan); ab = np.floor(live / 30).astype(int)    # completed buckets
    fwd = {}; dev = {}
    for k in range(1, 13):
        kk = min(k, 12); ok = np.isfinite(Bo[:, kk - 1]) & np.isfinite(Bo[:, min(kk + 3, NB - 1)]) & (st_idx >= 0) & (Bo[:, kk - 1] > 0)
        if kk + 3 >= NB or ok.sum() < 10: continue
        r = np.log((Bo[ok, kk:kk + 4].sum(1) + 1) / (4 * Bo[ok, kk - 1] + 1)); gr = np.log((Bo[ok, kk - 1] + 1) / (Bo[ok, 0] + 1))
        s = pd.DataFrame({"t": ptc[ok], "r": r, "g": gr}); glob_r, glob_g = np.median(r), np.median(gr); gb = s.groupby("t").agg(r=("r", "median"), g=("g", "median"), c=("r", "size"))
        sh = gb.c / (gb.c + 30); fwd[k] = (sh * gb.r + (1 - sh) * glob_r, glob_r); dev[k] = (sh * gb.g + (1 - sh) * glob_g, glob_g)
    cf = np.full(len(idx), np.nan); cd = np.full(len(idx), np.nan)
    for j, i in enumerate(idx):
        k = int(np.clip(ab[j], 1, 12))
        if k in fwd and st_idx[i] >= 0:
            cf[j] = fwd[k][0].get(ptc[i], fwd[k][1])
            if np.isfinite(Bo[i, k - 1]) and np.isfinite(Bo[i, 0]): cd[j] = np.log((Bo[i, k - 1] + 1) / (Bo[i, 0] + 1)) - dev[k][0].get(ptc[i], dev[k][1])
    R["curve_fwd"] = np.where(st_idx[idx] >= 0, cf, np.nan); R["curve_dev"] = cd
    R["first30"] = np.where(np.isfinite(Bo[idx, 0]), np.log1p(np.nan_to_num(Bo[idx, 0])), np.nan)
    R["best_bucket"] = [np.nanargmax(Bo[i]) if np.isfinite(Bo[i]).any() and np.nanmax(Bo[i]) > 0 else np.nan for i in idx]
    # PEAK
    X = Dm[idx, :e + 1]; c7 = np.concatenate([np.zeros((len(idx), 1)), np.cumsum(X, 1)], 1); r7 = c7[:, 7:] - c7[:, :-7]; r28 = c7[:, 28:] - c7[:, :-28]
    lv7 = (lev + 1) * 7 / 30.44; lv28 = (lev + 1) * 28 / 30.44
    a7 = r7.argmax(1); R["pk_life"] = np.log((r7.max(1) + 1) / lv7); R["pk_life_days"] = r7.shape[1] - 1 - a7
    y7 = r7[:, -365:]; ay = y7.argmax(1); R["pk_365"] = np.log((y7.max(1) + 1) / lv7); R["pk_365_days"] = y7.shape[1] - 1 - ay
    xx = np.mod(R.pk_life_days.values + 61, 365.25); R["pk_anniv_dist"] = np.where(R.pk_life_days.values + 61 >= 300, np.minimum(xx, 365.25 - xx), np.nan)
    y28 = r28[:, -365:].astype(float); livemask = (np.arange(e + 1)[None, -y28.shape[1]:] - 27) >= (st_idx[idx][:, None] + 28); y28 = np.where(livemask, y28, np.inf)
    mn = y28.min(1); ok = np.isfinite(mn); R["tr_365"] = np.where(ok, np.log((np.where(ok, mn, 0) + 1) / lv28), np.nan)
    R["tr_365_days"] = np.where(ok, y28.shape[1] - 1 - np.argmin(np.where(np.isfinite(y28), y28, 1e18), 1), np.nan)
    R["pk_spikes"] = (y7[:, ::7] > 2 * lv7[:, None]).sum(1); R["pk_28"] = np.log((r7[:, -22:].max(1) + 1) / lv7)
    # WDAY
    win = slice(e - 90, e + 1); Xw = Dm[idx, win]; wdw = wd[win]; tot = Xw.sum(1)
    sitew = site_d[win]; site_we = sitew[wdw >= 4].sum() / max(sitew.sum(), 1)
    R["wk_end_rel"] = np.where(tot >= 10, Xw[:, wdw >= 4].sum(1) / np.maximum(tot, 1) - site_we, np.nan)
    prof_site = np.array([sitew[wdw == k].sum() for k in range(7)]) / max(sitew.sum(), 1)
    prof = np.stack([Xw[:, wdw == k].sum(1) for k in range(7)], 1); prof = (prof + 20 * prof_site[None, :]) / (tot[:, None] + 20)
    cnt = lambda ds: np.array([(ds.weekday == k).sum() for k in range(7)])
    tdays = pd.date_range((P + 1).start_time, (P + 4).end_time.normalize()); rdays = days[win]
    rel = lambda p: p / np.array([1 / 7] * 7)
    R["cal_eff"] = np.log((prof * cnt(tdays)[None, :]).sum(1) / len(tdays)) - np.log((prof * cnt(rdays)[None, :]).sum(1) / len(rdays))
    # PLAN (launch plan known)
    fut = np.array([fm > o and fm <= str(P + 4) for fm in first_m])
    for g in ["ptype", "macro"]:
        lab = attr[g].values; fl = pd.Series(fut).groupby(lab).sum() / pd.Series(alive).groupby(lab).sum().clip(lower=1)
        R[f"plan_{g}"] = fl.reindex(lab[idx]).values
    rows.append(R); print(o, len(R), flush=True)
NF = pd.concat(rows, ignore_index=True); NF.to_parquet(f"data/_leak_nf2_{CUT}.parquet" if CUT else "data/nf2_feats.parquet")
if CUT: sys.exit()
BLK = {"AGE": ["days_live", "age_m", "launch_doy_s", "launch_doy_c", "anniv_s", "anniv_c", "anniv_dist"],
       "EV2": [c for c in NF.columns if c.startswith("e2_") or c.startswith("c_")],
       "TREND": [c for c in NF.columns if c.startswith("tr_") and not c.startswith("tr_365")] + ["item_vs_type"],
       "CURVE": ["curve_fwd", "curve_dev", "first30", "best_bucket"],
       "PEAK": ["pk_life", "pk_life_days", "pk_365", "pk_365_days", "pk_anniv_dist", "tr_365", "tr_365_days", "pk_spikes", "pk_28"],
       "WDAY": ["wk_end_rel", "cal_eff"], "PLAN": ["plan_ptype", "plan_macro"]}
json.dump(BLK, open("data/nf2_meta.json", "w"))
print(NF.shape); print(NF.drop(columns=["origin", "style_color"]).describe().T[["count", "mean", "min", "50%", "max"]].round(3).to_string())
