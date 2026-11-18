"""D157: weather and event features per item and origin (time-honest unless marked ORACLE).
Item regional mix: the item's online units by state over the 12 months up to the origin, shrunk toward the site's mix
(w = (u_is + 30 x site_s) / (u_i + 30)). Temperatures are mix-weighted state monthly means (data/weather_state_month.parquet).
  t_org        realised mean temperature over the origin window (months o-2..o)            known at the origin
  ta_org       its anomaly against the 2014-2023 normal                                      known
  t_tgt_norm   normal temperature over the 4 target months                                    known (climatology)
  dT_norm      t_tgt_norm - normal temperature of the origin window (the seasonal temperature swing)
  dT_real_from_org  t_tgt_norm - t_org (swing from what the origin window actually felt like)
  beta_T       the item's temperature preference: weighted slope of log(units / expected units) across states on the state's
               temperature relative to the site that month, 12 months up to the origin (expected = item month total x state share)
  bT_dT        beta_T x dT_norm (temperature-driven change expected for this item); bT_ta = beta_T x ta_org
  ORACLE: t_tgt_real, ta_tgt (realised target-window temperature and anomaly; not knowable at the origin, diagnostic only)
Events (data/events_month.parquet): days of each event window in the 4 target months (ev_<name>_t) and in the 3 origin months
(ev_<name>_o); interactions with the LLM item scores (D151): halloween, party x (holiday + nye), festival, back_to_school x bts,
summer x (july4 + memorial), winter x holiday, and bfcm days.
Writes data/weather_event_feats.parquet."""
import sys, glob, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
items = np.sort(pd.read_parquet("data/items.parquet").style_color.values); n = len(items)
W = pd.read_parquet("data/weather_state_month.parquet"); states = sorted(W.state.unique()); si = {s: i for i, s in enumerate(states)}; ns = len(states)
TM = W.pivot(index="m", columns="state", values="tmean")[states]; TN = W.pivot(index="m", columns="state", values="tmean_norm")[states]
S = pd.read_parquet("data/state_item_month.parquet"); S = S[S.state.isin(si)]
S["s"] = S.state.map(si)
months = sorted(S.m.unique()); mi = {m: i for i, m in enumerate(months)}
U = np.zeros((len(months), n, ns), np.float32); U[S.m.map(mi).values, S.it.values, S.s.values] = S.units.values
E = pd.read_parquet("data/events_month.parquet").set_index("m"); EV = list(E.columns)
L = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("data/llm/scores_part*.csv"))]).drop_duplicates("style_color").set_index("style_color").reindex(items) / 10
Pm = lambda o, k: str(pd.Period(o, "M") + k)
rows = []
for o in [str(p) for p in pd.period_range("2024-03", "2026-06", freq="M")]:
    win12 = [mi[m] for m in months if Pm(o, -11) <= m <= o]
    u12 = U[win12].sum(0); site = u12.sum(0) / max(u12.sum(), 1)
    w = (u12 + 30 * site[None, :]) / (u12.sum(1, keepdims=True) + 30)                         # items x states
    org = [Pm(o, -k) for k in range(3)]; tgt = [Pm(o, k) for k in range(1, 5)]
    tn = lambda ms: TN.reindex(ms).values.mean(0); tr = lambda ms: TM.reindex(ms).values.mean(0)
    t_org = w @ tr(org); tn_org = w @ tn(org); t_tgt_norm = w @ tn(tgt)
    t_tgt_real = w @ tr(tgt) if all(m in TM.index for m in tgt) else np.full(n, np.nan)
    # temperature preference across states (12 months up to the origin)
    num = np.zeros(n); den = np.zeros(n)
    for k in win12:
        Um = U[k]; tot = Um.sum(1, keepdims=True); sh = Um.sum(0) / max(Um.sum(), 1)
        Ex = tot * sh[None, :]; T_s = TM.loc[months[k]].values; x = T_s - (sh * T_s).sum()
        r = np.log((Um + 0.5) / (Ex + 0.5)); ok = Ex > 0.5
        num += (Ex * x[None, :] * r * ok).sum(1); den += (Ex * x[None, :] ** 2 * ok).sum(1)
    beta = np.where(den > 50, num / np.maximum(den, 1e-9), np.nan)
    f = pd.DataFrame({"origin": o, "style_color": items, "t_org": t_org, "ta_org": t_org - tn_org, "t_tgt_norm": t_tgt_norm, "dT_norm": t_tgt_norm - tn_org,
                      "dT_real_from_org": t_tgt_norm - t_org, "beta_T": beta, "t_tgt_real": t_tgt_real, "ta_tgt": t_tgt_real - t_tgt_norm})
    f["bT_dT"] = f.beta_T * f.dT_norm; f["bT_ta"] = f.beta_T * f.ta_org
    for e in EV: f[f"ev_{e}_t"] = E[e].reindex(tgt).fillna(0).sum(); f[f"ev_{e}_o"] = E[e].reindex(org).fillna(0).sum()
    f["x_halloween"] = L.halloween.values * f.ev_halloween_t; f["x_party"] = L.party.values * (f.ev_holiday_t + f.ev_nye_t)
    f["x_festival"] = L.festival.values * f.ev_festival_t; f["x_bts"] = L.back_to_school.values * f.ev_bts_t
    f["x_summer"] = L.summer.values * (f.ev_july4_t + f.ev_memorial_t); f["x_winter"] = L.winter.values * f.ev_holiday_t
    f["x_warm_dT"] = L.warmth.values * f.dT_norm
    rows.append(f); print(o, "beta_T sd", round(np.nanstd(beta), 3), "| items with beta", int(np.isfinite(beta).sum()), flush=True)
X = pd.concat(rows, ignore_index=True); X.to_parquet("data/weather_event_feats.parquet", index=False); print("saved", X.shape)
