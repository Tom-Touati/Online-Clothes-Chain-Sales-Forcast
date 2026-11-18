"""Fork 2 (user, 18 Nov: "did we actually use the campaign info they provided us in the sheet? ... we need to use information like this").
Does a change in campaign (deep-discount) days against the same months last year leave error in the submitted system that an explicit,
item-scaled uplift can remove? If so, size it time-honestly and apply the 2026 plan.
  deep day   a day whose edikted.com unit-weighted discount is >= 58.5% (D016 / D030 definition); realised days stand in for the plan in the
             backtest (an optimistic stand-in: in 2026 only headline windows are known)
  exposure   x = (deep days in the 4 target months - deep days in the same months a year earlier) / days in the window x the item's edikted.com
             share at the origin (campaigns run on the US site only); variant xp also x the item's promotion sensitivity (promo_lift, standardised)
  fit        e = log((y + 1) / (F + 1)) of the submitted system (nx6 = new nx5 backtest, data/geo_nx5r_x_preds.parquet) on x with origin fixed
             effects (only the cross-item variation within an origin identifies beta, so a site-wide level error cannot masquerade as a campaign
             effect); time-honest beta at origin o from origins whose windows ended by o
  apply      F x exp(beta x); WAPE / R3 recent / held out / focus windows / all 13 against the submitted, style bootstrap
2026: the plan's two new windows (Clearance 12-19 Aug, 8 days; sale 24-29 Sep, 6 days) are the only deep days 2025 did not have on those
dates; the recurring ones are assumed to repeat 2025's realised depth. stdout -> pipeline/campaign_test_output.txt"""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
d = pd.read_parquet("data/daily_item_chan.parquet", columns=["ny_date", "chan", "qty", "disc_w", "cancelled"]); d = d[~d.cancelled & (d.chan == "edikted.com")]
d["ny_date"] = pd.to_datetime(d.ny_date); w = d.groupby("ny_date").agg(q=("qty", "sum"), dw=("disc_w", "sum")); w["deep"] = (w.dw / w.q) >= 0.585
DM = w.deep.groupby(w.index.to_period("M")).sum()
print("deep days by month (Jul-Oct):", {str(k): int(v) for k, v in DM.items() if k.month in (7, 8, 9, 10)})
G = pd.read_parquet("data/geo_nx5r_x_preds.parquet")[["origin", "style_color", "geo", "y", "R3"]]; G["style"] = G.style_color.str.split("_").str[0]
E = pd.read_parquet("data/fk_err_master.parquet")[["origin", "style_color", "sh_web"]]
P = pd.read_parquet("data/breakout_preds_nx5r26_x.parquet")[["origin", "style_color", "promo_lift"]]
G = G.merge(E, on=["origin", "style_color"], how="left").merge(P, on=["origin", "style_color"], how="left")
G["sh_web"] = G.sh_web.fillna(G.sh_web.median()); pl = G.promo_lift; G["pz"] = ((pl - pl.median()) / (pl.std() + 1e-9)).clip(-2, 2).fillna(0)
def window(o, k=0):
    P_ = pd.Period(o, "M"); ms = [P_ + h - 12 * k for h in (1, 2, 3, 4)]; return sum(int(DM.get(m, 0)) for m in ms), sum(m.days_in_month for m in ms)
rows = []
for o in sorted(G.origin.unique()):
    dt, n = window(o); dl, _ = window(o, 1); rows.append(dict(origin=o, deep_target=dt, deep_last_year=dl, delta=dt - dl, days=n))
W = pd.DataFrame(rows); print(W.to_string(index=False))
G = G.merge(W[["origin", "delta", "days"]], on="origin"); G["x"] = G.delta / G.days * G.sh_web; G["xp"] = G.x * (1 + 0.5 * G.pz)
G["e"] = np.log((G.y + 1) / (G.geo + 1)); oi = pd.PeriodIndex(G.origin, freq="M").asi8
def beta(m, col):
    """slope of e on col with origin fixed effects (demeaned within origin), volume-weighted by R3"""
    g = G[m]; xe = g[col] - g.groupby("origin")[col].transform("mean"); ee = g.e - g.groupby("origin").e.transform("mean"); wt = np.sqrt(g.R3 + 1)
    den = (wt * xe ** 2).sum(); return float((wt * xe * ee).sum() / den) if den > 1e-12 else 0.0
for col in ["x", "xp"]:
    b_all = beta(np.ones(len(G), bool), col); print(f"\nin-sample slope ({col}), all 13 origins: {b_all:.3f}  (a deep day sells about 1.55x on the site, log 0.44)")
y, R3, O = G.y.values, G.R3.values, G.origin.values; rng = np.random.default_rng(0)
w_ = lambda f, m: np.abs(f - y)[m].sum() / np.abs(R3 - y)[m].sum()
def ci(a, b, m):
    dd = pd.DataFrame(dict(s=G["style"].values[m], a=np.abs(a - y)[m], b=np.abs(b - y)[m], r=np.abs(R3 - y)[m])).groupby("s").sum()
    i = rng.integers(0, len(dd), size=(400, len(dd))); v = (dd.a.values[i].sum(1) - dd.b.values[i].sum(1)) / dd.r.values[i].sum(1); return f"[{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}]"
for col in ["x", "xp"]:
    F = G.geo.values.copy(); bs = {}
    for o in np.unique(oi):
        hist = oi <= o - 4; m = oi == o
        b = beta(hist, col) if hist.sum() > 500 else 0.0; bs[str(pd.Period(ordinal=int(o), freq="M"))] = round(b, 3); F[m] = G.geo.values[m] * np.exp(b * G[col].values[m])
    print(f"\n== time-honest uplift on {col}; beta by origin: {bs}")
    for lab, m in [("recent 9", O >= "2025-06"), ("held out", O >= "2025-10"), ("all 13", O > "0"), ("Jul-Oct 25", O == "2025-06"), ("Mar-Jun 26", O == "2026-02")]:
        print(f"   {lab:10s} submitted {w_(G.geo.values, m):.4f} -> {w_(F, m):.4f}  {ci(F, G.geo.values, m)}")
    # oracle (in-sample beta) upper bound
    Fo = G.geo.values * np.exp(beta(np.ones(len(G), bool), col) * G[col].values); print(f"   oracle beta: recent {w_(Fo, O >= '2025-06'):.4f}, held out {w_(Fo, O >= '2025-10'):.4f}")
# 2026: new deep days in the plan against 2025's realised days on the same dates
d25 = {m: int(DM.get(pd.Period(f"2025-{m:02d}", "M"), 0)) for m in (7, 8, 9, 10)}
new26 = {7: 0, 8: 8, 9: 6, 10: 0}
print("\n2026 plan vs 2025 deep days by month: 2025", d25, "| new windows in 2026 (days)", new26, "| delta over Jul-Oct", sum(new26.values()), "of 123 days")
# fixed beta from the site-level evidence (a deep day sells ~1.55x like-for-like: log 0.44; D031 halved it), no fitting on item errors
print("\n== fixed beta (site-level lift, not fitted on the backtest errors)")
for b in (0.22, 0.44):
    F = G.geo.values * np.exp(b * G.x.values)
    print(f" beta {b}: " + " | ".join(f"{lab} {w_(G.geo.values, m):.4f}->{w_(F, m):.4f} {ci(F, G.geo.values, m)}" for lab, m in [("recent", O >= "2025-06"), ("held out", O >= "2025-10"), ("all 13", O > "0"), ("Jul-Oct 25", O == "2025-06"), ("Mar-Jun 26", O == "2026-02")]))
