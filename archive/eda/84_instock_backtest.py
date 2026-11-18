"""In-stock uplift, step 2 (D107): backtest uplift variants on the 13 rule-agent origins, same items as eda/77/79.
Truth for in-stock demand = actual sales + estimated store losses in the 4 target months (data/store_lost_daily.parquet, all run days).
Uplifts use only run days known at the origin (day and known <= origin end):
  none   : no uplift
  flat   : x (1 + site lost / all units, last 180 days)
  retail : x (1 + item retail share (last 90 days) x site lost / retail units (last 180 days))
  own    : x (1 + item lost / item units, last 90 days, capped at 0.5)
  mix    : x (1 + average of the retail and own uplifts)
Forecasts uplifted: the three-way centre (rules x quantile x NN) and the quantile median. Scored against demand and against sales."""
import numpy as np, pandas as pd, contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("eda/79_combo_eval.py", encoding="utf-8").read().split('C = ["rules"')[0])
L = pd.read_parquet("data/store_lost_daily.parquet")
d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[~d.cancelled]; d["ny_date"] = pd.to_datetime(d.ny_date)
d["ret"] = np.where(d["class"] == "RETAIL", d.qty, 0.0)
DD = d.groupby(["style_color", "ny_date"])[["qty", "ret"]].sum().reset_index()
rows = []
for o in ORIGINS:
    end = pd.Period(o, "M").end_time.normalize(); t_end = (pd.Period(o, "M") + 4).end_time.normalize()
    k = L[(L.date <= end) & (L.known <= end)]
    g180 = DD[(DD.ny_date > end - pd.Timedelta(days=180)) & (DD.ny_date <= end)]; kl180 = k[k.date > end - pd.Timedelta(days=180)].lost.sum()
    flat, rrate = kl180 / g180.qty.sum(), kl180 / g180.ret.sum()
    i90 = DD[(DD.ny_date > end - pd.Timedelta(days=90)) & (DD.ny_date <= end)].groupby("style_color")[["qty", "ret"]].sum()
    l90 = k[k.date > end - pd.Timedelta(days=90)].groupby("style_color").lost.sum()
    fut = L[(L.date > end) & (L.date <= t_end)].groupby("style_color").lost.sum()
    x = T[T.origin == o][["origin", "style_color", "y", "r_q_nn", "qmodel"]].copy()
    q = i90.reindex(x.style_color).fillna(0)
    x["lost_out"] = fut.reindex(x.style_color).fillna(0).values
    x["u_flat"] = flat
    x["u_retail"] = (q.ret / q.qty.replace(0, np.nan)).fillna(0).values * rrate
    x["u_own"] = np.clip(l90.reindex(x.style_color).fillna(0).values / q.qty.replace(0, np.nan).values, 0, 0.5); x["u_own"] = x.u_own.fillna(0)
    x["u_mix"] = (x.u_retail + x.u_own) / 2
    x["rshare"] = (q.ret / q.qty.replace(0, np.nan)).fillna(0).values
    rows.append(x)
X = pd.concat(rows); X["dem"] = X.y + X.lost_out
U = ["none", "flat", "retail", "own", "mix"]
for f in ["r_q_nn", "qmodel"]:
    for u in U: X[f"{f}|{u}"] = X[f] * (1 + (0 if u == "none" else X[f"u_{u}"]))
def sc(F, c, tgt): e = (F[c] - F[tgt]).abs().sum() / F[tgt].sum(); b = F[c].sum() / F[tgt].sum() - 1; return round(e, 4), round(b, 3)
print("lost in target window / sales, by origin:", (X.groupby("origin").lost_out.sum() / X.groupby("origin").y.sum()).round(3).to_dict())
print("mean uplift by origin (flat / retail / own):"); print(X.groupby("origin")[["u_flat", "u_retail", "u_own"]].mean().round(3).T.to_string())
REC_ = [o for o in ORIGINS if o >= "2025-06"]
for nm, F in [("all 13 origins", X), ("nine recent origins", X[X.origin.isin(REC_)]), ("recent, targets", X[X.origin.isin(REC_) & X.style_color.isin(pd.read_csv("data/targets.csv").style_color)]),
              ("recent, retail share >= 30%", X[X.origin.isin(REC_) & (X.rshare >= 0.3)])]:
    print(f"\n== {nm} ({len(F)} item-origins): WAPE, bias vs in-stock demand | vs sales")
    for f in ["r_q_nn", "qmodel"]:
        for u in U: print(f"  {f:7s} {u:7s} demand {sc(F, f'{f}|{u}', 'dem')} | sales {sc(F, f'{f}|{u}', 'y')}")
# bootstrap: uplift minus none, vs demand, recent origins, three-way centre
F = X[X.origin.isin(REC_)]; g = [x for _, x in F.groupby("origin")]; rng = np.random.default_rng(0)
for u in ["flat", "retail", "own", "mix"]:
    dl = []
    for _ in range(1000):
        s = pd.concat([x.iloc[rng.integers(0, len(x), len(x))] for x in g])
        dl.append(((s[f"r_q_nn|{u}"] - s.dem).abs().sum() - (s["r_q_nn|none"] - s.dem).abs().sum()) / s.dem.sum())
    print(f"  {u} minus none (demand, recent): {np.round(np.percentile(dl, [2.5, 50, 97.5]), 4)}")
# does an item's own past loss rate predict its future loss rate?
F = X[X.y > 0]; fr = F.lost_out / F.y
print("\ncorrelation of past own loss rate with future loss rate:", round(np.corrcoef(F.u_own, fr.clip(0, 1))[0, 1], 3), "| retail share with future:", round(np.corrcoef(F.rshare, fr.clip(0, 1))[0, 1], 3))
X.to_parquet("data/instock_backtest.parquet", index=False)
