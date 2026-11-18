"""Store-level stock-outs (D106). D021 measured zero runs on all-channel item totals and size gaps online only; both hide a
stock-out in one store while the item keeps selling elsewhere. Here the same two tests run per retail store:
A) store x item daily: zero runs on days the store sold something, inside the item's first..last sale at that store
   (tail runs after the last sale reported separately: could be a stock-out or the item leaving the store).
   Rate = mean of the previous 28 open days at that store (>= 7 open days, flagged days excluded).
   Strict: rate x days > 6.9 (P < 0.1%); loose: rate x days > 3.0 (P < 5%). Lost units = rate x days.
B) store x item x week: a core size (>= 15% of the item's units in that store) sells 0 while the other sizes imply >= 3 (P < 5%)
   or >= 7 (P < 0.1%). Lost units = the implied units.
Online gets the same B test at the >= 3 bar for comparison."""
import duckdb, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
con = duckdb.connect(); con.execute("SET memory_limit='1500MB'; SET threads=4")
S = con.execute("SELECT store, style_color, ny_date::DATE d, size, sum(qty_ordered) q FROM read_parquet('Dataset.parquet') WHERE NOT cancelled AND class='RETAIL' GROUP BY ALL").fetchdf()
S["d"] = pd.to_datetime(S.d)
print("retail units", int(S.q.sum()), "| stores", S.store.nunique(), "| first", S.d.min().date(), "| last", S.d.max().date())
DI = S.groupby(["store", "style_color", "d"]).q.sum()
open_days = S.groupby("store").d.unique()

# ---- A) store x item zero runs
rows, live_days, tail_rows = [], 0, []
for (st, sc), v in DI.groupby(level=[0, 1]):
    od = pd.DatetimeIndex(np.sort(open_days[st])); s = v.droplevel([0, 1]).reindex(od, fill_value=0).values.astype(float)
    nz = np.flatnonzero(s > 0)
    if len(nz) < 2: continue
    a0, a1 = nz[0], nz[-1]; live_days += a1 - a0 + 1
    z = s == 0; z[:a0] = False
    flag = np.zeros(len(s), bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], z.astype(int), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        pre_i = np.arange(max(a - 28, a0), a); pre_i = pre_i[~flag[pre_i]]
        if len(pre_i) < 7: continue
        lam = s[pre_i].mean(); k = b - a
        if k >= 3 and lam * k > 3.0:
            strict = lam * k > 6.9; tail = b == len(s)
            if strict: flag[a:b] = True
            (tail_rows if tail else rows).append((st, sc, od[a], k, lam, strict))
cols = ["store", "style_color", "start", "days", "rate", "strict"]
A = pd.DataFrame(rows, columns=cols); At = pd.DataFrame(tail_rows, columns=cols)
ret = S.q.sum()
print("\n== A) store x item zero runs (open days, between first and last sale at the store)")
print("store-item pairs with >= 2 selling days:", DI.groupby(level=[0, 1]).ngroups, "| live store-item-days:", live_days)
for nm, x in [("strict P<0.1%", A[A.strict]), ("loose P<5%", A)]:
    print(f"  {nm:14s} runs {len(x):6d} | flagged share of live store-item-days {x.days.sum() / live_days:.3f} | est. lost share of retail units {(x.days * x.rate).sum() / ret:.3f} | median run {x.days.median():.0f} days")
print(f"  tail runs after the last sale (stock-out or delisted): {len(At)} | if all stock-outs, lost to end of data would add more; median rate {At.rate.median():.2f}/day")
print("  by store (loose):"); print(A.assign(lost=A.days * A.rate).groupby("store").agg(runs=("days", "size"), days=("days", "sum"), lost=("lost", "sum")).join(S.groupby("store").q.sum().rename("units")).assign(lost_share=lambda t: (t.lost / t.units).round(3)).sort_values("units", ascending=False).to_string())

# ---- B) store x item x size weekly
S["wk"] = S.d.dt.to_period("W").dt.start_time
W = S.groupby(["store", "style_color", "wk", "size"]).q.sum().reset_index()
def size_gaps(W, keys, bar):
    tot = W.groupby(keys + ["wk"]).q.sum().rename("tot").reset_index()
    sh = (W.groupby(keys + ["size"]).q.sum() / W.groupby(keys).q.sum()).rename("share").reset_index(); core = sh[sh.share >= 0.15]
    X = tot.merge(core, on=keys).merge(W, on=keys + ["wk", "size"], how="left").fillna({"q": 0})
    X["exp"] = (X.tot - X.q) * X.share / (1 - X.share); b = X[(X.q == 0) & (X.exp >= bar)]
    return len(tot), b.groupby(keys + ["wk"]).ngroups, b.exp.sum() / W.q.sum()
print("\n== B) broken core size in a week")
for bar in [3, 7]:
    n, k, lost = size_gaps(W, ["store", "style_color"], bar)
    print(f"  stores, implied >= {bar}: store-item-weeks {n} | with a broken core size {k} ({k / n:.1%}) | est. lost share of retail units {lost:.3f}")
O = con.execute("SELECT style_color, date_trunc('week', ny_date) wk, size, sum(qty_ordered) q FROM read_parquet('Dataset.parquet') WHERE NOT cancelled AND class='ONLINE' GROUP BY ALL").fetchdf()
n, k, lost = size_gaps(O, ["style_color"], 3)
print(f"  online, implied >= 3: item-weeks {n} | broken {k} ({k / n:.1%}) | est. lost share of online units {lost:.3f}")
A.to_csv("data/store_oos_runs.csv", index=False)

# ---- C) do the flagged runs behave like stock-outs? (strict runs, interior only)
# (1) the same item elsewhere (other stores + online + drop) during the run vs its previous 28 days: ~1 = store-specific gap, << 1 = demand fell
# (2) the store's own rate in the 14 days after the run vs the 28 days before: ~1 = restocked
# (3) Poisson calibration: share of single open days with 0 sales, observed vs Poisson at the 28-day mean (overdispersion inflates runs)
print("\n== C) checks on strict interior runs")
ALL = con.execute("SELECT style_color, ny_date::DATE d, store, class, sum(qty_ordered) q FROM read_parquet('Dataset.parquet') WHERE NOT cancelled GROUP BY ALL").fetchdf(); ALL["d"] = pd.to_datetime(ALL.d)
ALL["store"] = ALL.store.where(ALL["class"] == "RETAIL", "_nonretail")
cum = {sc: g for sc, g in ALL.groupby("style_color")}
R = A[A.strict].sample(min(4000, int(A.strict.sum())), random_state=0)
el, post = [], []
for st, sc, a, k, lam in zip(R.store, R.style_color, R.start, R.days, R.rate):
    g = cum[sc]; o = g[g.store != st]
    end = a + pd.Timedelta(days=int(k * 7 / 6) + 1)                    # open days ~ calendar days for most stores
    during = o[(o.d >= a) & (o.d < end)].q.sum() / max((end - a).days, 1); before = o[(o.d >= a - pd.Timedelta(days=28)) & (o.d < a)].q.sum() / 28
    if before > 0: el.append(during / before)
    s = g[g.store == st]; aft = s[(s.d >= end) & (s.d < end + pd.Timedelta(days=14))].q.sum() / 14
    post.append(aft / lam)
el, post = np.array(el), np.array(post)
print(f"  same item elsewhere, run / previous 28 days: median {np.median(el):.2f} | IQR {np.percentile(el, 25):.2f}-{np.percentile(el, 75):.2f} | share >= 0.8: {(el >= 0.8).mean():.2f}")
print(f"  store's own rate after the run / before: median {np.median(post):.2f} | share >= 0.5: {(post >= 0.5).mean():.2f}")
# Poisson calibration on unflagged days
cal = []
for (st, sc), v in list(DI.groupby(level=[0, 1]))[:3000]:
    od = pd.DatetimeIndex(np.sort(open_days[st])); s = v.droplevel([0, 1]).reindex(od, fill_value=0).values.astype(float)
    nz = np.flatnonzero(s > 0)
    if len(nz) < 30: continue
    s = s[nz[0]:nz[-1] + 1]; m = pd.Series(s).rolling(28).mean().shift(1).values
    ok = ~np.isnan(m) & (m > 0.5); cal.append(np.c_[s[ok] == 0, np.exp(-m[ok]), m[ok]])
cal = np.vstack(cal); bins = pd.cut(cal[:, 2], [0.5, 1, 2, 4, 1e9])
print("  zero-day share, observed vs Poisson, by 28-day mean rate:"); print(pd.DataFrame({"observed": pd.Series(cal[:, 0]).groupby(bins).mean(), "poisson": pd.Series(cal[:, 1]).groupby(bins).mean(), "days": pd.Series(cal[:, 0]).groupby(bins).size()}).round(3).to_string())

# ---- D) a more conservative loss estimate: expected store units in a run = the store's share of the item's network sales over the
# previous 28 days x the item's network sales during the run (follows demand changes elsewhere), for runs of >= 7 and >= 14 open days.
# Also the share of retail units an always-open estimate would add, over the last 6 months, and that as a share of all units.
print("\n== D) share-based loss estimate (strict interior runs)")
net = ALL.groupby(["style_color", "d"]).q.sum()
st_d = ALL[ALL.store != "_nonretail"].groupby(["store", "style_color", "d"]).q.sum()
res = []
for st, sc, a, k in zip(A[A.strict].store, A[A.strict].style_color, A[A.strict].start, A[A.strict].days):
    end = a + pd.Timedelta(days=int(k * 7 / 6) + 1); b0 = a - pd.Timedelta(days=28)
    n = net.loc[sc]; s = st_d.loc[(st, sc)]
    nb = n[(n.index >= b0) & (n.index < a)].sum(); sb = s[(s.index >= b0) & (s.index < a)].sum()
    if nb <= 0: continue
    res.append((a, k, sb / nb * n[(n.index >= a) & (n.index < end)].sum()))
D_ = pd.DataFrame(res, columns=["start", "days", "lost"])
tot_units = ALL.q.sum(); recent = ALL.d >= "2026-01-01"
for mn in [3, 7, 14]:
    x = D_[D_.days >= mn]
    print(f"  runs >= {mn:2d} open days: {len(x):6d} | lost share of retail units {x.lost.sum() / ret:.3f} | of all units {x.lost.sum() / tot_units:.3f}"
          f" | 2026 H1: of retail {x[x.start >= '2026-01-01'].lost.sum() / ALL[recent & (ALL.store != '_nonretail')].q.sum():.3f}, of all {x[x.start >= '2026-01-01'].lost.sum() / ALL[recent].q.sum():.3f}")
