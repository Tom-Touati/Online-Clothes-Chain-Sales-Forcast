"""Same-season backtest of the item-level allocation: origin 2024-11-30 (BF base month) -> Dec-24..Mar-25.
Full rules cannot run there (no 2023 data for the LY factors), so the level / seasonal factors are neutral and
each (season group x month) is scaled to its ACTUAL total (oracle).  This tests base blend, flash-day cap,
ramp, age, size and tag multipliers -- i.e. the within-group allocation from a Black-Friday base month.
Also: age-bucket rel. at h=1 across the scoreboard origins; tag profiles from a Nov-24 base."""
import sys
import pandas as pd, numpy as np
sys.path.insert(0, ".")
import rules as R

it, di, st = R.load()
O = pd.Timestamp("2024-11-30")
tg = [pd.Period(m) for m in ["2024-12", "2025-01", "2025-02", "2025-03"]]
nov = di[di.m == pd.Period("2024-11")].groupby("style_color").units.sum()
last28 = di[(di.date > O - pd.Timedelta(days=28)) & (di.date <= O)].groupby("style_color").units.sum()
items = sorted(set(last28[last28 >= 10].index) & set(nov.index))
act = di[di.m.isin(tg)].pivot_table(index="style_color", columns="m", values="units", aggfunc="sum")
act = act.reindex(index=items, columns=tg).fillna(0)
act.columns = [str(c) for c in act.columns]
ly = [pd.Period("2023-11")] + [t - 12 for t in tg]
one = pd.Series(1.0, index=ly)
neutral = dict(seas_pow=1.0, anc=pd.Series(dtype=float), web=one, ret=one, other=one,
               grp_rel=pd.DataFrame({t: pd.Series({"WINTER": 1.0, "SUMMER": 1.0, "CORE": 1.0}) for t in ly}),
               seg_rel=pd.DataFrame({t: pd.Series(dtype=float) for t in ly}), seg_n=pd.Series(dtype=float))
grp = it.grp.reindex(items).fillna("CORE")

def score(pred, label):
    pred = pred.copy()
    for c in act.columns:                                   # oracle: group x month totals
        f = act[c].groupby(grp).sum() / pred[c].groupby(grp).sum().clip(lower=1e-9)
        pred[c] = pred[c] * grp.map(f)
    p, a = pred.sum(axis=1), act.sum(axis=1)
    e = (p - a).abs()
    r = p / a.clip(lower=1e-9)
    mw = np.mean([(pred[c] - act[c]).abs().sum() / act[c].sum() for c in act.columns])
    print(f"{label:34s} WAPE {e.sum()/a.sum():.4f}  ±20% {((r-1).abs()<=0.2).mean():.3f}  ±50% {((r-1).abs()<=0.5).mean():.3f}  monthly {mw:.4f}")
    return p

def run(label, **kw):
    old = {k: R.P[k] for k in kw}
    R.P.update(kw)
    p = R.forecast(O, items, it, di, st, factors=neutral)
    R.P.update(old)
    return p, score(p, label)

base, pb = run("v9 allocation")
for kw in [dict(W14=0.2), dict(W14=0.6), dict(WLONG=0.0), dict(WLONG=0.4), dict(SPIKE_X=99), dict(SPIKE_CAP=1.0),
           dict(SPIKE_CAP=0.5), dict(POW=0.05), dict(RAMP0=1.3, RAMP1=1.2), dict(RAMP0=1.9, RAMP1=1.6)]:
    run(str(kw), **kw)
# diagnostics: rel by age / size / tags (4-month totals, after oracle)
a = act.sum(axis=1)
df = pd.DataFrame({"p": pb, "a": a, "grp": grp})
df["age"] = pd.cut((pd.Timestamp("2024-11-01") - it.launch.reindex(items)).dt.days / 30.44, [-9, 0, 1, 2, 4, 8, 14, 99])
df["size"] = pd.qcut(df.p.rank(method="first"), 5, labels=False)
occ = it.occasion.reindex(items).fillna("")
tier = it.tier.reindex(items).fillna("")
for col in ["age", "size"]:
    t = df.groupby(col).agg(n=("p", "size"), p=("p", "sum"), a=("a", "sum")); t["rel"] = (t.a / t.p).round(3)
    print(t.to_string())
for tag in ["Halloween", "New Year's Eve", "Valentine's", "Homecoming", "Graduation", "4th of July", "St. Patrick"]:
    h = occ.str.contains(tag, regex=False)
    if h.sum():
        x = df[h.values]
        m = {c: round(act.loc[h.values, c].sum() / base.loc[h.values, c].sum() /
                      (act.loc[~h.values, c].sum() / base.loc[~h.values, c].sum()), 2) for c in act.columns}
        print(f"tag {tag:15s} n={h.sum():3d}  4m rel {x.a.sum()/x.p.sum():.3f}  monthly (vs untagged, pre-oracle) {m}")
for tr in ["Party", "Evening", "Social Daywear", "Casual", "Basics"]:
    h = (tier == tr).values
    m = {c: round(act.loc[h, c].sum() / base.loc[h, c].sum() / (act.loc[~h, c].sum() / base.loc[~h, c].sum()), 2) for c in act.columns}
    print(f"tier {tr:15s} n={h.sum():3d} monthly rel {m}")

# scoreboard: age bucket rel at h=1 per origin
sb = pd.read_csv("feedback/scoreboard.csv")
sb = sb[sb.h == 1].join(it[["launch"]], on="style_color")
sb["age"] = pd.cut((pd.PeriodIndex(sb.origin, freq="M").start_time - sb.launch).dt.days / 30.44, [-9, 0, 1, 2, 4, 8, 14, 99])
tot = sb.groupby("origin").apply(lambda x: x.actual.sum() / x.pred.sum())
t = sb.groupby(["origin", "age"]).apply(lambda x: x.actual.sum() / x.pred.sum()).unstack()
print("\nh=1 age rel by origin"); print(t.div(tot, axis=0).round(2).to_string())
sb["size"] = sb.groupby("origin").pred.transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
t = sb.groupby(["origin", "size"]).apply(lambda x: x.actual.sum() / x.pred.sum()).unstack()
print("\nh=1 size rel by origin"); print(t.div(tot, axis=0).round(2).to_string())
occ2 = it.occasion.fillna("")
for tag in ["New Year's Eve", "Halloween", "St. Patrick", "Valentine's"]:
    x = sb[sb.style_color.map(occ2).fillna("").str.contains(tag, regex=False)]
    print(tag, (x.groupby("origin").apply(lambda y: y.actual.sum() / y.pred.sum()) / tot).round(2).to_dict())
