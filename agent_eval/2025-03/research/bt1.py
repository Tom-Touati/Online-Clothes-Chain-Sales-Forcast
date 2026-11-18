"""Backtests of rule variants.
 A) honest out-of-time: origin 2025-02-28 (data <= origin only), score March-25 (1 month).
 B) oracle-level, 2024 origins: factors measured on the target months themselves (all items), so level
    and seasonality are in-sample; tests base/ramp/shrink choices. Scored on 4-month totals, also scaled."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import rules as R
pd.set_option("display.width", 220)
it, di, st = R.load()
mon = di.groupby(["style_color", "m"]).units.sum()
BASEP = dict(R.P)

def score(pred, act):
    e = (pred - act).abs().sum() / act.sum()
    s = pred * act.sum() / pred.sum()
    es = (s - act).abs().sum() / act.sum()
    r = pred / act.replace(0, np.nan)
    return dict(wape=round(e, 4), bias=round(pred.sum() / act.sum() - 1, 4), wape_sc=round(es, 4),
                w20=round(((r - 1).abs() <= .2).mean(), 3), w50=round(((r - 1).abs() <= .5).mean(), 3))

def run(origin, nm, oracle):
    o = pd.Timestamp(origin); B = o.to_period("M")
    items = [s for s in it.index[(it.launch <= o)] if mon.get((s, B), 0) > 0 or mon.reindex([(s, B - 1)]).fillna(0).iloc[0] > 0]
    T = [B + h for h in range(1, nm + 1)]
    if oracle:
        f = R.seasonal_factors(it, di, st, B + 12, [B + 12 + h for h in range(1, 5)])
        # forecast() looks factors up at t-12: shift the in-sample months back by 12
        for k in ["seg_rel", "grp_rel"]:
            f[k] = f[k].rename(columns=lambda c: c - 12)
        for k in ["web", "ret"]:
            f[k] = f[k].rename(index=lambda c: c - 12)
        P = R.forecast(origin, items, it, di[di.date <= o], st[st.date <= o], factors=f)
    else:
        P = R.forecast(origin, items, it, di, st)
    P = P[[str(t) for t in T]]
    A = pd.DataFrame({str(t): mon.reindex(pd.MultiIndex.from_product([items, [t]])).fillna(0).values for t in T}, index=items)
    return P, A

VARIANTS = {
    "v2 base (1.25/1.25/1.1/1.1)": {},
    "gradedA 1.8/1.5/1.2/1.1": dict(RAMP0=1.8, RAMP1=1.5, RAMP2=1.2),
    "gradedB 1.6/1.4/1.15/1.1": dict(RAMP0=1.6, RAMP1=1.4, RAMP2=1.15),
    "gradedC 2.0/1.6/1.25/1.15": dict(RAMP0=2.0, RAMP1=1.6, RAMP2=1.25, RAMP_H2=1.15),
    "gradedB + POW .05": dict(RAMP0=1.6, RAMP1=1.4, RAMP2=1.15, POW=0.05),
    "gradedB + POW .10": dict(RAMP0=1.6, RAMP1=1.4, RAMP2=1.15, POW=0.10),
    "gradedA + POW .05": dict(RAMP0=1.8, RAMP1=1.5, RAMP2=1.2, POW=0.05),
}
rows = []
for name, v in VARIANTS.items():
    R.P = dict(BASEP, **v)
    P, A = run("2025-02-28", 1, oracle=False)
    rows.append(dict(variant=name, test="OOT Feb25->Mar", **score(P.sum(1), A.sum(1))))
    res = []
    for o in ["2024-05-31", "2024-06-30", "2024-07-31", "2024-08-31", "2024-09-30", "2024-10-31", "2024-11-30"]:
        P, A = run(o, 4, oracle=True)
        res.append(score(P.sum(1), A.sum(1)))
    rows.append(dict(variant=name, test="oracle 2024 mean", **pd.DataFrame(res).mean().round(4).to_dict()))
R.P = BASEP
print(pd.DataFrame(rows).to_string())

# age / volume diagnostics on oracle origins for v2 (actual / pred, aggregated)
diag = []
for o in []:
    nm = 1 if o == "2025-02-28" else 4
    P, A = run(o, nm, oracle=(nm == 4))
    B = pd.Timestamp(o).to_period("M")
    lm = (B.start_time - it.launch.reindex(P.index)).dt.days / 30.44
    cls = np.select([lm < 0, lm < 1, lm < 2, lm < 3], ["launched in B", "B-1", "B-2", "B-3"], "older")
    df = pd.DataFrame(dict(p=P.sum(1), a=A.sum(1), cls=cls, o=o))
    df["q"] = pd.qcut(df.p.rank(method="first"), 5, labels=False)
    diag.append(df)
D = pd.concat(diag)
g = D.groupby(["o", "cls"]).apply(lambda x: x.a.sum() / x.p.sum()).unstack()
old = g["older"]
print("\nactual/pred by launch class, relative to older items"); print(g.div(old, axis=0).round(2))
print("counts"); print(D.groupby(["o", "cls"]).size().unstack())
q = D.groupby(["o", "q"]).apply(lambda x: x.a.sum() / x.p.sum()).unstack()
print("\nactual/pred by pred quintile, relative to overall"); print(q.div(D.groupby("o").apply(lambda x: x.a.sum() / x.p.sum()), axis=0).round(2))
