"""Seasonal factors from a January base (TY and LY), and a Jan-25-origin backtest (Feb-May-25)."""
import sys, copy
import pandas as pd, numpy as np
sys.path.insert(0, "research"); sys.path.insert(0, ".")
import r3_bt as bt, rules
it, di, st = bt.it, bt.di, bt.st
for B in [pd.Period("2026-01", "M"), pd.Period("2025-01", "M")]:
    o = B.end_time.normalize()
    tg = [B + h for h in range(1, 5)]
    f = rules.seasonal_factors(it, di[di.date <= o], st[st.date <= o], B, tg)
    print(B, "seas_pow", round(f["seas_pow"], 3), "anc", f["anc"].round(3).to_dict())
    print(" grp_rel\n", f["grp_rel"].round(3).to_string())
    print(" drift", f["drift"].round(3).to_dict(), "\n web", f["web"].round(3).to_dict(),
          "\n other", f["other"].round(3).to_dict(), "\n ret", f["ret"].round(3).to_dict())
# TY vs LY group moves Dec->Jan (LFL rel. to panel)
for ms in [["2024-12", "2025-01"], ["2025-12", "2026-01"], ["2025-11", "2025-12"], ["2024-11", "2024-12"]]:
    ms = [pd.Period(m, "M") for m in ms]
    U = di[di.m.isin(ms)].pivot_table(index="style_color", columns="m", values="units", aggfunc="sum").reindex(columns=ms).fillna(0)
    g, _ = rules.lfl_chain(U, it.launch, ms, it.grp)
    a, _ = rules.lfl_chain(U, it.launch, ms, pd.Series("ALL", index=U.index))
    print(ms, "all", round(a.loc["ALL", ms[1]], 3), (g[ms[1]] / a.loc["ALL", ms[1]]).round(3).to_dict())

# Jan-25 origin backtest: items = sold in the last 28 days of Jan-25
o = pd.Timestamp("2025-01-31")
items = di[(di.date > o - pd.Timedelta(days=28)) & (di.date <= o)].groupby("style_color").units.sum()
items = items[items > 0].index.tolist()
P0 = copy.deepcopy(rules.P)
def bt_jan(over):
    rules.P.clear(); rules.P.update(P0); rules.P.update(over)
    f = rules.forecast(o, items, it, di, st)
    a = bt.act[f.columns].reindex(items).fillna(0)
    e = (f.sum(axis=1) - a.sum(axis=1)).abs()
    grp = it.grp.reindex(items)
    gr = (a.groupby(grp).sum().sum(axis=1) / f.groupby(grp).sum().sum(axis=1) * f.values.sum() / a.values.sum()).round(2)
    return dict(n=len(items), wape=round(e.sum() / a.values.sum(), 4), bias=round(f.values.sum() / a.values.sum() - 1, 3),
                w20=round((e <= 0.2 * a.sum(axis=1)).mean(), 3), w50=round((e <= 0.5 * a.sum(axis=1)).mean(), 3),
                bym=(a.sum() / f.sum()).round(3).to_dict(), grp=gr.to_dict())
for name, over in {"base": {}, "amp0.6": dict(ADAPT_AMP=False, SEAS_POW=0.6), "amp0.8": dict(ADAPT_AMP=False, SEAS_POW=0.8),
                   "amp1.0": dict(ADAPT_AMP=False, SEAS_POW=1.0), "noanc": dict(SEAS_ANCHOR=0.0),
                   "fade90": dict(AGE_FADE=0.9)}.items():
    print(name, bt_jan(over))
