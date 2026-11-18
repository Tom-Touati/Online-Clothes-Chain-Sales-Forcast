"""Review of v1 March-25 misses by segment + March site/channel exploration."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
from rules_v1 import load
it, di, st = load()
sb = pd.read_csv("feedback/scoreboard.csv")
sb = sb.join(it[["product_type", "seg", "grp", "launch", "occasion"]], on="style_color")
sb["age"] = ((pd.Timestamp("2025-03-01") - sb.launch).dt.days / 30.44).clip(upper=13).fillna(0)
sb["ageb"] = pd.cut(sb.age, [-1, 1, 2, 3, 6, 12, 99])
# channel mix in Feb
f = di[di.m == pd.Period("2025-02")].groupby("style_color")[["units", "web_units", "retail", "other"]].sum()
sb = sb.join((f.web_units / f.units.clip(lower=1)).rename("webshare"), on="style_color")
sb["webb"] = pd.cut(sb.webshare, [-0.01, .3, .5, .7, .85, 1.01])
sb["volb"] = pd.qcut(sb.actual.rank(method="first"), 5, labels=False)
sb["predb"] = pd.qcut(sb.pred.rank(method="first"), 5, labels=False)
def agg(g):
    return pd.Series(dict(n=len(g), pred=g.pred.sum(), act=g.actual.sum(), ratio=g.actual.sum() / max(g.pred.sum(), 1),
                          wape=(g.pred - g.actual).abs().sum() / max(g.actual.sum(), 1),
                          med_lr=np.median(np.log((g.actual + 5) / (g.pred + 5)))))
print("overall", agg(sb).round(3).to_dict())
for k in ["grp", "ageb", "webb", "predb"]:
    print("\n==", k); print(sb.groupby(k, observed=True).apply(agg).round(3))
t = sb.groupby("product_type").apply(agg)
print("\n== type (n>=5)"); print(t[t.n >= 5].sort_values("act", ascending=False).round(3))
s = sb.groupby("seg").apply(agg); print("\n== seg LS"); print(s[s.index.str.contains("LS")].round(3))
# occasion tags
tags = sb.occasion.fillna("").str.split("; ").explode()
tg = sb.loc[tags.index].assign(tag=tags.values)
o = tg.groupby("tag").apply(agg); print("\n== occasion (n>=8)"); print(o[o.n >= 8].round(3))
# March site-level
ds = pd.read_parquet("data/daily_site.parquet"); ds["m"] = ds.date.dt.to_period("M")
ds["wdisc"] = ds.web_discount
mm = ds.groupby("m").agg(days=("date", "nunique"), units=("units", "sum"), web=("web_units", "sum"),
                         retail=("retail", "sum"), drop=("drop", "sum"), disc=("web_discount", "mean"))
for c in ["units", "web", "retail", "drop"]:
    mm[c + "_pd"] = (mm[c] / mm.days).round(0)
print("\n== site monthly per day"); print(mm.drop(columns=["units", "web", "retail", "drop"]).round(3))
w = ds[ds.date >= "2025-01-15"].set_index("date")[["units", "web_units", "retail", "drop", "web_discount"]]
w = w.resample("W").agg({"units": "mean", "web_units": "mean", "retail": "mean", "drop": "mean", "web_discount": "mean"})
print("\n== weekly avg/day 2025"); print(w.round(2))
w24 = ds[(ds.date >= "2024-02-01") & (ds.date <= "2024-08-15")].set_index("date")[["units", "web_units", "retail", "drop", "web_discount"]].resample("W").mean()
print("\n== weekly avg/day 2024 spring"); print(w24.round(2))
dc = pd.read_parquet("data/daily_channel.parquet"); dc["m"] = dc.date.dt.to_period("M")
cm = dc.groupby(["m", "channel"]).units.sum().unstack().fillna(0)
print("\n== channel monthly"); print(cm.tail(15).round(0))
dst = st.assign(m=st.date.dt.to_period("M")).groupby(["m", "store"]).units.sum().unstack()
print("\n== stores monthly"); print(dst.tail(16))
