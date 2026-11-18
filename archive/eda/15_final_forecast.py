"""Final forecast, origin = end of June 2026, horizon Jul-Oct 2026.
Model = D009-D013 exactly as backtested: average of two pooled median-loss LightGBMs (base / full features) on log ratio to recent level.
Judgement adjustments (D030-D033) are applied on top and the raw model total is kept."""
import pandas as pd, numpy as np, lightgbm as lgb, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
F = pd.read_parquet("data/features.parquet")
tg = pd.read_csv("data/targets.csv"); T = list(tg.style_color)
for c in ["ptype", "macro"] + [c for c in F.columns if c.startswith("a_")]: F[c] = F[c].astype("category")
OCC = [c for c in F.columns if c.startswith("occ_")]
BASE = [f"lagr{k}" for k in range(1, 7)] + ["lvl", "age_m", "ptype", "tmonth", "h", "color_num", "own_sea"] + OCC
FULL = BASE + ["f_r7", "f_r28", "trend91", "peak_r", "days_live", "sh_ret", "sh_drop", "sh_tt", "disc_item", "disc_rel", "lfl_type", "lfl_macro", "lfl_occ"] + [c for c in F.columns if c.startswith("a_")]
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=4)
F["rlog"] = np.log((F.y + 1) / (F.level + 1)); F["ageb"] = pd.cut(F.age_m, [-1, 3, 6, 12, 1e3], labels=False)
O = pd.Period("2026-06"); oo = O.ordinal
tr = F[(F.o_ord + F.h <= oo) & F.y.notna()]; te = F[F.origin == str(O)].copy()
print("training rows", len(tr), "| origins", tr.origin.nunique(), "| items eligible at origin", te.style_color.nunique(), "| targets eligible", te.style_color.isin(T).sum() // 4)
missing = sorted(set(T) - set(te.style_color)); print("targets not eligible at origin:", missing)
mods = {}
for nm, cols in [("base", BASE), ("full", FULL)]:
    mods[nm] = lgb.LGBMRegressor(objective="l1", **P).fit(tr[cols], tr.rlog)
    te["p_" + nm] = ((te.level + 1) * np.exp(mods[nm].predict(te[cols])) - 1).clip(lower=0)
te["raw"] = (te.p_base + te.p_full) / 2
# structured baseline B0 for the per-item sanity check (D023)
t = tr[tr.lfl_type.notna()]; res = t.rlog - t.lfl_type
cell = res.groupby([t.ageb, t.h]).agg(["median", "size"]); byh = res.groupby(t.h).median()
adj = [cell.loc[k, "median"] if k in cell.index and cell.loc[k, "size"] >= 30 else byh[k[1]] for k in zip(te.ageb, te.h)]
te["b0"] = ((te.level + 1) * np.exp(te.lfl_type.fillna(0).values + np.array(adj)) - 1).clip(lower=0)

# sanity check: aggregate 4-month multiplier for items >=12 months old (June origins: 1.12 in 2024, 1.22 in 2025)
h1 = te[te.h == 1]; tot_raw = te.groupby("style_color").raw.sum()
mat = h1[h1.age_m >= 12].style_color
print("aggregate multiplier, items >=12m old (all eligible):", round(tot_raw[mat].sum() / (4 * h1.set_index("style_color").level[mat].sum()), 3),
      "| all eligible items:", round(tot_raw.sum() / (4 * h1.level.sum()), 3))

W = te[te.style_color.isin(T)].pivot(index="style_color", columns="h", values="raw"); W.columns = ["jul", "aug", "sep", "oct"]
W["b0_total"] = te[te.style_color.isin(T)].groupby("style_color").b0.sum()
W["base_total"] = te[te.style_color.isin(T)].groupby("style_color").p_base.sum(); W["full_total"] = te[te.style_color.isin(T)].groupby("style_color").p_full.sum()
H = h1.set_index("style_color")
for c in ["level", "age_m", "sh_ret", "sh_drop", "sh_tt", "f_r28", "f_r7", "lfl_type", "macro", "ptype", "lvl", "trend91", "sh_ret"]: W[c] = H.loc[W.index, c]
W = W.loc[:, ~W.columns.duplicated()]

# ---- S16079_DARK-GRAY: no sales since 2026-04-10 (D021). Brief: assume in stock. Forecast = last in-stock rate x like-for-like Top seasonal ratio.
M = pd.read_parquet("data/monthly_item.parquet")
sc = "S16079_DARK-GRAY"
print(sc, "monthly 2025-06..2026-06:", M.loc[sc, "2025-06":].astype(int).to_dict())
rate = M.loc[sc, ["2026-01", "2026-02", "2026-03"]].mean()          # last three months with sales before the zero tail
tops = te[(te.ptype == "Top") & (te.age_m >= 12)]
mult = tops.groupby("h").apply(lambda d: d.raw.sum() / d.level.sum())  # model's month multiplier for mature tops
W.loc[sc, ["jul", "aug", "sep", "oct"]] = (rate * mult.values)
W.loc[sc, ["level", "age_m"]] = [rate, (pd.Timestamp("2026-06-30") - pd.Timestamp("2024-01-15")).days / 30.44]
W.loc[sc, "ptype"] = "Top"; W.loc[sc, "macro"] = "top"
print(f"{sc}: rate {rate:.0f}/month from Jan-Mar 2026 x mature-top month multipliers {mult.round(2).tolist()}")
W["raw_total"] = W[["jul", "aug", "sep", "oct"]].sum(axis=1)

# ---- judgement adjustments
W["web_share"] = (1 - W.sh_ret.fillna(0) - W.sh_drop.fillna(0) - W.sh_tt.fillna(0)).clip(0, 1)
CAMP = {"aug": 0.07, "sep": 0.04}                                   # D031: new Clearance (Aug 12-19) and new late-Sep sale, half of measured deep-day lift
adjm = W[["jul", "aug", "sep", "oct"]].copy()
for m, u in CAMP.items(): adjm[m] = adjm[m] * (1 + u * W.web_share)
WC = ["S24840_LIGHT-GRAY-MELANGE", "S24847_NAVY", "S25039_YELLOW"]   # D032: post-tournament decay caps, relative to July
# D032 (revised in D033): July = late-June daily pace for the 19 tournament days + 75% of it for the 12 days after the final;
# Aug/Sep/Oct capped at 0.75 / 0.50 / 0.45 of July = mean post-event decay of older 4th of July and St. Patrick's items (eda/14_adjustment_evidence.py)
dd = pd.read_parquet("data/daily_item_chan.parquet"); dd = dd[~dd.cancelled & dd.style_color.isin(WC)]; dd["ny_date"] = pd.to_datetime(dd.ny_date)
r14 = dd[dd.ny_date >= "2026-06-17"].groupby("style_color").qty.sum() / 14
DECAY = [("aug", 0.75), ("sep", 0.50), ("oct", 0.45)]
for s in WC:
    adjm.loc[s, "jul"] = min(adjm.loc[s, "jul"], r14[s] * (19 + 0.75 * 12))
    j = adjm.loc[s, "jul"]
    for m, cap in DECAY: adjm.loc[s, m] = min(adjm.loc[s, m], cap * j)
    print(f"World Cup {s}: late-June pace {r14[s]:.0f}/day -> July {j:.0f}; model July was {W.loc[s, 'jul']:.0f}")
W[["f_jul", "f_aug", "f_sep", "f_oct"]] = adjm.round(0).values
W["final_total"] = W[["f_jul", "f_aug", "f_sep", "f_oct"]].sum(axis=1)
W["adj_pct"] = W.final_total / W.raw_total - 1

# ---- confidence (D028): logistic P(within 20%) / P(off >50%) trained on all backtest origins, scored on the targets
I = pd.read_parquet("data/round3_item_totals.parquet")
E = I.merge(F[F.h == 1].drop(columns=["y", "h"]), left_on=["origin", "sc"], right_on=["origin", "style_color"])
def cfeat(d, ens, base, full, level):
    X = pd.DataFrame(index=d.index)
    X["lvl"] = np.log1p(level); X["age_m"] = d.age_m; X["young3"] = (d.age_m <= 3).astype(int); X["tagged"] = (d[OCC].sum(axis=1) > 0).astype(int)
    X["vol"] = d[[f"lagr{k}" for k in range(1, 7)]].std(axis=1).fillna(0); X["dis"] = np.abs(np.log((base + 1) / (full + 1)))
    X["f28"] = d.f_r28.abs(); X["mult"] = np.abs(np.log((ens + 1) / (4 * level + 1))); X["sh_ret"] = d.sh_ret; X["sh_drop"] = d.sh_drop; X["turn"] = d.lfl_type.abs().fillna(0)
    return X.fillna(0)
XE = cfeat(E, E.ens, E.base, E.full, E.level); E["ape"] = (E.ens - E.a).abs() / E.a.clip(lower=1)
cm = {k: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500)).fit(XE, y) for k, y in [("hit20", (E.ape <= .2).astype(int)), ("miss50", (E.ape > .5).astype(int))]}
E["p_hit20"] = cm["hit20"].predict_proba(XE)[:, 1]; E["p_miss50"] = cm["miss50"].predict_proba(XE)[:, 1]
q_hit_top, q_miss_top = E.p_hit20.quantile(0.9), E.p_miss50.quantile(0.9)
g = E[(E.p_hit20 >= q_hit_top) & (E.level >= 3000)]
print(f"backtest: top-decile P(within20) AND level >= 3000: n={len(g)}, within20 {(g.ape <= .2).mean():.3f}, within50 {(g.ape <= .5).mean():.3f}")
g2 = E[E.level >= 3000]; print(f"backtest: level >= 3000: n={len(g2)}, within20 {(g2.ape <= .2).mean():.3f}")
tt = te[(te.h == 1) & te.style_color.isin(T)].set_index("style_color").loc[[s for s in W.index if s != sc]]
XT = cfeat(tt, W.loc[tt.index, "raw_total"], W.loc[tt.index, "base_total"], W.loc[tt.index, "full_total"], tt.level)
W["p_hit20"] = np.nan; W["p_miss50"] = np.nan
W.loc[tt.index, "p_hit20"] = cm["hit20"].predict_proba(XT)[:, 1]; W.loc[tt.index, "p_miss50"] = cm["miss50"].predict_proba(XT)[:, 1]
conf = pd.Series(2, index=W.index)
three = (W.p_miss50 >= q_miss_top) | (W.age_m < 3) | (W.level < 100) | W.index.isin(WC + [sc])
conf[three] = 3
W["b0_gap"] = np.log((W.raw_total + 1) / (W.b0_total + 1))
one = (W.p_hit20 >= q_hit_top) & (W.level >= 3000) & (W.b0_gap.abs() < 0.3) & ~three & (W.adj_pct.abs() < 0.03)   # D035: backtest 56% within 20%
conf[one] = 1; W["conf"] = conf; W["one_candidate"] = one
W["dis"] = np.abs(np.log((W.base_total + 1) / (W.full_total + 1)))
W["b0_gap"] = np.log((W.raw_total + 1) / (W.b0_total + 1))
W["m4"] = W.raw_total / (4 * W.level)
W.to_csv("data/final_forecast.csv")
print("\nconfidence counts:", W.conf.value_counts().sort_index().to_dict(), "| candidates for 1:", int(one.sum()), list(W.index[one]))
print("why 3:", dict(score=int((W.p_miss50 >= q_miss_top).sum()), young=int((W.age_m < 3).sum()), low_vol=int((W.level < 100).sum())))
print("totals: raw", int(W.raw_total.sum()), "| final", int(W.final_total.sum()), f"({W.final_total.sum() / W.raw_total.sum() - 1:+.1%})", "| June 2026 actual of targets x4:", int(M.loc[T, "2026-06"].sum() * 4))
cols = ["ptype", "age_m", "level", "jul", "aug", "sep", "oct", "raw_total", "final_total", "adj_pct", "m4", "b0_total", "b0_gap", "dis", "p_hit20", "p_miss50", "conf"]
print(W.assign(jun26=M.loc[W.index, "2026-06"]).sort_values("raw_total", ascending=False)[["jun26"] + cols].round(2).to_string())
