"""Round 3 - ideas from the broader competition review, on the same 13 rolling origins.
B0 structured seasonal-ratio baseline | launch-curve analogs for young items | style (sibling colour) features | blends with the baseline |
downward shade fitted on earlier origins | confidence from hit-rate models incl. model disagreement."""
import pandas as pd, numpy as np, lightgbm as lgb, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore"); pd.set_option("display.width", 230)
F = pd.read_parquet("data/features.parquet")
T = set(pd.read_csv("data/targets.csv").style_color)
for c in ["ptype", "macro"] + [c for c in F.columns if c.startswith("a_")]: F[c] = F[c].astype("category")
# style-level (sibling colour) features, computed per origin from rows known at the origin
F["style"] = F.style_color.str.split("_").str[0]
H1 = F[F.h == 1].copy(); H1["lw"] = H1.level * H1.f_r28
g = H1.groupby(["origin", "style"]); H1["n_sib"] = g["level"].transform("size"); H1["st_level"] = g["level"].transform("sum"); H1["st_lw"] = g["lw"].transform("sum"); H1["st_age"] = g["age_m"].transform("max")
H1["sh_style"] = H1.level / H1.st_level
H1["sib_mom"] = np.where(H1.n_sib > 1, (H1.st_lw - H1.lw) / (H1.st_level - H1.level).clip(lower=1e-9), np.nan)
H1["age_gap"] = H1.st_age - H1.age_m
HIER = ["n_sib", "sh_style", "sib_mom", "age_gap"]
F = F.merge(H1[["origin", "style_color"] + HIER], on=["origin", "style_color"], how="left")
F["rlog"] = np.log((F.y + 1) / (F.level + 1)); F["ageb"] = pd.cut(F.age_m, [-1, 3, 6, 12, 1e3], labels=False)
OCC = [c for c in F.columns if c.startswith("occ_")]
BASE = [f"lagr{k}" for k in range(1, 7)] + ["lvl", "age_m", "ptype", "tmonth", "h", "color_num", "own_sea"] + OCC
FULL = BASE + ["f_r7", "f_r28", "trend91", "peak_r", "days_live", "sh_ret", "sh_drop", "sh_tt", "disc_item", "disc_rel", "lfl_type", "lfl_macro", "lfl_occ"] + [c for c in F.columns if c.startswith("a_")]
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=4)
def l1(tr, te, cols):
    m = lgb.LGBMRegressor(objective="l1", **P).fit(tr[cols], tr.rlog); return ((te.level + 1) * np.exp(m.predict(te[cols])) - 1).clip(lower=0)
back = lambda te, r: ((te.level + 1) * np.exp(r) - 1).clip(lower=0)

def b0(tr, te):
    """recent level x like-for-like type seasonal ratio x median age effect (age bucket x horizon), all from data before the origin"""
    t = tr[tr.lfl_type.notna()]
    if t.groupby("h").size().reindex([1, 2, 3, 4]).fillna(0).min() < 100 or te.lfl_type.isna().all(): return None
    res = t.rlog - t.lfl_type
    cell = res.groupby([t.ageb, t.h]).agg(["median", "size"]); byh = res.groupby(t.h).median()
    key = list(zip(te.ageb, te.h)); adj = np.array([cell.loc[k, "median"] if k in cell.index and cell.loc[k, "size"] >= 30 else byh[k[1]] for k in key])
    return back(te, te.lfl_type.values + adj)

def analog(tr, te):
    """young items: median seasonally-adjusted ratio of past items of the same category group at the same age (+-1 month) and horizon"""
    t = tr[tr.lfl_type.notna() & (tr.age_m <= 8)]
    if len(t) < 300: return None
    res = (t.rlog - t.lfl_type).values; out = np.full(len(te), np.nan)
    for i, (mac, age, h, s) in enumerate(zip(te.macro.values, te.age_m.values, te.h.values, te.lfl_type.values)):
        if age > 6: continue
        m = (t.h.values == h) & (np.abs(t.age_m.values - age) <= 1)
        mm = m & (t.macro.values == mac)
        use = mm if mm.sum() >= 20 else m
        if use.sum() >= 20: out[i] = s + np.median(res[use])
    return pd.Series(back(te, out), index=te.index)

def tot(p, te): return pd.DataFrame({"p": np.asarray(p), "a": te.y.values, "sc": te.style_color.values}).groupby("sc").sum()
def score(p, te, keep=None):
    g = tot(p, te); g = g[g.p.notna()]
    if keep is not None: g = g[g.index.isin(keep)]
    if len(g) < 10: return None
    ape = (g.p - g.a).abs() / g.a.clip(lower=1)
    return dict(n=len(g), wape=(g.p - g.a).abs().sum() / g.a.sum(), bias=g.p.sum() / g.a.sum() - 1, w20=(ape <= .2).mean(), w50=(ape <= .5).mean())

ORIGINS = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
res, items = [], []
for o in ORIGINS:
    oo = pd.Period(o).ordinal; tr = F[(F.o_ord + F.h <= oo) & F.y.notna()]; te = F[F.origin == o]
    pb, pf, ph = l1(tr, te, BASE), l1(tr, te, FULL), l1(tr, te, FULL + HIER)
    pr = {"ens (L1 base + L1 full) [current best]": (pb + pf) / 2, "L1 full + style features": ph, "ens (base + full + style)": (pb + pf + ph) / 3}
    e = pr["ens (L1 base + L1 full) [current best]"]
    p0 = b0(tr, te)
    if p0 is not None:
        pr["B0 level x like-for-like season x age"] = p0; pr["blend 75% ens + 25% B0"] = 0.75 * e + 0.25 * p0; pr["blend 50% ens + 50% B0"] = 0.5 * e + 0.5 * p0
    # shade: one multiplier chosen on earlier origins whose actuals are already known, to maximise the +-20% hit rate
    known = [x for x in items if pd.Period(x.origin.iloc[0]).ordinal + 4 <= oo]
    if len(known) >= 2:
        K = pd.concat(known); grid = np.round(np.arange(0.80, 1.101, 0.01), 2)
        hit = [(np.abs(s * K.ens - K.a) / K.a.clip(lower=1) <= .2).mean() for s in grid]; s = grid[int(np.argmax(hit))]
        pr["shade: ens x multiplier fitted on earlier origins"] = e * s; shade = s
    else: shade = np.nan
    young = set(te.style_color[te.age_m <= 6])
    pa = analog(tr, te)
    for k, v in pr.items():
        for seg, keep in [("all", None), ("targets", T), ("young<=6m", young)]:
            sc = score(v, te, keep)
            if sc: res.append(dict(origin=o, model=k, seg=seg, shade=shade, **sc))
    if pa is not None:
        for k, v in {"analog launch curve": pa, "blend 50% ens + 50% analog": 0.5 * e + 0.5 * pa}.items():
            sc = score(v.where(pa.notna()), te, young)
            if sc: res.append(dict(origin=o, model=k, seg="young<=6m", shade=shade, **sc))
        sc = score(e.where(pa.notna()), te, young); res.append(dict(origin=o, model="ens on the same young items", seg="young<=6m", shade=shade, **sc))
    it = tot(e, te).rename(columns={"p": "ens"}); it["base"] = tot(pb, te).p; it["full"] = tot(pf, te).p; it["b0"] = tot(p0, te).p if p0 is not None else np.nan
    it["origin"] = o; items.append(it.reset_index())
    print(o, "shade", shade, flush=True)
R = pd.DataFrame(res); R.to_csv("data/round3_by_fold.csv", index=False)
I = pd.concat(items); I.to_parquet("data/round3_item_totals.parquet")
ref = R[R.model == "ens (L1 base + L1 full) [current best]"].set_index(["seg", "origin"]).wape
for seg in ["all", "targets", "young<=6m"]:
    x = R[R.seg == seg]; gg = x.groupby("model", sort=False)
    S = pd.DataFrame({"origins": gg.size(), "wape": gg.wape.mean(), "wape_std": gg.wape.std(), "bias": gg.bias.mean(), "w20": gg.w20.mean(), "w50": gg.w50.mean()})
    # paired comparison with the current best on the same origins
    S["ens_wape_same_origins"] = [ref[seg].reindex(x[x.model == m].origin).mean() for m in S.index]
    S["origins_beating_ens"] = [int((x[x.model == m].set_index("origin").wape < ref[seg].reindex(x[x.model == m].origin).values).sum()) for m in S.index]
    print(f"\n=== {seg}"); print(S.round(3).to_string())
print("\nshade multipliers chosen by origin:", R.drop_duplicates("origin").set_index("origin").shade.to_dict())

# ---- confidence: P(within 20%) and P(off by more than 50%), trained on earlier origins only
E = I.merge(F[F.h == 1].drop(columns=["y", "h"]), left_on=["origin", "sc"], right_on=["origin", "style_color"])
E["ape"] = (E.ens - E.a).abs() / E.a.clip(lower=1); E["hit20"] = (E.ape <= .2).astype(int); E["miss50"] = (E.ape > .5).astype(int)
E["dis"] = np.abs(np.log((E.base + 1) / (E.full + 1))); E["dis_b0"] = np.abs(np.log((E.ens + 1) / (E.b0 + 1)))
E["tagged"] = (E[OCC].sum(axis=1) > 0).astype(int); E["vol"] = E[[f"lagr{k}" for k in range(1, 7)]].std(axis=1).fillna(0)
E["young3"] = (E.age_m <= 3).astype(int); E["turn"] = E.lfl_type.abs().fillna(0); E["f28"] = E.f_r28.abs(); E["mult"] = np.abs(np.log((E.ens + 1) / (4 * E.level + 1)))
cf = ["lvl", "age_m", "young3", "tagged", "vol", "dis", "f28", "mult", "sh_ret", "sh_drop", "turn"]
out = []
for o in ORIGINS:
    oo = pd.Period(o).ordinal; tr = E[E.o_ord + 4 <= oo]; te = E[E.origin == o].copy()
    if tr.origin.nunique() < 3: continue
    for tgt in ["hit20", "miss50"]:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500)).fit(tr[cf].fillna(0), tr[tgt]); te["p_" + tgt] = m.predict_proba(te[cf].fillna(0))[:, 1]
    out.append(te)
O = pd.concat(out)
print(f"\n=== confidence (logistic, trained on earlier origins; scored on {O.origin.nunique()} later origins, {len(O)} forecasts)")
print("base rates: within20", round(O.hit20.mean(), 3), "| off by >50%", round(O.miss50.mean(), 3))
print("P(within 20%) >= 0.5:", int((O.p_hit20 >= .5).sum()), "items; actual hit rate", round(O[O.p_hit20 >= .5].hit20.mean(), 3) if (O.p_hit20 >= .5).any() else "n/a")
for thr in (0.45, 0.40): s = O[O.p_hit20 >= thr]; print(f"P(within 20%) >= {thr}: {len(s)} items ({len(s)/len(O):.1%}); actual within20 {s.hit20.mean():.3f}, within50 {1 - s.miss50.mean():.3f}")
for thr in (0.40, 0.35): s = O[O.p_miss50 >= thr]; print(f"P(off >50%) >= {thr}: {len(s)} items ({len(s)/len(O):.1%}); actual off>50% {s.miss50.mean():.3f}")
print("calibration by decile of P(within 20%):"); print(O.groupby(pd.qcut(O.p_hit20, 10, labels=False)).agg(p=("p_hit20", "mean"), actual20=("hit20", "mean"), actual_miss50=("miss50", "mean")).round(3).T.to_string())
print("by disagreement between the two models (quintiles):"); print(O.groupby(pd.qcut(O.dis, 5, labels=False)).agg(dis=("dis", "mean"), within20=("hit20", "mean"), miss50=("miss50", "mean")).round(3).T.to_string())
co = pd.Series(m[-1].coef_[0], index=cf).round(2); print("drivers of being off by >50% (standardised logistic coefficients):", co.sort_values().to_dict())
