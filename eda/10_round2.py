"""Round 2: (a) is there drift in the multiplier across years? (b) how much error is top-level (fold-wide level) vs item-level?
(c) does anchoring the fold-wide level to last year's like-for-like ratio help? (d) L1 with the richer feature sets; mixed ensembles."""
import pandas as pd, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
F = pd.read_parquet("data/features.parquet")
T = set(pd.read_csv("data/targets.csv").style_color)
for c in ["ptype", "macro"] + [c for c in F.columns if c.startswith("a_")]: F[c] = F[c].astype("category")
OCC = [c for c in F.columns if c.startswith("occ_")]
BASE = [f"lagr{k}" for k in range(1, 7)] + ["lvl", "age_m", "ptype", "tmonth", "h", "color_num", "own_sea"] + OCC
LFL = ["lfl_type", "lfl_macro", "lfl_occ"]
FULL = BASE + ["f_r7", "f_r28", "trend91", "peak_r", "days_live", "sh_ret", "sh_drop", "sh_tt", "disc_item", "disc_rel"] + LFL + [c for c in F.columns if c.startswith("a_")]
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=4)

# (a) drift: realised 4-month multiplier (sum of next 4 months / 4 x recent level), all eligible items, by origin month and year
A = F[(F.h == 1) & F.ytot.notna()].assign(yr=lambda d: d.origin.str[:4], mo=lambda d: d.origin.str[5:])
g = A.groupby(["mo", "yr"]).apply(lambda d: d.ytot.sum() / (4 * d.level.sum())).unstack().round(2)
gm = A.groupby(["mo", "yr"]).apply(lambda d: (d.ytot / (4 * d.level)).median()).unstack().round(2)
print("== (a) realised multiplier = next-4-month units / (4 x recent monthly level). Aggregate (left) and item median (right), by origin month x year")
print(pd.concat([g, gm], axis=1, keys=["aggregate", "item median"]).to_string())
# same, restricted to items >= 12 months old at origin (removes launch-curve mix)
B = A[A.age_m >= 12]; print("items >=12 months old only, aggregate:"); print(B.groupby(["mo", "yr"]).apply(lambda d: d.ytot.sum() / (4 * d.level.sum())).unstack().round(2).to_string())

def fp(tr, te, cols, loss):
    if loss == "l1":
        m = lgb.LGBMRegressor(objective="l1", **P).fit(tr[cols], np.log((tr.y + 1) / (tr.level + 1))); return ((te.level + 1) * np.exp(m.predict(te[cols])) - 1).clip(lower=0)
    if loss == "l2":
        z = np.log((tr.y + 1) / (tr.level + 1)); m = lgb.LGBMRegressor(objective="l2", **P).fit(tr[cols], z)
        return ((te.level + 1) * np.exp(m.predict(te[cols])) * np.exp(z - m.predict(tr[cols])).mean() - 1).clip(lower=0)
    if loss == "tw":
        m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2, **P).fit(tr[cols], tr.y, init_score=np.log(tr.level)); return m.predict(te[cols]) * te.level

def score(p, te, idx=None):
    g = pd.DataFrame({"p": np.asarray(p), "a": te.y.values, "sc": te.style_color.values}).groupby("sc").sum()
    if idx is not None: g = g[g.index.isin(idx)]
    ape = (g.p - g.a).abs() / g.a.clip(lower=1)
    return dict(wape=(g.p - g.a).abs().sum() / g.a.sum(), bias=g.p.sum() / g.a.sum() - 1, w20=(ape <= .2).mean(), w50=(ape <= .5).mean())

ORIGINS = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
res = []
for o in ORIGINS:
    oo = pd.Period(o).ordinal; tr = F[(F.o_ord + F.h <= oo) & F.y.notna()]; te = F[F.origin == o]
    pr = {"l1 base": fp(tr, te, BASE, "l1"), "l1 base+lfl": fp(tr, te, BASE + LFL, "l1"), "l1 full": fp(tr, te, FULL, "l1"),
          "l2 base+lfl": fp(tr, te, BASE + LFL, "l2"), "tw base+lfl": fp(tr, te, BASE + LFL, "tw")}
    tr12 = tr[tr.o_ord >= oo - 15]                                   # train on the most recent ~12 usable origins only
    pr["l1 base+lfl, recent 12 origins only"] = fp(tr12, te, BASE + LFL, "l1") if tr12.origin.nunique() >= 6 else pr["l1 base+lfl"]
    pr["ens: l1+l2+tw (base+lfl)"] = (pr["l1 base+lfl"] + pr["l2 base+lfl"] + pr["tw base+lfl"]) / 3
    pr["ens: l1 base + l1 full"] = (pr["l1 base"] + pr["l1 full"]) / 2
    e = pr["ens: l1+l2+tw (base+lfl)"]
    pr["ORACLE: ens rescaled to true fold total"] = e * te.y.sum() / e.sum()
    # anchor: fold-wide multiplier observed one year earlier at the same origin month, on items live then (known at forecast time)
    prev = F[(F.origin == str(pd.Period(o) - 12)) & (F.h == 1)]
    if len(prev):
        r_ly = prev.ytot.sum() / (4 * prev.level.sum()); r_model = e.sum() / te.level.sum()
        pr["anchor: ens rescaled to last year's fold multiplier"] = e * r_ly / r_model
        pr["anchor: 50/50 model and last year's multiplier"] = e * (0.5 + 0.5 * r_ly / r_model)
    for k, v in pr.items():
        res.append(dict(origin=o, model=k, seg="all", **score(v, te))); res.append(dict(origin=o, model=k, seg="targets", **score(v, te, T)))
    print(o, flush=True)
R = pd.DataFrame(res); R.to_csv("data/round2_by_fold.csv", index=False)
ref = R[(R.model == "l1 base")].set_index(["seg", "origin"]).wape
for seg in ["all", "targets"]:
    x = R[R.seg == seg]; g = x.groupby("model", sort=False)
    S = pd.DataFrame({"folds": g.size(), "wape_mean": g.wape.mean(), "wape_std": g.wape.std(), "wape_worst": g.wape.max(), "bias_mean": g.bias.mean(), "bias_std": g.bias.std(), "abs_bias_mean": g.bias.apply(lambda s: s.abs().mean()),
                      "w20": g.w20.mean(), "w50": g.w50.mean(), "folds_beating_l1_base": g.apply(lambda d: int((d.set_index("origin").wape < ref[seg].reindex(d.origin).values).sum()))}).round(3)
    print(f"\n=== (b-d) {seg}: over rolling origins"); print(S.to_string())
print("\nbias by origin (all items):"); print(R[R.seg == "all"].pivot(index="origin", columns="model", values="bias")[["l1 base", "l1 base+lfl", "ens: l1+l2+tw (base+lfl)", "anchor: 50/50 model and last year's multiplier"]].round(2).to_string())
