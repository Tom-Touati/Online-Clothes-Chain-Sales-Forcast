"""Rolling-origin test of the M5 ideas that can transfer: loss, features, direct vs pooled horizons, partial pooling, equal-weight ensembles.
13 origins (Feb-2025 .. Feb-2026), each forecasting the next 4 months. Scored on the 4-month total per item."""
import pandas as pd, numpy as np, lightgbm as lgb, warnings, sys, time
warnings.filterwarnings("ignore")
F = pd.read_parquet("data/features.parquet")
T = set(pd.read_csv("data/targets.csv").style_color)
CATS = ["ptype", "macro"] + [c for c in F.columns if c.startswith("a_")]
for c in CATS: F[c] = F[c].astype("category")
OCC = [c for c in F.columns if c.startswith("occ_")]
BASE = [f"lagr{k}" for k in range(1, 7)] + ["lvl", "age_m", "ptype", "tmonth", "h", "color_num", "own_sea"] + OCC
RECENT = ["f_r7", "f_r28", "trend91", "peak_r", "days_live"]
CHAN = ["sh_ret", "sh_drop", "sh_tt", "disc_item", "disc_rel"]
LFL = ["lfl_type", "lfl_macro", "lfl_occ"]
ATTR = [c for c in F.columns if c.startswith("a_")]
FULL = BASE + RECENT + CHAN + LFL + ATTR
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=4)

def fit_predict(tr, te, cols, loss):
    if loss == "l1log":       # v1: median of the log ratio
        m = lgb.LGBMRegressor(objective="l1", **P).fit(tr[cols], np.log((tr.y + 1) / (tr.level + 1)))
        return ((te.level + 1) * np.exp(m.predict(te[cols])) - 1).clip(lower=0)
    if loss == "l2log":       # mean of the log ratio + smearing retransformation
        z = np.log((tr.y + 1) / (tr.level + 1)); m = lgb.LGBMRegressor(objective="l2", **P).fit(tr[cols], z)
        smear = np.exp(z - m.predict(tr[cols])).mean()
        return ((te.level + 1) * np.exp(m.predict(te[cols])) * smear - 1).clip(lower=0)
    if loss == "tw_units":    # M5 backbone: Tweedie on units, recent level as offset
        m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2, **P).fit(tr[cols], tr.y, init_score=np.log(tr.level))
        return m.predict(te[cols]) * te.level
    if loss == "tw_ratio":    # Tweedie on the ratio: every item counts equally
        m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2, **P).fit(tr[cols], tr.y / tr.level)
        return m.predict(te[cols]) * te.level
    if loss == "flat": return te.level * 1.0

def variant(name, tr, te):
    if name == "flat (91-day run-rate)": return fit_predict(tr, te, None, "flat")
    if name.startswith("loss:"): return fit_predict(tr, te, BASE, name.split(":")[1])
    feats = {"feat:base": BASE, "feat:+recent": BASE + RECENT, "feat:+recent+chan": BASE + RECENT + CHAN, "feat:+recent+chan+lfl": BASE + RECENT + CHAN + LFL, "feat:full(+attr)": FULL}
    if name in feats: return fit_predict(tr, te, feats[name], LOSS)
    if name == "struct:direct per horizon":
        out = pd.Series(index=te.index, dtype=float)
        for h in range(1, 5): out[te.h == h] = fit_predict(tr[tr.h == h], te[te.h == h], [c for c in FEAT if c != "h"], LOSS)
        return out
    if name == "struct:partial pool by macro":
        out = pd.Series(index=te.index, dtype=float)
        for g in te.macro.unique():
            trg = tr[tr.macro == g]; sel = te.macro == g
            out[sel] = fit_predict(trg if len(trg) >= 1500 else tr, te[sel], FEAT, LOSS)
        return out
    if name == "struct:recent origins x2 weight":
        m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2, **P).fit(tr[FEAT], tr.y / tr.level, sample_weight=np.where(tr.o_ord >= tr.o_ord.max() - 6, 2.0, 1.0))
        return m.predict(te[FEAT]) * te.level

def score(pred, te, idx=None):
    g = pd.DataFrame({"p": pred, "a": te.y, "sc": te.style_color}).groupby("sc").sum()
    if idx is not None: g = g[g.index.isin(idx)]
    if len(g) == 0: return None
    ape = (g.p - g.a).abs() / g.a.clip(lower=1)
    return dict(n=len(g), wape=(g.p - g.a).abs().sum() / g.a.sum(), bias=g.p.sum() / g.a.sum() - 1, medape=ape.median(), w20=(ape <= .2).mean(), w50=(ape <= .5).mean())

ORIGINS = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
LOSS, FEAT = "tw_ratio", FULL
stage = sys.argv[1] if len(sys.argv) > 1 else "all"
names = ["flat (91-day run-rate)", "loss:l1log", "loss:l2log", "loss:tw_units", "loss:tw_ratio",
         "feat:+recent", "feat:+recent+chan", "feat:+recent+chan+lfl", "feat:full(+attr)",
         "struct:direct per horizon", "struct:partial pool by macro", "struct:recent origins x2 weight"]
res, preds = [], {}
t0 = time.time()
for o in ORIGINS:
    oo = pd.Period(o).ordinal
    tr = F[(F.o_ord + F.h <= oo) & F.y.notna()]; te = F[F.origin == o]
    for nm in names:
        p = variant(nm, tr, te); preds[(o, nm)] = pd.Series(np.asarray(p), index=te.index)
    ens = {"ens:pooled+direct+macro (equal wt)": ["feat:full(+attr)", "struct:direct per horizon", "struct:partial pool by macro"],
           "ens:+l2log+tw_units base (5 models)": ["feat:full(+attr)", "struct:direct per horizon", "struct:partial pool by macro", "loss:l2log", "loss:tw_units"]}
    for nm, parts in ens.items(): preds[(o, nm)] = sum(preds[(o, k)] for k in parts) / len(parts)
    for nm in names + list(ens):
        for seg, idx in [("all", None), ("targets", T)]:
            s = score(preds[(o, nm)], te, idx)
            if s: res.append(dict(origin=o, model=nm, seg=seg, **s))
    print(o, len(tr), len(te) // 4, round(time.time() - t0), "s", flush=True)
R = pd.DataFrame(res); R.to_csv("data/m5_experiments_by_fold.csv", index=False)
order = names + ["ens:pooled+direct+macro (equal wt)", "ens:+l2log+tw_units base (5 models)"]
for seg in ["all", "targets"]:
    g = R[R.seg == seg].groupby("model")
    S = pd.DataFrame({"wape_mean": g.wape.mean(), "wape_std": g.wape.std(), "wape_worst": g.wape.max(), "bias_mean": g.bias.mean(), "bias_std": g.bias.std(),
                      "w20_mean": g.w20.mean(), "w50_mean": g.w50.mean(), "w50_min": g.w50.min()}).loc[order].round(3)
    print(f"\n=== {seg}: mean / std over {len(ORIGINS)} origins (4-month total per item)"); print(S.to_string())
    for o in ["2025-06", "2026-02"]:
        print(f"--- fold {o} ({seg})"); print(R[(R.seg == seg) & (R.origin == o)].set_index("model").loc[order, ["n", "wape", "bias", "medape", "w20", "w50"]].round(3).to_string())
b = R[(R.seg == "all") & (R.model == "ens:pooled+direct+macro (equal wt)")].set_index("origin")[["n", "wape", "bias", "w20", "w50"]].round(3); print("\n=== ensemble by origin (all items)"); print(b.to_string())
# keep ensemble predictions for the confidence study
out = []
for o in ORIGINS:
    te = F[F.origin == o]; out.append(pd.DataFrame({"origin": o, "style_color": te.style_color, "h": te.h, "pred": preds[(o, "ens:pooled+direct+macro (equal wt)")], "y": te.y}))
pd.concat(out).to_parquet("data/ens_backtest_preds.parquet")
