"""D178 (overnight M5 goal, direction 8, user request): training-data augmentation for the D159 quantile model (eda/164 cpllm_we).
Only training rows change; every variant is tested on the real item-origins.
  agg     + style and product-type aggregate rows (level-weighted mean features, summed level and target, n_items; flag agg_lvl
          0 / 1 / 2), the hierarchy as extra training data (M5 winners trained on several aggregation levels)
  jitter  + 2 copies of each training row with the level shifted by exp(N(0, 0.35)) (lvl, cp_nb_lvl moved together; the ratio
          target unchanged): scale invariance
  bag     5 models on block-bootstrapped training origins (whole origins resampled), predictions averaged in log space
  mixup   + 1 synthetic row per training row: a convex mix (lambda ~ Beta(0.4, 0.4)) of two items of the same product type and origin,
          features and log-ratio target mixed
Writes data/q_backtest_aug_{agg,jitter,bag,mixup}.parquet; scored by eda/91_score_context.py aug data/q_backtest_wx_cpllm_we.parquet."""
import sys, numpy as np, pandas as pd, time
src = open("eda/164_q_weather.py", encoding="utf-8").read(); _a = sys.argv; sys.argv = sys.argv[:1]
exec(src[src.index("import sys"):src.index("for var in RUN:")]); sys.argv = _a
RUN = sys.argv[1:] or ["agg", "jitter", "bag", "mixup"]
NEW = CP + LL + WE; qm.BASE = B0 + NEW; qm.FULL = F0 + NEW; COLS = list(dict.fromkeys(qm.FULL + qm.BASE))
T0["style"] = T0.style_color.str.split("_").str[0]; T0["agg_lvl"] = 0
num = [c for c in COLS if c != "ptype" and T0[c].dtype.kind in "fiub" and c not in ("agg_lvl",)]
def agg(df, key, lv):
    w = df.level.fillna(0).values + 1; X = df[num].astype(float); X = X.fillna(X.mean()).fillna(0)
    g = pd.DataFrame(X.values * w[:, None], columns=num); g["origin"] = df.origin.values; g[key] = df[key].values; g["_w"] = w
    s_ = g.groupby(["origin", key]).sum(); out = s_[num].div(s_["_w"], axis=0)
    e = df.groupby(["origin", key]).agg(level=("level", "sum"), ytot=("ytot", lambda v: v.sum(min_count=1)), o_ord=("o_ord", "first"),
                                         ptype=("ptype", lambda v: v.astype(str).mode().iloc[0]))
    out = out.join(e).reset_index(); out["ptype"] = pd.Categorical(out.ptype, categories=T0.ptype.cat.categories)
    out["lvl"] = np.log1p(out.level); out["z"] = np.log((out.ytot + 1) / (4 * out.level + 1)); out["agg_lvl"] = lv; return out
rng = np.random.default_rng(0)
for var in RUN:
    out, t0 = [], time.time()
    extra = None
    if var == "agg":
        extra = pd.concat([agg(T0, "style", 1), agg(T0.assign(ptype_s=T0.ptype.astype(str)), "ptype_s", 2)], ignore_index=True)
        qm.BASE = B0 + NEW + ["agg_lvl"]; qm.FULL = F0 + NEW + ["agg_lvl"]
    else: qm.BASE = B0 + NEW; qm.FULL = F0 + NEW
    for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
        oo = pd.Period(o).ordinal; tr = T0[(T0.o_ord + 4 <= oo) & T0.ytot.notna()]; te = T0[T0.origin == o]
        if var == "agg": tr = pd.concat([tr, extra[(extra.o_ord + 4 <= oo) & extra.ytot.notna()]], ignore_index=True)
        if var == "jitter":
            cp = [tr.copy() for _ in range(2)]
            for c in cp:
                d = rng.normal(0, 0.35, len(c)); c["lvl"] = c.lvl + d; c["cp_nb_lvl"] = c.cp_nb_lvl + d
            tr = pd.concat([tr] + cp, ignore_index=True)
        if var == "mixup":
            parts = []
            for _, g in tr.groupby(["origin", "ptype"], observed=True):
                if len(g) < 2: continue
                j = rng.permutation(len(g)); lam = rng.beta(0.4, 0.4, len(g))[:, None]
                A = g[num].astype(float).fillna(0).values; Bm = A[j]; Mx = pd.DataFrame(lam * A + (1 - lam) * Bm, columns=num)
                Mx["z"] = lam[:, 0] * g.z.values + (1 - lam[:, 0]) * g.z.values[j]; Mx["ptype"] = g.ptype.values; parts.append(Mx)
            tr = pd.concat([tr, pd.concat(parts, ignore_index=True)], ignore_index=True)
        if var == "bag":
            origs = tr.origin.unique(); Z = []
            for b in range(5):
                pick = rng.choice(origs, len(origs), replace=True); trb = pd.concat([tr[tr.origin == x] for x in pick], ignore_index=True); Z.append(qm.fit_q(trb, te))
            zq = np.sort(np.mean(Z, axis=0), axis=1)
        else:
            zq = qm.fit_q(tr, te)
        R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy()
        for j, q in enumerate(qm.QS): R[f"q{int(q*100):02d}"] = np.clip((4 * te.level.values + 1) * np.exp(zq[:, j]) - 1, 0, None)
        out.append(R)
    pd.concat(out).to_parquet(f"data/q_backtest_aug_{var}.parquet"); print(var, round(time.time() - t0), "s", flush=True)
