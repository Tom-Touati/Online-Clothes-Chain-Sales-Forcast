"""D280 (fork; user comment on the error study: "give more weight to models that have a higher score before averaging, or identify a dense
cluster of higher model outputs and ignore models outside that cluster"; and "if 42% of error sits in 5% of forecasts, how come such a
low percentage is within the 20% range").
Part 1: other ways to combine the five members (rules, LightGBM ec_ns, NN, TabPFN conformal, TabPFN dyn 32), all time-honest:
  weights by score   w_m proportional to (1 / member's WAPE on earlier origins with known outcomes)^k, k = 1, 2, 4, 8; the same within age bands;
                     best single member so far
  clusters           median; trimmed mean (drop lowest and highest); densest three (the triple with the smallest log range); drop the one
                     member farthest from the median; mean of the upper three / upper four (drop the lowest); mean of the lower four
  references         arithmetic mean (the submitted stack before corrections), geometric mean, best member per item picked afterwards
Each scored before corrections and with the submitted correction multiplier (F / F0) applied on top: WAPE / R3, +/-20% hit rate, bias.
Part 2: the two counts side by side: by volume fifth, the share of forecasts, of misses beyond 20% and of the error; and what fixing the
worst 5% of forecasts (by units) would do to the hit rate.
Adds J["stackvar"] and J["twocounts"] to data/fk_err_analysis.json; stdout -> eda/stack_variants_output.txt."""
import sys, json, itertools, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
D = pd.read_parquet("data/fk_err_master.parquet"); ORIG = sorted(D.origin.unique()); REC = [o for o in ORIG if o >= "2025-06"]; D["rec"] = D.origin.isin(REC); D["oo"] = D.origin.map(lambda o: pd.Period(o, "M").ordinal)
MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; X = D[MEM].values.astype(float); y = D.y.values; ar = (D.R3 - D.y).abs().values; corr = (D.F / D.F0).values; rec = D.rec.values
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
L = np.log(X + 1); S = np.sort(X, axis=1); Ls = np.sort(L, axis=1)
V = {"arithmetic mean (submitted stack)": X.mean(1), "geometric mean": np.exp(L.mean(1)) - 1, "median of the five": np.median(X, 1), "trimmed mean (drop lowest and highest)": S[:, 1:4].mean(1),
     "upper four (drop the lowest)": S[:, 1:].mean(1), "upper three": S[:, 2:].mean(1), "lower four (drop the highest)": S[:, :4].mean(1)}
tri = list(itertools.combinations(range(5), 3)); rng_ = np.stack([L[:, t].max(1) - L[:, t].min(1) for t in tri], 1); bt = rng_.argmin(1)
V["densest three (smallest spread)"] = np.array([X[i, tri[b]].mean() for i, b in enumerate(bt)])
far = np.abs(L - np.median(L, 1, keepdims=True)).argmax(1); m4 = np.ones_like(X, bool); m4[np.arange(len(X)), far] = False; V["drop the member farthest from the median"] = (X * m4).sum(1) / 4
# densest cluster among the higher outputs: of the two triples (lower three, upper three) of sorted values, take the upper if it is at least as tight
up = (Ls[:, 4] - Ls[:, 2]) <= (Ls[:, 2] - Ls[:, 0]); V["tighter of upper / lower three"] = np.where(up, S[:, 2:].mean(1), S[:, :3].mean(1))
# weights by past score
age = pd.cut(D.age_m, [-1, 3, 12, 999], labels=False).values
def weighted(k, by_age=False, best=False):
    f = X.mean(1).copy()
    for o in ORIG:
        te = (D.origin == o).values; past = (D.oo + 4 <= pd.Period(o).ordinal).values
        if not past.any(): continue
        if by_age:
            for a in range(3):
                pa, ta = past & (age == a), te & (age == a)
                if pa.sum() < 100 or not ta.any(): continue
                w = (y[pa].sum() / np.abs(X[pa] - y[pa, None]).sum(0)) ** k; f[ta] = X[ta] @ (w / w.sum())
        else:
            e = np.abs(X[past] - y[past, None]).sum(0) / y[past].sum()
            if best: f[te] = X[te][:, e.argmin()]
            else: w = (1 / e) ** k; f[te] = X[te] @ (w / w.sum())
    return f
for k in [1, 2, 4, 8]: V[f"weights by past score, power {k}"] = weighted(k)
V["weights by past score within age band, power 2"] = weighted(2, by_age=True); V["best single member so far"] = weighted(1, best=True)
V["best member per item, picked afterwards (bound)"] = X[np.arange(len(X)), np.abs(X - y[:, None]).argmin(1)]
def sc(f, m): return (np.abs(f - y)[m]).sum() / ar[m].sum(), float((np.abs(f - y)[m] <= 0.2 * y[m]).mean()), f[m].sum() / y[m].sum() - 1
rng = np.random.default_rng(0); base = V["arithmetic mean (submitted stack)"]
def boot(f, B=1000):
    Z = pd.DataFrame({"s": D.style_color, "a": np.abs(base - y), "b": np.abs(f - y), "r": ar, "ha": np.abs(base - y) <= 0.2 * y, "hb": np.abs(f - y) <= 0.2 * y, "n": 1.0})[rec].groupby("s").sum().values; n = len(Z); ix = rng.integers(0, n, (B, n)); T = Z[ix].sum(1)
    d = T[:, 1] / T[:, 2] - T[:, 0] / T[:, 2]; hd = (T[:, 4] - T[:, 3]) / T[:, 5]; return [r(np.percentile(d, 2.5)), r(np.percentile(d, 97.5))], [r(np.percentile(hd, 2.5)), r(np.percentile(hd, 97.5))]
OUT = []
for lab, f in V.items():
    w, h, b = sc(f, rec); wa, ha, _ = sc(f, np.ones(len(D), bool)); wc, hc, bc = sc(f * corr, rec); ci, hci = boot(f) if lab != "arithmetic mean (submitted stack)" else ([0, 0], [0, 0])
    OUT.append(dict(variant=lab, wape=r(w), hit=r(h), bias=r(b, 3), wape_all=r(wa), hit_all=r(ha), wape_corr=r(wc), hit_corr=r(hc), bias_corr=r(bc, 3), ci=ci, hit_ci=hci))
T = pd.DataFrame(OUT); print(T.drop(columns=["ci", "hit_ci"]).to_string(index=False)); print(); [print(f"  {d['variant'][:46]:46s} WAPE diff vs mean {d['ci']}  hit diff {d['hit_ci']}") for d in OUT[1:]]
# what the weights were
wts = []
for o in REC:
    past = (D.oo + 4 <= pd.Period(o).ordinal).values; e = np.abs(X[past] - y[past, None]).sum(0) / y[past].sum(); w = (1 / e) ** 2; wts.append(dict(origin=o, **{m: r(v, 3) for m, v in zip(MEM, w / w.sum())}))
print("\nweights (power 2) by origin:"); print(pd.DataFrame(wts).to_string(index=False))
SV = dict(rows=OUT, weights=wts, spread_hit=[dict(q=int(q) + 1, hit=r(g.h.mean(), 3), spread=r(g.s.median(), 3)) for q, g in pd.DataFrame({"h": np.abs(base * corr - y) <= 0.2 * y, "s": Ls[:, 4] - Ls[:, 0]})[rec].assign(q=lambda z: pd.qcut(z.s.rank(method="first"), 5, labels=False)).groupby("q")])
print("hit rate by fifth of the five models' log range:", SV["spread_hit"])

# ---- part 2: the two counts
R = D[rec].copy(); R["ae"] = (R.F - R.y).abs(); R["hit"] = R.ae <= 0.2 * R.y; R["vq"] = R.groupby("origin")["level"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
n = len(R); top = R.sort_values("ae", ascending=False).head(int(n * 0.05)); rest = R.drop(top.index)
TC = dict(top5=dict(n=int(len(top)), err=r(top.ae.sum() / R.ae.sum(), 3), units=r(top.y.sum() / R.y.sum(), 3), already_hit=r(top.hit.mean(), 3), med_level=r(top.level.median(), 0), med_ape=r((top.ae / top.y).median(), 3), med_err_units=r(top.ae.median(), 0), under=r((top.F < top.y).mean(), 3)),
          all=dict(med_level=r(R.level.median(), 0), med_err_units=r(R.ae.median(), 0), med_ape=r((R.ae / R.y).median(), 3)),
          hit_if_top5_fixed=r((R.hit.sum() + (~top.hit).sum()) / n, 3), hit_now=r(R.hit.mean(), 3), wape_if_top5_fixed=r(rest.ae.sum() / (R.R3 - R.y).abs().sum(), 3),
          misses=dict(n=int((~R.hit).sum()), share=r((~R.hit).mean(), 3), err=r(R.ae[~R.hit].sum() / R.ae.sum(), 3), med_err_units=r(R.ae[~R.hit].median(), 0)),
          by_fifth=[dict(fifth=int(q) + 1, med_level=r(g.level.median(), 0), forecasts=r(len(g) / n, 3), misses=r((~g.hit).sum() / (~R.hit).sum(), 3), err=r(g.ae.sum() / R.ae.sum(), 3), hit=r(g.hit.mean(), 3), units=r(g.y.sum() / R.y.sum(), 3)) for q, g in R.groupby("vq")])
sm = R[R.vq <= 2]; TC["small3"] = dict(forecasts=r(len(sm) / n, 3), misses=r((~sm.hit).sum() / (~R.hit).sum(), 3), err=r(sm.ae.sum() / R.ae.sum(), 3), units=r(sm.y.sum() / R.y.sum(), 3))
big = R[R.vq == 4]; TC["hit_if_top_fifth_perfect"] = r((R.hit.sum() + (~big.hit).sum()) / n, 3); TC["wape_if_top_fifth_perfect"] = r(R.ae[R.vq < 4].sum() / (R.R3 - R.y).abs().sum(), 3)
print("\npart 2:", json.dumps(TC, indent=0)[:1800])
J = json.load(open("data/fk_err_analysis.json")); J["stackvar"] = SV; J["twocounts"] = TC; json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
