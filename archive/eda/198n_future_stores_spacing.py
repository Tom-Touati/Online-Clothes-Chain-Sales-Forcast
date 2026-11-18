"""D283, D284 (fork; two user comments on the error study).
A  "but we can add it as a known future - how many stores did it reach. right?"  The number of own stores an item is added to during
   the forecast window, used as a known-future input. In the backtest it is read afterwards from the sales (a store counts once the
   item sells there), so it is an upper bound on what a plan made before the window gives. Time-honest residual correction of the
   submitted system (earlier origins only, whose windows have closed; origin total kept), same reference as D282:
     + stores added in the window and stores reached at its end | + partners added | + the whole distribution block (D276)
     + the store counts with noise of +/-2 stores (a plan that is not carried out exactly)
   and where the gain sits: launches under 3 months, items that gained 2+ stores.
B  "lets do it in origins which are 2 month apart instead of 1 month, right?"  The submitted system and every adopted step re-scored
   on origins 2 months apart (2 sets) and 4 months apart (4 sets, no shared months), with the correlation of an item's error between
   origins at each spacing and the number of independent tests each spacing leaves.
Adds J["future_stores"], J["spacing"] to data/fk_err_analysis.json; stdout -> eda/future_stores_spacing_output.txt."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
D = pd.read_parquet("data/fk_err_master.parquet"); ORIG = sorted(D.origin.unique()); REC = [o for o in ORIG if o >= "2025-06"]; D["rec"] = D.origin.isin(REC); D["oo"] = D.origin.map(lambda o: pd.Period(o, "M").ordinal)
J = json.load(open("data/fk_err_analysis.json")); y = D.y.values; ar = (D.R3 - D.y).abs().values; rec = D.rec.values; D["ar"] = ar
D["res"] = np.log((D.y + 1) / (D.F + 1)); D["res_c"] = D.res - D.groupby("origin").res.transform("mean"); D["llv"] = np.log1p(D.level); D["lf"] = np.log((D.F + 1) / (D.R3 + 1)); D["macro_c"] = D.macro.astype("category").cat.codes
D["stores_end"] = D.n_stores + D.h_stores_chg; rng = np.random.default_rng(0); D["h_st_noisy"] = np.round(D.h_stores_chg + rng.normal(0, 2, len(D))); D["end_noisy"] = np.maximum(D.n_stores + D.h_st_noisy, 0)
BASEF = ["llv", "age_m", "lf", "spread", "bf_window", "macro_c", "disc_web", "size_run", "sh_tiktok", "sh_retail", "sh_partner", "n_stores", "trend91"]
PR = dict(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=6)
def th_corr(cols, shrink=0.5):
    f = D.F.values.copy()
    for o in REC:
        tr = D[D.oo + 4 <= pd.Period(o).ordinal]; te = D[D.origin == o]
        if len(tr) < 500: continue
        v = (te.F + 1) * np.exp(shrink * lgb.LGBMRegressor(**PR).fit(tr[cols], tr.res_c, sample_weight=np.log1p(tr.R3)).predict(te[cols])) - 1; f[(D.origin == o).values] = v * te.F.sum() / v.sum()
    return f
hit = lambda f, m=rec: float((np.abs(f - y)[m] <= 0.2 * y[m]).mean()); wape = lambda f, m=rec: float(np.abs(f - y)[m].sum() / ar[m].sum()); bias = lambda f, m=rec: float(f[m].sum() / y[m].sum() - 1)
def boot(fa, fb_, B=1000):
    Z = pd.DataFrame({"s": D.style_color, "a": np.abs(fa - y), "b": np.abs(fb_ - y), "r": ar, "ha": np.abs(fa - y) <= 0.2 * y, "hb": np.abs(fb_ - y) <= 0.2 * y, "n": 1.0})[rec].groupby("s").sum().values; n = len(Z); T = Z[rng.integers(0, n, (B, n))].sum(1)
    d = T[:, 1] / T[:, 2] - T[:, 0] / T[:, 2]; hd = (T[:, 4] - T[:, 3]) / T[:, 5]; return [r(np.percentile(d, 2.5)), r(np.percentile(d, 97.5))], [r(np.percentile(hd, 2.5), 3), r(np.percentile(hd, 97.5), 3)]
fb = th_corr(BASEF); F0 = D.F.values; VAR = [("+ stores added in the window, stores reached at its end", ["h_stores_chg", "stores_end"]), ("+ partners added as well", ["h_stores_chg", "stores_end", "h_partners_chg"]),
       ("+ the whole distribution block (stores, partners, channel shifts)", ["h_stores_chg", "stores_end", "h_partners_chg", "h_tiktok_chg", "h_retail_chg", "h_partner_chg"]), ("+ the store counts, off by about 2 stores", ["h_st_noisy", "end_noisy"])]
rows = [dict(variant="submitted system", wape=r(wape(F0)), hit=r(hit(F0), 3)), dict(variant="correction on origin-known features (reference)", wape=r(wape(fb)), hit=r(hit(fb), 3))]; FV = {}
for lab, cols in VAR:
    f = th_corr(BASEF + cols); FV[lab] = f; ci, hci = boot(fb, f); rows.append(dict(variant=lab, wape=r(wape(f)), hit=r(hit(f), 3), ci=ci, hit_ci=hci, d=r(wape(f) - wape(fb))))
print("A  stores reached as a known future (time-honest, origin total kept):"); print(pd.DataFrame(rows).to_string(index=False))
f1 = FV[VAR[0][0]]; coh = []
for lab, m in [("launches under 3 months", rec & (D.age_m.values < 3)), ("items that gained 2+ stores", rec & (D.h_stores_chg.values >= 2)), ("items that gained 5+ stores", rec & (D.h_stores_chg.values >= 5)), ("stores flat or fewer", rec & (D.h_stores_chg.values <= 0)), ("all", rec)]:
    coh.append(dict(g=lab, n=int(m.sum()), units=r(y[m].sum() / y[rec].sum(), 3), wape_ref=r(wape(fb, m)), wape_k=r(wape(f1, m)), bias_ref=r(bias(fb, m), 3), bias_k=r(bias(f1, m), 3), hit_ref=r(hit(fb, m), 3), hit_k=r(hit(f1, m), 3)))
print(pd.DataFrame(coh).to_string(index=False)); sc = J["store_check"]
J["future_stores"] = dict(rows=rows, cohorts=coh, rho_hind=J["sig_hind"][0]["rho"] if J["sig_hind"][0]["key"] == "h_stores_chg" else next(d["rho"] for d in J["sig_hind"] if d["key"] == "h_stores_chg"),
                          web_bias_add2=sc["add2"]["web_bias"], bias_add2=sc["add2"]["bias"], rho_web_growth=sc["rho_web_growth"], share_gain2=r(float((D[D.rec].h_stores_chg >= 2).mean()), 3))

# ---------- B origin spacing
SYS = [("R3 + rules", "rules"), ("first three-way", "s_rq"), ("+ weather, events (D161)", "s_wx3bf"), ("+ co-purchase neighbours (D203)", "s_dyn3bf"), ("+ TabPFN v2 (D204)", "s_p2"), ("arithmetic stack (D207)", "s_p2c32a"),
       ("new tree member (D253)", "s_ecns"), ("new NN inputs (D272)", "F0"), ("corrections (D273) = submitted", "F")]
w = lambda Z, c: float((Z[c] - Z.y).abs().sum() / Z.ar.sum()); h = lambda Z, c: float(((Z[c] - Z.y).abs() <= 0.2 * Z.y).mean())
SETS = [("all 13 origins, 1 month apart", ORIG, 1), ("2 months apart, set A", ORIG[0::2], 2), ("2 months apart, set B", ORIG[1::2], 2)] + [(f"4 months apart, set {k + 1}", ORIG[k::4], 4) for k in range(4)]
sets = []
for lab, oo, sp in SETS:
    Z = D[D.origin.isin(oo)]; sets.append(dict(set=lab, spacing=sp, origins=list(oo), n=len(oo), wape=r(w(Z, "F")), hit=r(h(Z, "F"), 3), hit_r3=r(h(Z, "R3"), 3), bias=r(float(Z.F.sum() / Z.y.sum() - 1), 3), rules=r(w(Z, "rules")), n_rows=int(len(Z))))
print("\nB  the submitted system by origin spacing:"); print(pd.DataFrame(sets).drop(columns=["origins"]).to_string(index=False))
steps = []
for (la, a), (lb_, b) in zip(SYS[:-1], SYS[1:]):
    d = {lab: w(D[D.origin.isin(oo)], b) - w(D[D.origin.isin(oo)], a) for lab, oo, sp in SETS}
    steps.append(dict(step=lb_, d_all=r(d[SETS[0][0]]), d2=[r(d[SETS[1][0]]), r(d[SETS[2][0]])], d4=[r(d[s[0]]) for s in SETS[3:]], wins2=int(sum(d[s[0]] < 0 for s in SETS[1:3])), wins4=int(sum(d[s[0]] < 0 for s in SETS[3:]))))
print(pd.DataFrame(steps).to_string(index=False))
# correlation of an item's error between origins k months apart, and the independent tests each spacing leaves
P = D.pivot_table(index="style_color", columns="origin", values="res_c"); lag = {}
for k in range(1, 9):
    a = np.concatenate([P[ORIG[i]].values for i in range(len(ORIG) - k)]); b = np.concatenate([P[ORIG[i + k]].values for i in range(len(ORIG) - k)]); ok = np.isfinite(a) & np.isfinite(b); lag[k] = float(np.corrcoef(a[ok], b[ok])[0, 1])
LC = {d["lag"]: d["corr"] for d in J["confounds"]["lag_corr"]}; neff = lambda n, sp: n / (1 + 2 * sum(LC[sp * j] for j in range(1, n) if sp * j <= 4))   # as J["neff"] (D276): lags up to 4 months, the stored correlations
NE = [dict(spacing=1, n=13, shared=3, corr=r(LC[1], 3), neff=r(neff(13, 1), 1)), dict(spacing=2, n=7, shared=2, corr=r(LC[2], 3), neff=r(neff(7, 2), 1)), dict(spacing=3, n=5, shared=1, corr=r(LC[3], 3), neff=r(neff(5, 3), 1)), dict(spacing=4, n=4, shared=0, corr=r(LC[4], 3), neff=r(neff(4, 4), 1))]
print("item-error correlation by lag:", {k: round(v, 3) for k, v in lag.items()}); print(pd.DataFrame(NE).to_string(index=False))
J["spacing"] = dict(sets=sets, steps=steps, neff=NE, lag={str(k): r(v, 3) for k, v in lag.items()}, range4=[r(min(s["wape"] for s in sets[3:])), r(max(s["wape"] for s in sets[3:]))], hit4=[r(min(s["hit"] for s in sets[3:]), 3), r(max(s["hit"] for s in sets[3:]), 3)])
json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
