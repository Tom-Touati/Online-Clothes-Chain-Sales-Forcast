"""D308 (fork; user comments on the error study, at 'Known only afterwards': on "own stores added in the window", "we can add that info as
future known i believe"; on "new colours of the style launched in the window", "also can be a known future"). The tree member (Main's 'ec_ns', eda/180q_models.py) refitted with the roll-out as a known-future input:
  KF1  own stores added during the forecast window and the stores reached at its end
  KF2  KF1 + partners added during the window
  KFP  new colours of the same style and launches of the same product type during the window (the launch plan); KF2 + KFP together
In training the values are the realised ones (windows closed: o' + 4 <= o); at the test origin they are also the realised ones, i.e.
an exact plan. They are read from the sales (a store counts once the item sells there), so this is an upper bound on a real plan; a
version with the test-origin count off by about two stores is scored too. Scored alone and inside the five-member average.
Adds J["refit_future"] to data/fk_err_analysis.json; stdout -> eda/member_future_stores_output.txt."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
argv0 = sys.argv; sys.argv = ["eda/180q_models.py"]; ns = {}; exec(compile(open("eda/180q_models.py", encoding="utf-8").read(), "eda/180q_models.py", "exec"), ns); sys.argv = argv0
T = ns["T"].reset_index(drop=True); sets = ns["sets"]; fit = ns["fit"]; ORIG = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
src = open("eda/198d_error_master.py", encoding="utf-8").read(); src = src[src.index("import sys, glob"):src.index("\ndef axes(")]; ax = {}; exec(compile(src, "eda/198d_error_master.py", "exec"), ax); win = ax["win"]
fm = ax["first"].dt.to_period("M").astype(str); T["kf_new_style"] = np.nan; T["kf_new_type"] = np.nan
nz = lambda v: np.nan_to_num(v, neginf=0); T["kf_st_add"] = np.nan; T["kf_st_end"] = np.nan; T["kf_par_add"] = np.nan; T["kf_st_now"] = np.nan
with np.errstate(all="ignore"):
    for o in sorted(T.origin.unique()):
        m = (T.origin == o).values; s = T.style_color.values[m]; now = nz(win("n_stores", s, o, [0], "max")); b3 = nz(win("n_stores", s, o, [-2, -1, 0], "max")); t4 = nz(win("n_stores", s, o, [1, 2, 3, 4], "max"))
        T.loc[m, "kf_st_add"] = t4 - b3; T.loc[m, "kf_st_end"] = now + (t4 - b3); T.loc[m, "kf_st_now"] = now; T.loc[m, "kf_par_add"] = nz(win("n_partners", s, o, [1, 2, 3, 4], "max")) - nz(win("n_partners", s, o, [-2, -1, 0], "max"))
        new = fm[(fm > o) & (fm <= ax["P"](o, 4))].index; T.loc[m, "kf_new_style"] = ax["sty"].reindex(new).value_counts().reindex(ax["sty"].reindex(s).values).fillna(0).values   # new colours of the style launched in the window
        T.loc[m, "kf_new_type"] = ax["it"].product_type.reindex(new).value_counts().reindex(ax["it"].product_type.reindex(s).values).fillna(0).values                              # launches of the same product type in the window
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); K = pd.MultiIndex.from_arrays([D.origin, D.style_color]); chk = T.set_index(["origin", "style_color"]).kf_st_add.reindex(K).values
print("stores added: agreement with the error master's own-stores-added column:", float(np.mean(chk == D.h_stores_chg.values)), flush=True)
rng = np.random.default_rng(0); T["kf_st_add_noisy"] = np.round(T.kf_st_add + rng.normal(0, 2, len(T))); T["kf_st_end_noisy"] = np.maximum(T.kf_st_now + T.kf_st_add_noisy, 0)
B0, F0_ = sets("ec_ns"); KF1 = ["kf_st_add", "kf_st_end"]; KF2 = KF1 + ["kf_par_add"]; PRED = {}
def run(lab, add, test_swap=None):
    t0 = time.time(); out = []
    for o in ORIG:
        oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()]; te = T[T.origin == o].copy(); assert (tr.o_ord + 4 <= oo).all()
        if test_swap:
            for a, b in test_swap.items(): te[a] = te[b].values   # the plan given at the test origin is off; the model was trained on realised values
        z = (fit(tr, te, B0 + add) + fit(tr, te, F0_ + add)) / 2; out.append(pd.DataFrame({"origin": o, "style_color": te.style_color.values, "q50": np.clip((4 * te.level.values + 1) * np.exp(z) - 1, 0, None)}))
    PRED[lab] = pd.concat(out).set_index(["origin", "style_color"]).q50; print(lab, "|", round(time.time() - t0), "s", flush=True)
KFP = ["kf_new_style", "kf_new_type"]; run("+ stores added in the window, stores reached at its end (known future)", KF1); run("+ stores and partners added (known future)", KF2); run("+ new colours of the style and same-type launches in the window (known future)", KFP); run("+ stores, partners and launches (all known futures)", KF2 + KFP); run("+ the store counts, off by about 2 stores at the forecast date", KF1, {"kf_st_add": "kf_st_add_noisy", "kf_st_end": "kf_st_end_noisy"})
S = pd.read_parquet("data/fk_member_refit_preds.parquet").set_index(["origin", "style_color"]).reindex(K); m0 = S["ec_ns (refit, check)"].values
rec = (D.origin >= "2025-06").values; allm = np.ones(len(D), bool); y = D.y.values; ar = (D.R3 - D.y).abs().values; MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; X = D[MEM].values.astype(float); corr = (D.F / D.F0).values
wape = lambda f, m=rec: float(np.abs(f - y)[m].sum() / ar[m].sum()); hit = lambda f, m=rec: float((np.abs(f - y)[m] <= 0.2 * y[m]).mean()); bias = lambda f, m=rec: float(f[m].sum() / y[m].sum() - 1)
def boot(fa, fb_, B=1000):
    Z = pd.DataFrame({"s": D.style_color, "a": np.abs(fa - y), "b": np.abs(fb_ - y), "r": ar, "ha": np.abs(fa - y) <= 0.2 * y, "hb": np.abs(fb_ - y) <= 0.2 * y, "n": 1.0})[rec].groupby("s").sum().values; n = len(Z); T_ = Z[rng.integers(0, n, (B, n))].sum(1)
    d = T_[:, 1] / T_[:, 2] - T_[:, 0] / T_[:, 2]; hd = (T_[:, 4] - T_[:, 3]) / T_[:, 5]; return [r(np.percentile(d, 2.5)), r(np.percentile(d, 97.5))], [r(np.percentile(hd, 2.5), 3), r(np.percentile(hd, 97.5), 3)]
s0 = (X.sum(1) - D.qdyn.values + m0) / 5; rows = [dict(variant="tree member as submitted (refit)", alone=r(wape(m0)), alone_hit=r(hit(m0), 3), stack=r(wape(s0)), stack_hit=r(hit(s0), 3), stack_bias=r(bias(s0), 3), corrected=r(wape(s0 * corr)), corrected_hit=r(hit(s0 * corr), 3), stack_all=r(wape(s0, allm)))]
g5 = rec & (D.h_stores_chg.values >= 5); g2 = rec & (D.h_stores_chg.values >= 2); g0 = rec & (D.h_stores_chg.values <= 0)
for lab in PRED:
    m = PRED[lab].reindex(K).values; assert np.isfinite(m).all(); st = (X.sum(1) - D.qdyn.values + m) / 5; ci_m, _ = boot(m0, m); ci_s, hci = boot(s0, st); po = [wape(st, (D.origin == o).values) - wape(s0, (D.origin == o).values) for o in sorted(D.origin.unique())]
    rows.append(dict(variant=lab, alone=r(wape(m)), alone_hit=r(hit(m), 3), alone_ci=ci_m, stack=r(wape(st)), stack_hit=r(hit(st), 3), stack_bias=r(bias(st), 3), stack_ci=ci_s, stack_hit_ci=hci, corrected=r(wape(st * corr)), corrected_hit=r(hit(st * corr), 3), stack_all=r(wape(st, allm)),
                     wins13=int(sum(v < 0 for v in po)), bias5=r(bias(st * corr, g5), 3), bias5_base=r(bias(s0 * corr, g5), 3), bias2=r(bias(st * corr, g2), 3), bias2_base=r(bias(s0 * corr, g2), 3), bias0=r(bias(st * corr, g0), 3), bias0_base=r(bias(s0 * corr, g0), 3)))
print(pd.DataFrame(rows).to_string(index=False)); J = json.load(open("data/fk_err_analysis.json")); J["refit_future"] = dict(rows=rows, submitted=r(wape(D.F.values)), submitted_hit=r(hit(D.F.values), 3)); json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
