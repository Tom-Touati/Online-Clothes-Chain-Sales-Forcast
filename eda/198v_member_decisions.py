"""D309 (fork; user comment on the error study, at 'Known only afterwards': "anything not related to sales but is a result of new
partnerships or other business decisions can be added as a known future"). The tree member (Main's 'ec_ns') refitted with every
business decision of the forecast window as a known-future input (eda/198d axes(), hindsight on, computed for every training row):
  DISC  the discount plan: the item's web discount in the window against before, and the same against the site's
  DEC   every decision: own stores added and reached, partners added, new colours of the style and same-type launches, the discount plan
  DEC + AV   plus availability in the window (size run, stock gaps) and end of life: these are partly a RESULT of sales (an item sells
        out because it sold), so this row leaks the outcome and is shown only as a bound
  DEC + SIG  every decision on top of the origin-known signals of D307
Realised values in training (closed windows) and at the test origin (an exact plan); read from sales, so an upper bound on a real plan.
Scored alone and inside the five-member average. Adds J["refit_decisions"]; stdout -> eda/member_decisions_output.txt."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
argv0 = sys.argv; sys.argv = ["eda/180q_models.py"]; ns = {}; exec(compile(open("eda/180q_models.py", encoding="utf-8").read(), "eda/180q_models.py", "exec"), ns); sys.argv = argv0
ns["P"]["n_jobs"] = 3   # user: "you can use the laptop but dont let it lag the computer" (3 threads, low priority)
T = ns["T"].reset_index(drop=True); sets = ns["sets"]; fit = ns["fit"]; ORIG = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
src = open("eda/198d_error_master.py", encoding="utf-8").read(); src = src[src.index("import sys, glob"):src.index("\ndef build(")]; ax = {}; exec(compile(src, "eda/198d_error_master.py", "exec"), ax)
if "look" not in T: T["look"] = ""
H = pd.concat([ax["axes"](g, o, hindsight=(o <= "2026-02")) for o, g in T.groupby("origin")]).reindex(T.index).add_prefix("ax_"); H["ax_stores_end"] = H.ax_n_stores + H.ax_h_stores_chg; T = pd.concat([T, H], axis=1)
SIG = ["ax_" + c for c in ["oos_28", "newest_colour", "size_run", "size_hhi", "sh_partner", "n_partners", "share_of_style", "n_siblings", "disc_web", "disc_vs_site", "top_day_share"]]
DISC = ["ax_h_disc_chg", "ax_h_disc_chg_rel"]; DEC = ["ax_h_stores_chg", "ax_stores_end", "ax_h_partners_chg", "ax_h_new_style", "ax_h_new_type"] + DISC; AV = ["ax_h_size_run", "ax_h_oos", "ax_h_dead"]
B0, F0_ = sets("ec_ns"); PRED = {}
def run(lab, add):
    t0 = time.time(); out = []
    for o in ORIG:
        oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()]; te = T[T.origin == o]; assert (tr.o_ord + 4 <= oo).all() and tr[add].notna().any().all()
        z = (fit(tr, te, B0 + add) + fit(tr, te, F0_ + add)) / 2; out.append(pd.DataFrame({"origin": o, "style_color": te.style_color.values, "q50": np.clip((4 * te.level.values + 1) * np.exp(z) - 1, 0, None)}))
    PRED[lab] = pd.concat(out).set_index(["origin", "style_color"]).q50; print(lab, "|", len(add), "inputs |", round(time.time() - t0), "s", flush=True)
STO = ["ax_h_stores_chg", "ax_stores_end"]; run("+ stores added and reached (known future)", STO); run("+ the discount plan (known future)", DISC); run("+ every business decision: stores, partners, launches, discount (known future)", DEC)
run("+ every business decision and the origin-known signals", DEC + SIG); run("+ decisions and availability in the window (partly a result of sales: a bound)", DEC + AV)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); K = pd.MultiIndex.from_arrays([D.origin, D.style_color]); S = pd.read_parquet("data/fk_member_refit_preds.parquet").set_index(["origin", "style_color"]).reindex(K); m0 = S["ec_ns (refit, check)"].values
rec = (D.origin >= "2025-06").values; allm = np.ones(len(D), bool); y = D.y.values; ar = (D.R3 - D.y).abs().values; MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; X = D[MEM].values.astype(float); corr = (D.F / D.F0).values; rng = np.random.default_rng(0)
wape = lambda f, m=rec: float(np.abs(f - y)[m].sum() / ar[m].sum()); hit = lambda f, m=rec: float((np.abs(f - y)[m] <= 0.2 * y[m]).mean()); bias = lambda f, m=rec: float(f[m].sum() / y[m].sum() - 1)
def boot(fa, fb_, B=1000):
    Z = pd.DataFrame({"s": D.style_color, "a": np.abs(fa - y), "b": np.abs(fb_ - y), "r": ar, "ha": np.abs(fa - y) <= 0.2 * y, "hb": np.abs(fb_ - y) <= 0.2 * y, "n": 1.0})[rec].groupby("s").sum().values; n = len(Z); T_ = Z[rng.integers(0, n, (B, n))].sum(1)
    d = T_[:, 1] / T_[:, 2] - T_[:, 0] / T_[:, 2]; hd = (T_[:, 4] - T_[:, 3]) / T_[:, 5]; return [r(np.percentile(d, 2.5)), r(np.percentile(d, 97.5))], [r(np.percentile(hd, 2.5), 3), r(np.percentile(hd, 97.5), 3)]
s0 = (X.sum(1) - D.qdyn.values + m0) / 5; rows = [dict(variant="tree member as submitted (refit)", alone=r(wape(m0)), alone_hit=r(hit(m0), 3), stack=r(wape(s0)), stack_hit=r(hit(s0), 3), stack_bias=r(bias(s0), 3), corrected=r(wape(s0 * corr)), corrected_hit=r(hit(s0 * corr), 3), stack_all=r(wape(s0, allm)))]
for lab in PRED:
    m = PRED[lab].reindex(K).values; assert np.isfinite(m).all(); st = (X.sum(1) - D.qdyn.values + m) / 5; ci_m, _ = boot(m0, m); ci_s, hci = boot(s0, st); po = [wape(st, (D.origin == o).values) - wape(s0, (D.origin == o).values) for o in sorted(D.origin.unique())]
    rows.append(dict(variant=lab, alone=r(wape(m)), alone_hit=r(hit(m), 3), alone_ci=ci_m, stack=r(wape(st)), stack_hit=r(hit(st), 3), stack_bias=r(bias(st), 3), stack_ci=ci_s, stack_hit_ci=hci, corrected=r(wape(st * corr)), corrected_hit=r(hit(st * corr), 3), stack_all=r(wape(st, allm)), wins13=int(sum(v < 0 for v in po))))
print(pd.DataFrame(rows).to_string(index=False))
# launches under 3 months by what happened to their distribution (the table of the Distribution section; user comment there: "those can be future knowns")
sv = D[["origin", "style_color"]].copy(); young = rec & (D.age_m.values < 3); hs = D.h_stores_chg.values; GR = [("stores added: 5 or more", hs >= 5), ("stores added: 2 to 4", (hs >= 2) & (hs <= 4)), ("stores about flat or fewer", hs < 2)]; launch = []
for lab in ["+ stores added and reached (known future)", "+ every business decision: stores, partners, launches, discount (known future)"]:
    m = PRED[lab].reindex(K).values; st = (X.sum(1) - D.qdyn.values + m) / 5 * corr
    for g, gm in GR:
        mm = young & gm; launch.append(dict(variant=lab, g=g, n=int(mm.sum()), bias_base=r(bias(s0 * corr, mm), 3), bias=r(bias(st, mm), 3), wape_base=r(wape(s0 * corr, mm), 3), wape=r(wape(st, mm), 3), hit_base=r(hit(s0 * corr, mm), 3), hit=r(hit(st, mm), 3)))
    launch.append(dict(variant=lab, g="all launches under 3 months", n=int(young.sum()), bias_base=r(bias(s0 * corr, young), 3), bias=r(bias(st, young), 3), wape_base=r(wape(s0 * corr, young), 3), wape=r(wape(st, young), 3), hit_base=r(hit(s0 * corr, young), 3), hit=r(hit(st, young), 3)))
print(pd.DataFrame(launch).to_string(index=False))
for lab in PRED: sv[lab] = PRED[lab].reindex(K).values
sv.to_parquet("data/fk_member_decisions_preds.parquet"); J = json.load(open("data/fk_err_analysis.json")); J["refit_decisions"] = dict(rows=rows, launch=launch); json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
