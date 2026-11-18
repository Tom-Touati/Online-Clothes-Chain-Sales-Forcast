"""D326 (fork; overnight tuning, user in Fork 2's session: "/goal hyperparameter optimization on all models through the night"): the TabPFN v2
dyn member (pf2d32: 32 estimators, softmax temperature 0.9, no context cap, absolute target off) tuned with Optuna TPE on the GPU
(eda/hpo_gpu_driver.py tabpfn, tune origins 2025-02..2025-09; data/hpo_tabpfn_trials.csv), then rerun with the best settings on all 13 origins
and without last year's window (eda/hpo_gpu_finals.py tabpfn -> data/fk_hpo_pf_final{,_ly}.parquet). Scored as D325 against the stored member:
  periods   tune (2025-02..09) | held out (2025-10..2026-02) | recent (2025-06..2026-02) | all 13 | early (2025-02..05)
  scores    the member alone and the five-member system with the submitted correction multiplier (WAPE / R3), within +/-20%, GMRAE; bootstrap of
            the system difference (items resampled)
  gate      without last year's window: the tuned member against the stored one without it (D324, data/fk_ly_pf_e32.parquet), alone and in the system
  check     trial 0 (the current settings, rerun on the GPU) against the stored member's criterion on the tune origins (reproduction)
If the best trial is the current settings, the finals were not run and the stored member stands. Adds J["hpo_tabpfn"]; stdout -> eda/hpo_tabpfn_score_output.txt."""
import sys, os, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); K = pd.MultiIndex.from_arrays([D.origin, D.style_color]); y = D.y.values; ar = np.abs(D.R3.values - y); corr = (D.F / D.F0).values
T = pd.read_csv("data/hpo_tabpfn_trials.csv"); best = json.load(open("data/hpo_tabpfn_best.json")); print(T.drop(columns=[c for c in ["error"] if c in T.columns]).round(4).to_string(index=False)); print("best:", best)
P = {"tune": D.origin.between("2025-02", "2025-09").values, "held out": D.origin.between("2025-10", "2026-02").values, "recent": (D.origin >= "2025-06").values, "all 13": np.ones(len(D), bool), "early": D.origin.between("2025-02", "2025-05").values}
w = lambda f, m: float(np.abs(f - y)[m].sum() / ar[m].sum()); hit = lambda f, m: float((np.abs(f - y)[m] <= 0.2 * y[m]).mean()); gm = lambda f, m: float(np.exp(np.mean(np.log((np.abs(f - y)[m] + 1) / (ar[m] + 1)))))
pre = lambda f: (D.rules.values + D.qdyn.values + D.nn.values + D.pf2c.values + f) / 5; sysf = lambda f: pre(f) * corr
tm = P["tune"]; crit_stored = (w(D.pf2d32.values, tm) + w(pre(D.pf2d32.values), tm)) / 2; t0 = T[T.trial == 0].crit.values
print("reproduction: stored member's criterion on the tune origins", r(crit_stored), "| trial 0 (current settings rerun on the GPU)", r(float(t0[0])) if len(t0) else None)
out = dict(best=best.get("best"), base=best.get("base"), n_trials=int(len(T)), crit_range=[r(T.crit.min()), r(T.crit.max())], crit_stored=r(crit_stored))
if not os.path.exists("data/fk_hpo_pf_final.parquet"):
    print("no final runs: the best trial is the current settings; the stored member stands"); out["verdict"] = "best = current settings"
else:
    tq = lambda f: (lambda q: (q.groupby(["origin", "style_color"])[["q25", "q50", "q75"]].sum() if "h" in q.columns and q.h.nunique() > 1 else q.set_index(["origin", "style_color"])[["q25", "q50", "q75"]]).reindex(K))(pd.read_parquet(f))
    Qt = tq("data/fk_hpo_pf_final.parquet"); pf_t = Qt.q50.values; pf_c = D.pf2d32.values; assert np.isfinite(pf_t).all()
    rows = []
    for lab, fc, ft in [("member alone", pf_c, pf_t), ("system (submitted corrections)", sysf(pf_c), sysf(pf_t))]:
        for pn, m in P.items(): rows.append(dict(view=lab, period=pn, current=r(w(fc, m)), tuned=r(w(ft, m)), diff=r(w(ft, m) - w(fc, m)), hit_current=r(hit(fc, m), 3), hit_tuned=r(hit(ft, m), 3), gmrae_current=r(gm(fc, m), 3), gmrae_tuned=r(gm(ft, m), 3)))
    R = pd.DataFrame(rows); print(R.to_string(index=False))
    rng = np.random.default_rng(0)
    def boot(fa, fb, m, B_=1000):
        Z = pd.DataFrame({"s": D.style_color, "a": np.abs(fa - y), "b": np.abs(fb - y), "r": ar})[m].groupby("s").sum().values; n = len(Z); T_ = Z[rng.integers(0, n, (B_, n))].sum(1); d = T_[:, 1] / T_[:, 2] - T_[:, 0] / T_[:, 2]
        return [r(np.percentile(d, 2.5)), r(np.percentile(d, 97.5))]
    ci = {pn: boot(sysf(pf_c), sysf(pf_t), P[pn]) for pn in ["held out", "recent", "all 13"]}; print("system difference, 95% interval:", ci)
    ly = []
    if os.path.exists("data/fk_hpo_pf_final_ly.parquet"):
        pf_tly = tq("data/fk_hpo_pf_final_ly.parquet").q50.values; pf_cly = tq("data/fk_ly_pf_e32.parquet").q50.values
        for pn in ["recent", "all 13"]:
            m = P[pn]; ly.append(dict(period=pn, alone_current_ly=r(w(pf_cly, m)), alone_tuned_ly=r(w(pf_tly, m)), sys_current_ly=r(w(sysf(pf_cly), m)), sys_tuned_ly=r(w(sysf(pf_tly), m)),
                                      deg_current=r(w(pf_cly, m) - w(pf_c, m)), deg_tuned=r(w(pf_tly, m) - w(pf_t, m))))
        print("without last year:"); print(pd.DataFrame(ly).to_string(index=False))
    rec = P["recent"]; cov = r(float(((y >= Qt.q25.values) & (y <= Qt.q75.values))[rec].mean()), 3); print("tuned member P25-P75 inside (recent):", cov)
    out.update(rows=rows, ci=ci, ly=ly, quantile_cover_recent=cov)
J = json.load(open("data/fk_err_analysis.json")); J["hpo_tabpfn"] = out; json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0, default=float); print("saved")
