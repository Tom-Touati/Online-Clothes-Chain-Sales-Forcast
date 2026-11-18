"""D325 (fork; overnight tuning, user in Fork 2's session: "/goal hyperparameter optimization on all models through the night"): the NN S13ecx6
tuned with Optuna TPE on the GPU (eda/hpo_gpu_driver.py nn, 45 trials, seeds=1, tune origins 2025-02..2025-09; data/hpo_nn_trials.csv), then rerun
with the best settings on all 13 origins with seeds 0-2 and 10-12 (as the 6-seed member) and with seeds 0-2 without last year's window
(eda/hpo_gpu_finals.py). Scored against the stored member S13ecx6:
  periods   tune (2025-02..09) | held out (2025-10..2026-02) | recent (2025-06..2026-02) | all 13 | early (2025-02..05)
  scores    the member alone and the five-member system with the submitted correction multiplier (WAPE / R3), within +/-20%, GMRAE; bootstrap of
            the system difference (items resampled)
  gate      without last year's window, like for like on 3 seeds: the tuned NN against the current NN (D324), alone and in the system
Also the tuned member's P25 / P75 (monthly quantiles summed; calibrated as D327). Writes data/fk_hpo_nn_preds.parquet; adds J["hpo_nn"];
stdout -> eda/hpo_nn_score_output.txt."""
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); K = pd.MultiIndex.from_arrays([D.origin, D.style_color]); y = D.y.values; ar = np.abs(D.R3.values - y); corr = (D.F / D.F0).values
QS = ["q10", "q25", "q50", "q75", "q90"]
tot = lambda q: q.groupby(["origin", "style_color"])[QS].sum().reindex(K)
A, B, LY = pd.read_parquet("data/fk_hpo_nn_a.parquet"), pd.read_parquet("data/fk_hpo_nn_b.parquet"), pd.read_parquet("data/fk_hpo_nn_ly.parquet")
x6 = (tot(A) + tot(B)) / 2; nn_t = x6.q50.values; nn_t3 = tot(A).q50.values; nn_tly = tot(LY).q50.values
nn_c3 = tot(pd.read_parquet("data/nn_backtest_preds_S13ec.parquet")).q50.values; nn_cly = tot(pd.read_parquet("data/fk_ly_nn_S13ecLY.parquet")).q50.values; nn_c = D.nn.values
for nm, v in [("tuned x6", nn_t), ("tuned LY", nn_tly), ("current LY", nn_cly)]: assert np.isfinite(v).all(), nm
MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; sysf = lambda nnv: (D.rules.values + D.qdyn.values + nnv + D.pf2c.values + D.pf2d32.values) / 5 * corr
P = {"tune": D.origin.between("2025-02", "2025-09").values, "held out": D.origin.between("2025-10", "2026-02").values, "recent": (D.origin >= "2025-06").values, "all 13": np.ones(len(D), bool), "early": D.origin.between("2025-02", "2025-05").values}
w = lambda f, m: float(np.abs(f - y)[m].sum() / ar[m].sum()); hit = lambda f, m: float((np.abs(f - y)[m] <= 0.2 * y[m]).mean()); gm = lambda f, m: float(np.exp(np.mean(np.log((np.abs(f - y)[m] + 1) / (ar[m] + 1)))))
rows = []
for lab, f_cur, f_tun in [("member alone", nn_c, nn_t), ("system (submitted corrections)", sysf(nn_c), sysf(nn_t))]:
    for pn, m in P.items(): rows.append(dict(view=lab, period=pn, current=r(w(f_cur, m)), tuned=r(w(f_tun, m)), diff=r(w(f_tun, m) - w(f_cur, m)), hit_current=r(hit(f_cur, m), 3), hit_tuned=r(hit(f_tun, m), 3), gmrae_current=r(gm(f_cur, m), 3), gmrae_tuned=r(gm(f_tun, m), 3)))
R = pd.DataFrame(rows); print(R.to_string(index=False))
rng = np.random.default_rng(0)
def boot(fa, fb, m, B_=1000):
    Z = pd.DataFrame({"s": D.style_color, "a": np.abs(fa - y), "b": np.abs(fb - y), "r": ar})[m].groupby("s").sum().values; n = len(Z); T = Z[rng.integers(0, n, (B_, n))].sum(1); d = T[:, 1] / T[:, 2] - T[:, 0] / T[:, 2]
    return [r(np.percentile(d, 2.5)), r(np.percentile(d, 97.5))]
ci = {pn: boot(sysf(nn_c), sysf(nn_t), P[pn]) for pn in ["held out", "recent", "all 13"]}; print("system difference, 95% interval:", ci)
ly = []
for pn in ["recent", "all 13"]:
    m = P[pn]; ly.append(dict(period=pn, alone_current3=r(w(nn_c3, m)), alone_tuned3=r(w(nn_t3, m)), alone_current_ly=r(w(nn_cly, m)), alone_tuned_ly=r(w(nn_tly, m)),
                              sys_current_ly=r(w(sysf(nn_cly), m)), sys_tuned_ly=r(w(sysf(nn_tly), m)), deg_current=r(w(nn_cly, m) - w(nn_c3, m)), deg_tuned=r(w(nn_tly, m) - w(nn_t3, m))))
print("without last year (3 seeds, like for like):"); print(pd.DataFrame(ly).to_string(index=False))
# P25 / P75 of the tuned member, calibrated as D327 (each side's log width scaled to 25% / 25%, factors left one origin out)
Q = x6.copy(); Q = pd.DataFrame(np.sort(Q.values, axis=1), columns=QS, index=Q.index).reset_index(); l50 = np.log(Q.q50.values + 1)
wl = np.maximum(l50 - np.log(Q.q25.values + 1), 0.02); wu = np.maximum(np.log(Q.q75.values + 1) - l50, 0.02); e = np.log(y + 1) - l50; Q["q25_cal"] = Q.q25; Q["q75_cal"] = Q.q75
for o in sorted(D.origin.unique()):
    tr = (D.origin != o).values; mo = (D.origin == o).values; kl = float(np.quantile(-e[tr] / wl[tr], 0.75)); ku = float(np.quantile(e[tr] / wu[tr], 0.75))
    Q.loc[mo, "q25_cal"] = np.exp(l50[mo] - kl * wl[mo]) - 1; Q.loc[mo, "q75_cal"] = np.exp(l50[mo] + ku * wu[mo]) - 1
rec = P["recent"]; cov = dict(raw=r(float(((y >= Q.q25.values) & (y <= Q.q75.values))[rec].mean()), 3), cal=r(float(((y >= Q.q25_cal.values) & (y <= Q.q75_cal.values))[rec].mean()), 3)); print("tuned NN P25-P75 inside (recent):", cov)
Q.to_parquet("data/fk_hpo_nn_preds.parquet")
T = pd.read_csv("data/hpo_nn_trials.csv"); best = json.load(open("data/hpo_nn_best.json"))
J = json.load(open("data/fk_err_analysis.json")); J["hpo_nn"] = dict(rows=rows, ci=ci, ly=ly, quantile_cover=cov, best=best["best"], base=best["base"], n_trials=int(len(T)), crit_range=[r(T.crit.min()), r(T.crit.max())])
json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0, default=float); print("saved")
