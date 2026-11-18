"""D327 (fork; user requirement relayed by Fork 2: "all the models should output 0.25 quantile and 0.75 quantile, not only median"). P25 / P50 / P75
(and P10 / P90) of the 4-month totals for the fork's three members, the 13 backtest origins and June 2026, in one file for Main's system quantiles:
  nn      S13ecx6 (backtest data/nn_backtest_preds_S13ecx6.parquet; 2026 = the mean of finalSec and finalSec_s10, as the member): monthly quantiles
          summed over the four months, as the member's median is (exact if an item's months move together; otherwise slightly wide)
  pf2c    conformal TabPFN v2 dyn (data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet; 2026 data/fk_2026_pf2c.parquet)
  pf2d32  TabPFN v2 dyn, 32 estimators (data/q_backtest_fktab_tabpfn_v2_dyn_e32.parquet; 2026 data/fk_2026_pf2d32.parquet)
Quantiles sorted per row. Checks: P50 equals the member in data/fk_err_master(_2026).parquet; the share of actuals inside P25-P75 (nominal 50%).
Also q25_cal / q75_cal: each side's log width scaled to 25% below / 25% above, factors fitted leaving one origin out (2026: all 13 origins).
Writes data/fk_member_quantiles.parquet; adds J["memberq"]; stdout -> eda/member_quantiles_output.txt."""
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 200)
QS = ["q10", "q25", "q50", "q75", "q90"]
def totals(f):
    q = pd.read_parquet(f); return (q.groupby(["origin", "style_color"])[QS].sum() if "h" in q.columns and q.h.nunique() > 1 else q.set_index(["origin", "style_color"])[QS])
nn26 = (totals("data/nn_backtest_preds_finalSec.parquet") + totals("data/nn_backtest_preds_finalSec_s10.parquet")) / 2
SRC = {"nn": (totals("data/nn_backtest_preds_S13ecx6.parquet"), nn26), "pf2c": (totals("data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet"), totals("data/fk_2026_pf2c.parquet")),
       "pf2d32": (totals("data/q_backtest_fktab_tabpfn_v2_dyn_e32.parquet"), totals("data/fk_2026_pf2d32.parquet"))}
D = pd.read_parquet("data/fk_err_master.parquet"); T = pd.read_parquet("data/fk_err_master_2026.parquet"); KB = pd.MultiIndex.from_arrays([D.origin, D.style_color]); KT = pd.MultiIndex.from_arrays([T.origin, T.style_color])
parts, chk = [], []
for m, (bt, t26) in SRC.items():
    for lab, src, K, ref in [("backtest", bt, KB, D), ("2026-06", t26, KT, T)]:
        Q = src.reindex(K); Q = pd.DataFrame(np.sort(Q.values, axis=1), columns=QS, index=K); ok = Q.q50.notna().values
        eq = float(np.nanmax(np.abs(Q.q50.values[ok] - ref[m].values[ok]) / np.maximum(ref[m].values[ok], 1))); P = Q.reset_index(); P.insert(2, "member", m); parts.append(P)
        row = dict(member=m, part=lab, n=int(ok.sum()), max_rel_diff_q50=round(eq, 6))
        if lab == "backtest":
            y = D.y.values; rec = (D.origin >= "2025-06").values
            for nm, mm in [("recent", rec), ("all13", np.ones(len(D), bool))]:
                row[f"inside_25_75_{nm}"] = round(float(((y >= Q.q25.values) & (y <= Q.q75.values))[mm].mean()), 3); row[f"inside_10_90_{nm}"] = round(float(((y >= Q.q10.values) & (y <= Q.q90.values))[mm].mean()), 3)
                row[f"below_25_{nm}"] = round(float((y < Q.q25.values)[mm].mean()), 3); row[f"above_75_{nm}"] = round(float((y > Q.q75.values)[mm].mean()), 3)
        chk.append(row)
O = pd.concat(parts, ignore_index=True)
# the NN's own P25 / P75 are far too narrow (about 12% of actuals inside instead of 50%): calibrated columns for all three members, the log width
# on each side scaled so that 25% of actuals fall below P25 and 25% above P75; factors fitted leaving one origin out (as D317), 2026 with all 13
O["q25_cal"] = O.q25; O["q75_cal"] = O.q75; cal = []
for m in SRC:
    B = O[(O.member == m) & (O.origin <= "2026-02")].merge(D[["origin", "style_color", "y"]], on=["origin", "style_color"]); l50 = np.log(B.q50 + 1)
    wl = np.maximum(l50 - np.log(B.q25 + 1), 0.02); wu = np.maximum(np.log(B.q75 + 1) - l50, 0.02); e = np.log(B.y + 1) - l50
    kf = lambda msk: (float(np.quantile(-e[msk] / wl[msk], 0.75)), float(np.quantile(e[msk] / wu[msk], 0.75)))
    for o in sorted(B.origin.unique()):
        kl, ku = kf((B.origin != o).values); mo = (B.origin == o).values; idx = O.index[(O.member == m) & (O.origin == o)]
        Bo = B[mo].set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([O.loc[idx, "origin"], O.loc[idx, "style_color"]]))
        lo = np.log(Bo.q50.values + 1); O.loc[idx, "q25_cal"] = np.exp(lo - kl * np.maximum(lo - np.log(Bo.q25.values + 1), 0.02)) - 1; O.loc[idx, "q75_cal"] = np.exp(lo + ku * np.maximum(np.log(Bo.q75.values + 1) - lo, 0.02)) - 1
    klA, kuA = kf(np.ones(len(B), bool)); i26 = O.index[(O.member == m) & (O.origin == "2026-06")]; l26 = np.log(O.loc[i26, "q50"] + 1)
    O.loc[i26, "q25_cal"] = np.exp(l26 - klA * np.maximum(l26 - np.log(O.loc[i26, "q25"] + 1), 0.02)) - 1; O.loc[i26, "q75_cal"] = np.exp(l26 + kuA * np.maximum(np.log(O.loc[i26, "q75"] + 1) - l26, 0.02)) - 1
    Bc = O[(O.member == m) & (O.origin <= "2026-02")].merge(D[["origin", "style_color", "y"]], on=["origin", "style_color"]); r = (Bc.origin >= "2025-06").values
    cal.append(dict(member=m, kl_all13=round(klA, 3), ku_all13=round(kuA, 3), inside_cal_recent=round(float(((Bc.y >= Bc.q25_cal) & (Bc.y <= Bc.q75_cal))[r].mean()), 3), below_cal_recent=round(float((Bc.y < Bc.q25_cal)[r].mean()), 3),
                    above_cal_recent=round(float((Bc.y > Bc.q75_cal)[r].mean()), 3), inside_cal_all13=round(float(((Bc.y >= Bc.q25_cal) & (Bc.y <= Bc.q75_cal)).mean()), 3)))
print(pd.DataFrame(cal).to_string(index=False)); O.to_parquet("data/fk_member_quantiles.parquet"); C = pd.DataFrame(chk); print(C.to_string(index=False))
print("rows:", O.groupby(["member"]).size().to_dict(), "| origins:", O.origin.nunique())
J = json.load(open("data/fk_err_analysis.json")); J["memberq"] = dict(checks=chk, calibrated=cal); json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
