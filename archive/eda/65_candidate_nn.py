"""Candidate 2026 forecast with the embedding NN in the model answer (D081). Nothing in the submitted files is changed.
Runs the production chain eda/22 -> eda/32 -> eda/41 from their own source text with two edits and redirected file names:
(1) every quantile distribution of the production model (backtest and the June 2026 fit) is re-centred on the 50/50 geometric
    blend of its median and the NN's median 4-month total (sum of the NN's monthly P50s, eda/57, 3 seeds, no interaction);
    the spread is unchanged, and tau* / k* are re-fitted by eda/22 on the re-centred backtest as usual;
(2) outputs go to data/cand_nn_final_forecast_q.csv, ..._q3.csv, ..._q4.csv.
All rules (mature blend, floors, swim cap, campaign uplift, World Cup, out-of-stock item, July calibration) run unchanged."""
import sys, re, io, contextlib, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
_bt = sys.argv[1] if len(sys.argv) > 1 else "nn_backtest_preds_B13"; _fi = sys.argv[2:] if len(sys.argv) > 2 else ["nn_backtest_preds_final"]
NB = pd.read_parquet(f"data/{_bt}.parquet").groupby(["origin", "style_color"]).q50.sum()
NF = sum(pd.read_parquet(f"data/{f}.parquet").groupby("style_color").q50.sum() for f in _fi) / len(_fi)     # seed sets averaged
print("NN files:", _bt, _fi)
src22 = open("eda/22_quantile_final.py", encoding="utf-8").read()
src22 = src22.replace("R = Q[Q.origin.isin(REP)].copy(); L = np.log1p(R[QC].values); A = R.ytot.values",
    "R = Q[Q.origin.isin(REP)].copy()\n"
    "_nn = NB.reindex(pd.MultiIndex.from_arrays([R.origin, R.style_color])).values; _sh = np.where(np.isnan(_nn), 0.0, 0.5 * (np.log1p(_nn) - np.log1p(R.q50.values)))\n"
    "for _c in QC: R[_c] = np.expm1(np.log1p(R[_c].values) + _sh)\n"
    "print('re-centred backtest on NN blend: median shift', round(float(np.median(_sh)), 3))\n"
    "L = np.log1p(R[QC].values); A = R.ytot.values")
src22 = src22.replace("Lf = np.log1p(np.clip(lv[:, None] * np.exp(zq) - 1, 0, None))",
    "Lf = np.log1p(np.clip(lv[:, None] * np.exp(zq) - 1, 0, None))\n"
    "_nf = NF.reindex(te.style_color.values).values; _sf = np.where(np.isnan(_nf), 0.0, 0.5 * (np.log1p(_nf) - Lf[:, 3])); Lf = Lf + _sf[:, None]\n"
    "print('2026 fit re-centred on NN blend: median shift', round(float(np.median(_sf)), 3), '| aggregate median x', round(float(np.expm1(Lf[:, 3]).sum() / np.expm1(Lf[:, 3] - _sf).sum()), 3))")
src22 = src22.replace('out.to_csv("data/final_forecast_q.csv")', 'out.to_csv("data/cand_nn_final_forecast_q.csv")')
assert src22.count("NB.reindex") == 1 and src22.count("NF.reindex") == 1 and "cand_nn_final_forecast_q.csv" in src22
src32 = open("eda/32_final_v3.py", encoding="utf-8").read().replace('read_csv("data/final_forecast_q.csv"', 'read_csv("data/cand_nn_final_forecast_q.csv"').replace('to_csv("data/final_forecast_q3.csv")', 'to_csv("data/cand_nn_final_forecast_q3.csv")')
src41 = open("eda/41_july_calibration.py", encoding="utf-8").read().replace('read_csv("data/final_forecast_q3.csv"', 'read_csv("data/cand_nn_final_forecast_q3.csv"').replace('to_csv("data/final_forecast_q4.csv")', 'to_csv("data/cand_nn_final_forecast_q4.csv")')
assert "cand_nn_final_forecast_q3.csv" in src32 and "cand_nn_final_forecast_q4.csv" in src41
for name, s in [("22", src22), ("32", src32), ("41", src41)]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): exec(compile(s, f"eda/{name}_cand", "exec"), {"__name__": "__main__", "NB": NB, "NF": NF})
    txt = buf.getvalue(); open(f"data/cand_nn_{name}_output.txt", "w", encoding="utf-8").write(txt)
    print(f"--- {name}: " + " | ".join(l for l in txt.splitlines() if re.search(r"tau\* =|k\* =|re-centred|totals:|confidence:|July/June|aging factor", l))[:1500])
C = pd.read_csv("data/cand_nn_final_forecast_q4.csv", index_col=0); S = pd.read_csv("data/final_forecast_q4.csv", index_col=0)
M = pd.read_parquet("data/monthly_item.parquet"); jun = M.loc[S.index, "2026-06"].sum()
for n, X in [("submitted", S), ("candidate (NN blend)", C)]:
    m = X[["f_jul", "f_aug", "f_sep", "f_oct"]].sum()
    print(f"{n:22s} total {int(X.final_total.sum()):,} | months {m.astype(int).tolist()} | Jul/Jun {m.f_jul / jun:.3f} | Jul-Oct/Jun {m.sum() / jun:.3f} | conf {X.conf.value_counts().sort_index().to_dict()}")
