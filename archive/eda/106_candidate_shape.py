"""Candidate with the D120 monthly split (D121): the rules + quantile + NN candidate (eda/80, data/cand_rnn_final_forecast_q.csv) run
through eda/32 from its own source with two edits, and without the D061 July calibration (eda/41):
 (1) the model monthly profile is the geometric mean of the June 2026 rule agent's monthly forecasts (agent_eval/2026-06) and the NN's
     monthly P50 (finalS + finalS_s10 averaged) instead of the LightGBM point model's profile; items missing either fall back to the old profile;
 (2) no 50/50 mix with the item's own Jul-Oct 2025 shape (that mix raised the backtest shape error from 0.177 to 0.220 and caused the
     July under-allocation that D061 then patched: July share 0.244 against an actual 0.293 at the June 2025 origin).
Item totals are unchanged (year-on-year blend, segment rules, campaign uplift, World Cup, out-of-stock item all run as before);
segment items without a year of history keep their segment shape. Writes data/cand_rnns_final_forecast_q3.csv and prints the comparison."""
import sys, io, contextlib, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
R = pd.read_csv("agent_eval/2026-06/predictions_verified.csv", dtype={"month": str}).pivot(index="style_color", columns="month", values="pred")
R.columns = ["jul", "aug", "sep", "oct"]
N = sum(pd.read_parquet(f"data/{f}.parquet").pivot(index="style_color", columns="h", values="q50") for f in ["nn_backtest_preds_finalS", "nn_backtest_preds_finalS_s10"]) / 2
N.columns = ["jul", "aug", "sep", "oct"]
GEO = np.exp((np.log(R.clip(lower=0.01)) + np.log(N.reindex(R.index).clip(lower=0.01))) / 2)
s = open("eda/32_final_v3.py", encoding="utf-8").read()
E = [('O = pd.read_csv("data/final_forecast_q.csv", index_col=0)', 'O = pd.read_csv("data/cand_rnn_final_forecast_q.csv", index_col=0)'),
     ('prof = W[["jul", "aug", "sep", "oct"]].div(W[["jul", "aug", "sep", "oct"]].sum(axis=1), axis=0).loc[T]; prof.columns = ["jul", "aug", "sep", "oct"]',
      'prof = W[["jul", "aug", "sep", "oct"]].div(W[["jul", "aug", "sep", "oct"]].sum(axis=1), axis=0).loc[T]; prof.columns = ["jul", "aug", "sep", "oct"]\n'
      '_g = GEO.reindex(T); _ok = _g.notna().all(axis=1) & (_g.sum(axis=1) > 0); prof.loc[_ok] = _g[_ok].div(_g[_ok].sum(axis=1), axis=0).values\n'
      'print("items on the rules + NN profile:", int(_ok.sum()), "of", len(T))'),
     ('shape = prof.copy(); shape[mm] = 0.5 * prof[mm] + 0.5 * own[mm]', 'shape = prof.copy()          # D121: no mix with the own last-year shape'),
     ('out.to_csv("data/final_forecast_q3.csv")', 'out.to_csv("data/cand_rnns_final_forecast_q3.csv")')]
for a, b in E:
    assert s.count(a) == 1, a[:60]; s = s.replace(a, b)
buf = io.StringIO()
with contextlib.redirect_stdout(buf): exec(compile(s, "eda/32_shape", "exec"), {"__name__": "__main__", "GEO": GEO})
print("\n".join(l for l in buf.getvalue().splitlines() if "profile" in l or "totals" in l or "by month" in l))
C = pd.read_csv("data/cand_rnns_final_forecast_q3.csv", index_col=0); P = pd.read_csv("data/cand_rnn_final_forecast_q4.csv", index_col=0); S = pd.read_csv("data/final_forecast_q4.csv", index_col=0)
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; jun = M.loc[C.index, "2026-06"].sum()
MC = ["f_jul", "f_aug", "f_sep", "f_oct"]
hist = {y: M.loc[C.index, [f"{y}-07", f"{y}-08", f"{y}-09", f"{y}-10"]].sum() / M.loc[C.index, f"{y}-06"].sum() for y in (2024, 2025)}
for n, X in [("submitted", S), ("candidate rules+q+NN (old split + July calibration)", P), ("candidate rules+q+NN, rules + NN split", C)]:
    m = X[MC].sum(); print(f"{n:52s} total {int(m.sum()):,} | months {m.astype(int).tolist()} | Jul/Jun {m.f_jul / jun:.3f} | shares {np.round((m / m.sum()).values, 3).tolist()}")
for y, h in hist.items(): print(f"history {y} (same 100 items, month / June): {np.round(h.values, 3).tolist()} | shares {np.round((h / h.sum()).values, 3).tolist()}")
