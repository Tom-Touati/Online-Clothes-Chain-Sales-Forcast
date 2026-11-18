"""Candidate 2026 forecast centred on rules + quantile model + NN (D105). Nothing in the submitted or NN-candidate files is changed.
Runs eda/65_candidate_nn.py from its own source (which itself runs the production chain eda/22 -> 32 -> 41) with two edits:
(1) every quantile distribution is re-centred on the equal-weight geometric mean of three medians: the production quantile model,
    the embedding NN (S13x6 backtest, finalS + finalS_s10 for 2026) and the rule agents' 4-month totals
    (agent_eval/<origin>/predictions_verified.csv; v1-v13 for the backtest, the June 2026 agent for the live fit).
    Where an item has no rules forecast the centre falls back to the two-model blend of D097. The spread is unchanged;
    tau* / k* are re-fitted by eda/22 on the re-centred backtest as usual.
(2) outputs go to data/cand_rnn_*.
All production rules (mature blend, floors, swim cap, campaign uplift, World Cup, out-of-stock item, July calibration) run unchanged."""
import sys, glob, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
def rules_totals(o): return pd.read_csv(f"agent_eval/{o}/predictions_verified.csv", dtype={"month": str}).groupby("style_color").pred.sum()
BT = [p.replace("\\", "/").split("/")[1] for p in glob.glob("agent_eval/*/predictions_verified.csv")]
RB = pd.concat({o: rules_totals(o) for o in BT if o != "2026-06"}); RB.index.names = ["origin", "style_color"]
RF = rules_totals("2026-06")
print("rules backtest origins:", sorted(set(RB.index.get_level_values(0))), "| live rules items:", len(RF))
s = open("eda/65_candidate_nn.py", encoding="utf-8").read()
E = [('0.5 * (np.log1p(_nn) - np.log1p(R.q50.values))',
      '(np.log1p(_nn) + _lr - 2 * np.log1p(R.q50.values)) / 3'),
     ('_nn = NB.reindex(pd.MultiIndex.from_arrays([R.origin, R.style_color])).values; ',
      '_nn = NB.reindex(pd.MultiIndex.from_arrays([R.origin, R.style_color])).values; _rb = RB.reindex(pd.MultiIndex.from_arrays([R.origin, R.style_color])).values; '
      '_lr = np.where(np.isnan(_rb), 0.5 * (np.log1p(_nn) + np.log1p(R.q50.values)), np.log1p(_rb)); print(\\"backtest rows with rules:\\", int((~np.isnan(_rb)).sum()), \\"of\\", len(_rb)); '),
     ('0.5 * (np.log1p(_nf) - Lf[:, 3])', '(np.log1p(_nf) + _lrf - 2 * Lf[:, 3]) / 3'),
     ('_nf = NF.reindex(te.style_color.values).values; ',
      '_nf = NF.reindex(te.style_color.values).values; _rf = RF.reindex(te.style_color.values).values; '
      '_lrf = np.where(np.isnan(_rf), 0.5 * (np.log1p(_nf) + Lf[:, 3]), np.log1p(_rf)); print(\\"2026 items with rules:\\", int((~np.isnan(_rf)).sum()), \\"of\\", len(_rf)); '),
     ('{"__name__": "__main__", "NB": NB, "NF": NF}', '{"__name__": "__main__", "NB": NB, "NF": NF, "RB": RB, "RF": RF}'),
     ('("candidate (NN blend)", C)', '("candidate (rules+q+NN)", C)')]
for a, b in E:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
s = s.replace("cand_nn_", "cand_rnn_").replace("re-centred backtest on NN blend", "re-centred backtest on rules+q+NN").replace("2026 fit re-centred on NN blend", "2026 fit re-centred on rules+q+NN")
sys.argv = ["eda/65", "nn_backtest_preds_S13x6", "nn_backtest_preds_finalS", "nn_backtest_preds_finalS_s10"]
exec(compile(s, "eda/80_from_65", "exec"), {"__name__": "__main__", "RB": RB, "RF": RF})
