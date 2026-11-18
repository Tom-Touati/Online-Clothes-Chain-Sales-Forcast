"""Buying-simulation replay of the candidate (D081) against the submitted method, at past origins: eda/51_rules_any_origin.py
run from its own source with one change, the model P50 total = 50/50 geometric blend of the quantile model's P50 and the NN's
median 4-month total (monthly split unchanged), so every rule variant is rebuilt on top of the candidate answer.
Usage: python eda/66_rules_replay_nn.py 2025-06 ; then 'python eda/66_rules_replay_nn.py summary' compares with data/sim_rules_test_{O}.csv."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
if sys.argv[1] == "summary":
    rows = []
    for O in [str(p) for p in pd.period_range("2025-03", "2025-12", freq="M")]:
        try: a = pd.read_csv(f"data/sim_rules_test_{O}.csv"); b = pd.read_csv(f"data/sim_rules_test_nnS_{O}.csv")
        except FileNotFoundError: continue
        for v in ["model P50", "all rules"]:
            for al in (0.0, 0.5):
                x = a[(a.variant == v) & (a["items"] == "all") & (a.alpha == al)].iloc[0]; y = b[(b.variant == v) & (b["items"] == "all") & (b.alpha == al)].iloc[0]
                rows.append(dict(origin=O, variant=v, alpha=al, loss_sub=x.loss_pct, loss_cand=y.loss_pct, wape_sub=x.wape, wape_cand=y.wape, bias_sub=x.bias, bias_cand=y.bias))
    R = pd.DataFrame(rows); print(R.round(3).to_string(index=False))
    for (v, al), g in R.groupby(["variant", "alpha"]):
        print(f"{v:10s} alpha={al}: mean loss {g.loss_sub.mean():.4f} -> {g.loss_cand.mean():.4f} (better at {int((g.loss_cand < g.loss_sub).sum())}/{len(g)}) | WAPE {g.wape_sub.mean():.4f} -> {g.wape_cand.mean():.4f} (better at {int((g.wape_cand < g.wape_sub).sum())}/{len(g)})")
    sys.exit()
O = sys.argv[1]
NB = pd.read_parquet("data/nn_backtest_preds_S13x6.parquet").groupby(["origin", "style_color"]).q50.sum()
src = open("eda/51_rules_any_origin.py", encoding="utf-8").read()
old = "base = b.values * (q.q50.values / b.sum(axis=1).values)[:, None]"
assert src.count(old) == 1
src = src.replace(old, "_nn = NB.loc[O].reindex(I).values; _q = np.where(np.isnan(_nn), q.q50.values, np.expm1(0.5 * np.log1p(q.q50.values) + 0.5 * np.log1p(np.nan_to_num(_nn))))\n"
                       "base = b.values * (_q / b.sum(axis=1).values)[:, None]")
src = src.replace('R.to_csv(f"data/sim_rules_test_{O}.csv", index=False)', 'R.to_csv(f"data/sim_rules_test_nnS_{O}.csv", index=False)')
sys.argv = ["eda/51_rules_any_origin.py", O]
exec(compile(src, "eda/51_cand", "exec"), {"__name__": "__main__", "NB": NB})
