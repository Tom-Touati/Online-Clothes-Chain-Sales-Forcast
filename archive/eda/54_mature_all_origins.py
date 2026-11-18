"""Mature blend at every origin with a known 4-month outcome and a full prior year (2025-03 .. 2026-02), with the ageing factor
re-fitted without the tested origin (leave-one-out) and, where possible, from earlier origins only (past-only: outcomes known
by the origin). Runs eda/51_rules_any_origin.py per origin and collects the mature-item rows."""
import pandas as pd, numpy as np, subprocess, sys
sys.stdout.reconfigure(encoding="utf-8")
M = pd.read_parquet("data/monthly_item.parquet"); cols = list(M.columns); it = pd.read_parquet("data/items.parquet").set_index("style_color")
start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
Q = pd.read_parquet("data/q_backtest.parquet"); LR = {}
for o, g in Q.groupby("origin"):
    k = cols.index(o)
    if k < 14: continue
    ids = g.style_color.values; old = (start.reindex(ids) < pd.Period(cols[k - 14]).start_time).values
    ly = M.loc[ids, cols[k - 11:k - 7]].sum(axis=1).values; yn = ly * (M.loc[ids, cols[k - 2:k + 1]].sum(axis=1).values + 5) / (M.loc[ids, cols[k - 14:k - 11]].sum(axis=1).values + 5)
    keep = old & (ly >= 50); LR[o] = np.log((g.ytot.values[keep] + 5) / (yn[keep] + 5))
print("per-origin ageing factor:", {o: round(float(np.exp(np.median(v))), 3) for o, v in LR.items()})
print("pooled:", round(float(np.exp(np.median(np.concatenate(list(LR.values()))))), 3))
rows = []
for o in [o for o in LR if o <= "2025-12"]:
    loo = float(np.exp(np.median(np.concatenate([v for p, v in LR.items() if p != o]))))
    past = [v for p, v in LR.items() if pd.Period(p) + 4 <= pd.Period(o)]
    pf = float(np.exp(np.median(np.concatenate(past)))) if past else np.nan
    for tag, a in [("loo", loo)] + ([("past-only", pf)] if past else []):
        if subprocess.run([sys.executable, "eda/51_rules_any_origin.py", o, str(a)], capture_output=True).returncode: break   # window needs backtest origins not available (2026-01, 2026-02)
        R = pd.read_csv(f"data/sim_rules_test_{o}.csv"); R = R[(R["items"] == "mature") & R.variant.isin(["model P50", "+ mature blend"])]
        for _, r in R.iterrows(): rows.append(dict(origin=o, ageing=tag, factor=round(a, 3), **r.to_dict()))
    print(o, "done", flush=True)
R = pd.DataFrame(rows); R.to_csv("data/mature_blend_all_origins.csv", index=False)
for tag in ["loo", "past-only"]:
    x = R[(R.ageing == tag) & (R.alpha == 0.0)].pivot_table(index=["origin", "factor", "n"], columns="variant", values=["wape", "bias", "loss_pct"]).round(3)
    x["wape_better"] = x[("wape", "+ mature blend")] < x[("wape", "model P50")]; x["loss_better"] = x[("loss_pct", "+ mature blend")] < x[("loss_pct", "model P50")]
    print(f"\n== ageing {tag} (alpha 0)"); print(x.to_string())
    print("blend better on WAPE in", int(x.wape_better.sum()), "of", len(x), "| on loss in", int(x.loss_better.sum()), "of", len(x),
          "| mean WAPE model", round(x[("wape", "model P50")].mean(), 3), "blend", round(x[("wape", "+ mature blend")].mean(), 3),
          "| mean loss model", round(x[("loss_pct", "model P50")].mean(), 3), "blend", round(x[("loss_pct", "+ mature blend")].mean(), 3))
y = R[(R.ageing == "loo") & (R.alpha == 0.5)].pivot_table(index="origin", columns="variant", values="loss_pct")
print("\nalpha 0.5 loo: blend better on loss in", int((y["+ mature blend"] < y["model P50"]).sum()), "of", len(y))
