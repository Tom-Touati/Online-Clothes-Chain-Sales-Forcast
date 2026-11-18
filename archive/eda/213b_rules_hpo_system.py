"""Fork 2 (D389b): system-level check of the tuned rule settings. The rules v14 harness (agent_eval/2026-06/rules.py, oracle site) is run
at every backtest origin for the scored items of data/fk_err_master.parquet, once with v14 and once with the tuned P (data/hpo_rules_best.json),
and each replaces the rules member in the five-member mean (x the submitted correction multiplier). Like for like between the two; both
differ from the stored rules member (each origin's own agent version). Writes data/rules_hpo_system.parquet."""
import os, sys, json, numpy as np, pandas as pd
ROOT = os.getcwd(); D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True)
bp = json.load(open("data/hpo_rules_best.json"))["best"]; a = bp.pop("AMP_FLOOR"); bp["AMP_CLIP"] = (a, 1.0)
os.chdir("agent_eval/2026-06"); sys.path.insert(0, "."); sys.argv = ["x"]
import rules as R
it, di, st = R.load(); BASE = dict(R.P); out = []
for o in sorted(D.origin.unique()):
    B = pd.Period(o, "M"); items = D[D.origin == o].style_color.tolist(); ms = [str(B + h) for h in range(1, 5)]
    row = {}
    for nm, over in (("r_v14", {}), ("r_tuned", bp)):
        R.P.clear(); R.P.update(BASE); R.P.update(over)
        f = R.forecast(B.end_time.normalize(), items, it, di, st); row[nm] = f.reindex(items)[ms].sum(axis=1).values
    out.append(pd.DataFrame({"origin": o, "style_color": items, **row})); print(o, flush=True)
os.chdir(ROOT); P = pd.concat(out); P.to_parquet("data/rules_hpo_system.parquet", index=False)
D = D.merge(P, on=["origin", "style_color"], how="left"); y = D.y.values; ar = np.abs(D.R3 - y).values; corr = (D.F / D.F0).values
w = lambda f, m: round(float(np.abs(f - y)[m].sum() / ar[m].sum()), 4); b = lambda f, m: round(100 * (f[m].sum() - y[m].sum()) / y[m].sum(), 1)
rec = (D.origin >= "2025-06").values; ho = (D.origin >= "2025-10").values; al = np.ones(len(D), bool); er = ~rec; tu = (D.origin <= "2025-09").values
O = D[["qdyn", "nn", "pf2c", "pf2d32"]].values.sum(1)
for nm in ("rules", "r_v14", "r_tuned"):
    f = D[nm].fillna(D.R3).values; s = (O + f) / 5 * corr
    print(f"{nm:8s} alone tune {w(f, tu)} held {w(f, ho)} recent {w(f, rec)} all {w(f, al)} | system tune {w(s, tu)} held {w(s, ho)} recent {w(s, rec)} all {w(s, al)} early {w(s, er)} bias {b(s, rec)}")
rng = np.random.default_rng(0); sa = (O + D.r_v14.fillna(D.R3).values) / 5 * corr; sb = (O + D.r_tuned.fillna(D.R3).values) / 5 * corr
Z = pd.DataFrame({"s": D.style_color, "a": np.abs(sa - y), "b": np.abs(sb - y), "r": ar})[ho].groupby("s").sum().values; T = Z[rng.integers(0, len(Z), (1000, len(Z)))].sum(1)
print("tuned - v14, held-out system, 95%:", np.percentile((T[:, 1] - T[:, 0]) / T[:, 2], [2.5, 97.5]).round(4))
