"""Backtest DRIFT_ADJ variants of v10 (origins Feb-Oct-25, site levels at actuals)."""
import sys
import pandas as pd
sys.path.insert(0, ".")
import rules as R10
it, di, st = R10.load()
sb = pd.read_csv("feedback/scoreboard.csv")
act_m = di.groupby(["style_color", "m"]).units.sum()
for dadj in [1.0, 1.01, 1.02]:
    R10.P["DRIFT_ADJ"] = dadj
    res = []
    for org in sorted(sb.origin.unique()):
        if org == "2025-11":
            continue
        its = sorted(sb[sb.origin == org].style_color.unique())
        pr = R10.forecast(pd.Period(org).end_time.normalize(), its, it, di, st)
        cols = [c for c in pr.columns if pd.Period(c) <= pd.Period("2025-11")]
        a = pd.DataFrame({c: act_m.xs(pd.Period(c), level="m").reindex(its).fillna(0) for c in cols})
        P_, A = pr[cols].sum(axis=1), a.sum(axis=1)
        r = P_ / A.clip(lower=1e-9)
        res.append(dict(origin=org, wape=(P_ - A).abs().sum() / A.sum(), bias=P_.sum() / A.sum() - 1,
                        w20=((r - 1).abs() <= .2).mean(), w50=((r - 1).abs() <= .5).mean()))
    r = pd.DataFrame(res).set_index("origin")
    print(dadj, "mean", r.mean().round(4).to_dict(), "| bias by origin", r.bias.round(3).to_dict())
