"""Rule-agent evaluation harness (D103).
  python eda/76_rules_harness.py prepare <origin>   copy the previous origin's rules/notes into <origin>/prev and write
                                                    <origin>/feedback/scoreboard.csv: every earlier origin's forecasts for months <= origin,
                                                    against the actuals in this origin's own snapshot (so nothing after the cutoff)
  python eda/76_rules_harness.py run <origin>       run <origin>/rules.py in its folder (isolated Python), validate predictions.csv,
                                                    audit the file for paths outside its folder, keep a verified copy
"""
import sys, os, re, shutil, subprocess, pandas as pd, numpy as np
ROOT = os.path.abspath("agent_eval"); PY = os.path.abspath(".venv/Scripts/python.exe")
ORIGINS = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")] + ["2026-06"]   # 2026-06 = live origin (D105), prev = 2026-02
cmd, o = sys.argv[1], sys.argv[2]; od = f"{ROOT}/{o}"

if cmd == "prepare":
    k = ORIGINS.index(o)
    if k > 0:
        p = ORIGINS[k - 1]; os.makedirs(f"{od}/prev", exist_ok=True)
        for f in ["rules.py", "notes.md"]: shutil.copy(f"{ROOT}/{p}/{f}", f"{od}/prev/{f}")
        act = pd.read_parquet(f"{od}/data/monthly_item.parquet")[["style_color", "month", "units"]].rename(columns={"units": "actual"})
        rows = []
        for q in ORIGINS[:k]:
            pr = pd.read_csv(f"{ROOT}/{q}/predictions_verified.csv", dtype={"month": str}); pr = pr[pr.month <= o]
            rows.append(pr.assign(origin=q))
        sb = pd.concat(rows).merge(act, on=["style_color", "month"], how="left"); sb["actual"] = sb.actual.fillna(0)
        sb["h"] = sb.apply(lambda r: (pd.Period(r.month, "M") - pd.Period(r.origin, "M")).n, axis=1)
        os.makedirs(f"{od}/feedback", exist_ok=True)
        sb[["origin", "style_color", "month", "h", "pred", "actual"]].sort_values(["origin", "style_color", "month"]).to_csv(f"{od}/feedback/scoreboard.csv", index=False)
        g = sb.groupby(["origin", "month"]).apply(lambda x: pd.Series({"items": len(x), "pred": x.pred.sum(), "actual": x.actual.sum(),
                                                                      "wape": (x.pred - x.actual).abs().sum() / x.actual.sum()})).reset_index()
        g["bias"] = g.pred / g.actual - 1
        g.round(4).to_csv(f"{od}/feedback/scoreboard_summary.csv", index=False)
        print(g.round(3).tail(8).to_string(index=False))
    print("prepared", o)

elif cmd == "run":
    src = open(f"{od}/rules.py", encoding="utf-8").read()
    bad = [m for m in re.findall(r"""['"]([^'"]*[/\\][^'"]*)['"]""", src) if (".." in m or re.match(r"^[A-Za-z]:|^/", m) or "actuals" in m or "eda" in m.split("/")[0])]
    imports = sorted(set(re.findall(r"^\s*(?:import|from)\s+([\w\.]+)", src, re.M)))
    print("imports:", imports, "| suspicious paths:", bad or "none")
    if os.path.exists(f"{od}/predictions.csv"): os.remove(f"{od}/predictions.csv")
    r = subprocess.run([PY, "-I", "rules.py"], cwd=od, capture_output=True, text=True, timeout=600)
    print(r.stdout[-1500:], r.stderr[-1500:])
    pr = pd.read_csv(f"{od}/predictions.csv", dtype={"month": str})
    tf = pd.read_csv(f"{od}/data/to_forecast.csv"); months = [str(pd.Period(o, "M") + h) for h in range(1, 5)]
    need = pd.MultiIndex.from_product([tf.style_color, months]); got = pr.set_index(["style_color", "month"]).index
    assert set(need) == set(got) and len(got) == len(need), f"coverage: missing {len(set(need) - set(got))}, extra {len(set(got) - set(need))}"
    assert np.isfinite(pr.pred).all() and (pr.pred >= 0).all(), "non-finite or negative predictions"
    pr[["style_color", "month", "pred"]].to_csv(f"{od}/predictions_verified.csv", index=False)
    print(f"ok {o}: {len(tf)} items x 4 months, total {pr.pred.sum():,.0f}")
