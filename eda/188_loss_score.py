"""D353 scoring of the loss / M5-trick grid (eda/187_losses.py, run on Main's CPU server 18.234.59.109; files copied to data/srv4/).
For each variant: the member alone, the five-member system (rules, member, new NN, pf2c, pf2d32) and the system x the submitted correction
multiplier (geo / F of data/geo_ecnsnn_x_preds.parquet); WAPE / R3 on the nine recent origins, all 13 and the four early origins, the
item-equal GMRAE (typical item / R3), within +/-20% (items), bias; the same with last year's window removed (_noly). Bootstrap of the
corrected system against the l1log baseline (= the current loss) on the same training. Writes eda/loss_score_output.txt (stdout)."""
import sys, os, contextlib, io, importlib.util, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); REC = T.origin >= "2025-06"; EARLY = T.origin.isin(["2025-02", "2025-03", "2025-04", "2025-05"])
rd = lambda f: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).q50.reindex(key).values
B = pd.read_parquet("data/dec_base_ecnsnn.parquet").set_index(["origin", "style_color"]).reindex(key)
G = pd.read_parquet("data/geo_ecnsnn_x_preds.parquet").set_index(["origin", "style_color"]).reindex(key); mult = (G.geo / G.F.clip(lower=1e-9)).fillna(1).values
AM = lambda *xs: np.nanmean(np.column_stack([np.clip(x, 0, None) for x in xs]), axis=1)
V = os.environ.get("VLIST", "l1log l2log huber fair q55 q60 tweedie poisson wape w05 w10 recency permacro mix_tw mix_q").split()
y, R3 = T.y.values, T.R3.values
def met(F, m):
    e, e3 = np.abs(F[m] - y[m]), np.abs(R3[m] - y[m])
    return {"wape_r3": e.sum() / e3.sum(), "gmrae": np.exp(np.mean(np.log((e + 1) / (e3 + 1)))), "within20": np.mean((e <= 0.2 * y[m]) & (y[m] > 0)), "bias": F[m].sum() / y[m].sum() - 1}
rows = []
loc = rd("data/q_backtest_nx_ec_ns.parquet"); base_srv = rd("data/srv4/q_backtest_ls_l1log.parquet")
print("l1log on the server vs the stored ec_ns member: max rel diff", float(np.nanmax(np.abs(base_srv - loc) / np.maximum(loc, 1))))
for v in V:
    for sfx, tr in [("", "full"), ("_noly", "no LY window")]:
        f = f"data/srv4/q_backtest_ls_{v}{sfx}.parquet"
        if not os.path.exists(f): continue
        x = rd(f); S = AM(B.rules.values, x, B.nn.values, B.pf2c.values, B.pf2d32.values); C = S * mult
        T[f"C|{v}|{sfx}"] = C
        r = {"loss": v, "training": tr}
        a = met(x, REC.values); r |= {"member_rec": a["wape_r3"], "member_bias": a["bias"]}
        s = met(S, REC.values); r |= {"system_rec": s["wape_r3"]}
        c = met(C, REC.values); r |= {"corr_rec": c["wape_r3"], "corr_gmrae": c["gmrae"], "corr_w20": c["within20"], "corr_bias": c["bias"]}
        r["corr_all13"] = met(C, np.ones(len(T), bool))["wape_r3"]; r["corr_early"] = met(C, EARLY.values)["wape_r3"]
        rows.append(r)
R = pd.DataFrame(rows).round(4); print("member = the tree member alone; system = five-member mean; corr = system x the submitted corrections; WAPE / R3 unless named")
print(R.to_string(index=False))
print("\n== bootstrap, corrected system vs l1log on the same training (recent | all 13)")
for v in V[1:]:
    for sfx in ["", "_noly"]:
        if f"C|{v}|{sfx}" in T and f"C|l1log|{sfx}" in T:
            print(f"  {v:9s} {sfx or 'full':6s}", np.round(sl.boot(T[REC], f"C|{v}|{sfx}", f"C|l1log|{sfx}"), 4), np.round(sl.boot(T, f"C|{v}|{sfx}", f"C|l1log|{sfx}"), 4))
