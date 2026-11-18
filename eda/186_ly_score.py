"""D351 scoring: fewer last-year features (ec_ns_lyA..D) and partial pooling of the seasonal ratios (ec_ns_shr, ec_ns_shr25; D297 / #10b),
all run on the CPU server 18.234.59.109 (data/srv2/). Each alone, in the system (rules, member, new NN, pf2c, pf2d32) and with the submitted
correction multiplier, on normal training and with last year's window removed (LYDROP, the fork's D315 gate); bootstrap against ec_ns
from the same server on the same footing. Writes eda/ly_score_output.txt (stdout)."""
import sys, os, contextlib, io, importlib.util, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); REC_ = [o for o in ORIGINS if o >= "2025-06"]
rd = lambda f: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).q50.reindex(key).values
B = pd.read_parquet("data/dec_base_ecnsnn.parquet").set_index(["origin", "style_color"]).reindex(key)
G = pd.read_parquet("data/geo_ecnsnn_x_preds.parquet").set_index(["origin", "style_color"]).reindex(key); mult = (G.geo / G.F.clip(lower=1e-9)).fillna(1).values
AM = lambda *xs: np.nanmean(np.column_stack([np.clip(x, 0, None) for x in xs]), axis=1)
NAMES = {"ec_ns": "ec_ns (all 7 LY features)", "ec_ns_lyA": "no LY features", "ec_ns_lyB": "type LY only", "ec_ns_lyC": "type + own LY", "ec_ns_lyD": "all but the item's own LY",
         "ec_ns_shr": "pooled seasonality 0.5", "ec_ns_shr25": "pooled seasonality 0.25"}
V = {}
for v in NAMES:
    for sfx, nm in [("", "full"), ("_noly", "no LY window")]:
        f = f"data/srv2/q_backtest_nx_{v}{sfx}.parquet"
        if os.path.exists(f): V[(v, nm)] = rd(f)
if ("ec_ns", "full") in V:
    loc = rd("data/q_backtest_nx_ec_ns.parquet"); print("reproduction, server ec_ns vs laptop: max rel diff", float(np.nanmax(np.abs(V[("ec_ns", "full")] - loc) / np.maximum(loc, 1))))
for (v, nm), x in V.items():
    T[f"m|{v}|{nm}"] = x; T[f"S|{v}|{nm}"] = AM(B.rules.values, x, B.nn.values, B.pf2c.values, B.pf2d32.values); T[f"C|{v}|{nm}"] = T[f"S|{v}|{nm}"] * mult
rows = []
for (v, nm) in V:
    r = {"variant": NAMES[v], "training": nm}
    for sub, m in [("recent", T.origin.isin(REC_)), ("all13", np.ones(len(T), bool))]:
        Z = T[m]; e3 = (Z.R3 - Z.y).abs().sum()
        for pre in ["m", "S", "C"]: r[f"{pre}_{sub}"] = round(float((Z[f"{pre}|{v}|{nm}"] - Z.y).abs().sum() / e3), 4)
        r[f"bias_C_{sub}"] = round(float(Z[f"C|{v}|{nm}"].sum() / Z.y.sum() - 1), 4)
        if v != "ec_ns" and ("ec_ns", nm) in V: lo, hi = sl.boot(Z, f"C|{v}|{nm}", f"C|ec_ns|{nm}"); r[f"boot_{sub}"] = f"{lo:+.4f}..{hi:+.4f}"
    rows.append(r)
print("m = member alone, S = five-member system, C = system x the submitted correction multiplier; WAPE / R3; boot = C vs ec_ns on the same training")
print(pd.DataFrame(rows).to_string(index=False))
