"""D298 scoring: last-year-nulling augmentation of the tree member (eda/180q_models.py ec_ns_aug on the CPU server; data/srv/*).
Views: aug (rows twice: as is + last-year features nulled; normal features at prediction), aug_tta (average of the normal and nulled views,
in log), aug_null (nulled view only). Each alone and in the system (rules, member, new NN S13ecx6, pf2c, pf2d32 arithmetic mean), before
corrections and with the submitted correction multiplier (geo / F of data/geo_ecnsnn_x_preds.parquet), on normal training and with last
year's window removed (LYDROP: training origins o - 12 +/- 1 dropped, the fork's D315 gate). Bootstrap against ec_ns on the same footing.
Writes eda/aug_score_output.txt (stdout)."""
import sys, contextlib, io, importlib.util, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); REC_ = [o for o in ORIGINS if o >= "2025-06"]
rd = lambda f, c="q50": pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"])[c].reindex(key).values
B = pd.read_parquet("data/dec_base_ecnsnn.parquet").set_index(["origin", "style_color"]).reindex(key)
G = pd.read_parquet("data/geo_ecnsnn_x_preds.parquet").set_index(["origin", "style_color"]).reindex(key); mult = (G.geo / G.F.clip(lower=1e-9)).fillna(1).values
AM = lambda *xs: np.nanmean(np.column_stack([np.clip(x, 0, None) for x in xs]), axis=1)
V = {}
for sfx, nm in [("", "full"), ("_noly", "no LY")]:
    V[f"ec_ns | {nm}"] = rd(f"data/srv/q_backtest_nx_ec_ns{sfx}.parquet")
    for c, lab in [("q50", "aug"), ("q50_tta", "aug_tta"), ("q50_null", "aug_null")]: V[f"{lab} | {nm}"] = rd(f"data/srv/q_backtest_nx_ec_ns_aug{sfx}.parquet", c)
COLS = []
for k, v in V.items():
    T[f"m:{k}"] = v; T[f"S:{k}"] = AM(B.rules.values, v, B.nn.values, B.pf2c.values, B.pf2d32.values); T[f"S+corr:{k}"] = T[f"S:{k}"] * mult
for nm, m in [("nine recent origins", T.origin.isin(REC_)), ("all 13 origins", np.ones(len(T), bool))]:
    for pre in ["m:", "S:", "S+corr:"]:
        cols = [f"{pre}{k}" for k in V]; print(f"\n######## {nm} | {pre}"); print(sl.table(T[m], cols, D).round(4).iloc[1:, [0, 1, 6]].to_string())
for nm, m in [("recent", T.origin.isin(REC_)), ("all 13", np.ones(len(T), bool))]:
    for k in [k for k in V if not k.startswith("ec_ns")]:
        base = "ec_ns | " + k.split(" | ")[1]
        print(f"bootstrap S+corr {k:16s} vs {base:13s} ({nm}):", np.round(sl.boot(T[m], f"S+corr:{k}", f"S+corr:{base}"), 4))
