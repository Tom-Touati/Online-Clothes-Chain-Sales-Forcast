"""D388 (Fork 2): the same last-year gate for the business-plan tree twin (f_biz_gbm, data/rc_recsys_preds_x_biz.parquet; without last year:
data/rc_recsys_preds_x_biznoly_g.parquet, gbm only). Original D358 note: the last-year-removed gate (fork D315 / #10a) for the recommender systems, on the same footing as D298 / D351 / D353: the tree
member and the recommender refitted without the training origins o - 12 +/- 1 (data/srv2/q_backtest_nx_ec_ns_noly, data/srv4/
q_backtest_ls_huber_noly, Fork 2's data/rc_recsys_preds_noly.parquet rc_deepfm), the other members as stored (their no-LY refits are the
fork's GPU job), the system x the submitted correction multiplier. Normal training alongside. Bootstrap against the submitted system's
footing (ec_ns, 5 members). Writes eda/rc_gate_output.txt (stdout)."""
import sys, contextlib, io, importlib.util, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); REC = T.origin >= "2025-06"; EARLY = T.origin <= "2025-05"
rd = lambda f, c="q50": pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"])[c].reindex(key).values
B = pd.read_parquet("data/dec_base_ecnsnn.parquet").set_index(["origin", "style_color"]).reindex(key)
G = pd.read_parquet("data/geo_ecnsnn_x_preds.parquet").set_index(["origin", "style_color"]).reindex(key); mult = (G.geo / G.F.clip(lower=1e-9)).fillna(1).values
AM = lambda *xs: np.nanmean(np.column_stack([np.clip(x, 0, None) for x in xs]), axis=1)
TREE = {"full": {"ec_ns": rd("data/srv2/q_backtest_nx_ec_ns.parquet"), "huber": rd("data/srv4/q_backtest_ls_huber.parquet")},
        "noly": {"ec_ns": rd("data/srv2/q_backtest_nx_ec_ns_noly.parquet"), "huber": rd("data/srv4/q_backtest_ls_huber_noly.parquet")}}
RC = {"full": rd("data/rc_recsys_preds.parquet", "rc_deepfm"), "noly": rd("data/rc_recsys_preds_noly.parquet", "rc_deepfm")}
base = lambda tree: [B.rules.values, tree, B.nn.values, B.pf2c.values, B.pf2d32.values]
WM = lambda xs, ws: np.nansum(np.column_stack([np.clip(x, 0, None) * w for x, w in zip(xs, ws)]), axis=1) / sum(ws)
BZ = {"full": rd("data/rc_recsys_preds_x_biz.parquet", "f_biz_gbm"), "noly": rd("data/rc_recsys_preds_x_biznoly_g.parquet", "f_biz_gbm")}
RCG = {"full": rd("data/rc_recsys_preds.parquet", "rc_gbm"), "noly": rd("data/rc_recsys_preds_noly.parquet", "rc_gbm")}
for tr in ["full", "noly"]:
    T[f"sub|{tr}"] = AM(*base(TREE[tr]["ec_ns"])) * mult; T[f"cand|{tr}"] = WM(base(TREE[tr]["huber"]) + [RC[tr]], [1, 1, 1, 1, 1, 0.25]) * mult
e3 = (T.R3 - T.y).abs()
for tr in ["full", "noly"]:
    for k, src in [("biz", BZ), ("rc_gbm", RCG)]:
        f = src[tr]; e = np.abs(np.nan_to_num(f, nan=0) - T.y); print(f"{tr:5s} {k} alone recent {e[REC].sum() / e3[REC].sum():.4f} all {e.sum() / e3.sum():.4f}")
print("Huber tree + sixth member at weight w (system x submitted corrections): recent | all 13 | early, normal || last year removed")
for lab, src, w in [("candidate D358 (rc_deepfm w=0.25)", RC, 0.25), ("biz w=0.25", BZ, 0.25), ("biz w=0.5", BZ, 0.5), ("biz w=1", BZ, 1.0)]:
    out = []
    for tr in ["full", "noly"]:
        c = f"{lab}|{tr}"; T[c] = WM(base(TREE[tr]["huber"]) + [src[tr]], [1, 1, 1, 1, 1, w]) * mult; e = (T[c] - T.y).abs()
        out.append(f"{e[REC].sum() / e3[REC].sum():.4f} | {e.sum() / e3.sum():.4f} | {e[EARLY].sum() / e3[EARLY].sum():.4f} (boot recent vs cand {np.round(sl.boot(T[REC], c, f'cand|{tr}'), 4)})")
    print(f"  {lab:34s} " + "  ||  ".join(out))
