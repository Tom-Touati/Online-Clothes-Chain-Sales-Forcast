"""D358: the last-year-removed gate (fork D315 / #10a) for the recommender systems, on the same footing as D298 / D351 / D353: the tree
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
for tr in ["full", "noly"]:
    T[f"sub|{tr}"] = AM(*base(TREE[tr]["ec_ns"])) * mult
    T[f"hub|{tr}"] = AM(*base(TREE[tr]["huber"])) * mult
    T[f"rc|{tr}"] = AM(*base(TREE[tr]["ec_ns"]), RC[tr]) * mult
    T[f"hubrc|{tr}"] = AM(*base(TREE[tr]["huber"]), RC[tr]) * mult
    print(f"\n== training: {'normal' if tr == 'full' else 'last year window removed'} (system x the submitted corrections)")
    for c in ["sub", "hub", "rc", "hubrc"]:
        f = T[f"{c}|{tr}"]; e, e3 = (f - T.y).abs(), (T.R3 - T.y).abs()
        print(f"  {c:6s} recent {e[REC].sum() / e3[REC].sum():.4f} | all 13 {e.sum() / e3.sum():.4f} | early {e[EARLY].sum() / e3[EARLY].sum():.4f}" +
              ("" if c == "sub" else f" | boot vs sub: recent {np.round(sl.boot(T[REC], f'{c}|{tr}', f'sub|{tr}'), 4)} all 13 {np.round(sl.boot(T, f'{c}|{tr}', f'sub|{tr}'), 4)}"))
# D358b (Fork 2's suggestion): a smaller weight for rc_deepfm (w relative to one member), and rc_gbm as the sixth member
RCG = {"full": rd("data/rc_recsys_preds.parquet", "rc_gbm"), "noly": rd("data/rc_recsys_preds_noly.parquet", "rc_gbm")}
WM = lambda xs, ws: np.nansum(np.column_stack([np.clip(x, 0, None) * w for x, w in zip(xs, ws)]), axis=1) / sum(ws)
print("\n== weighted sixth member (huber tree), recent | all 13, normal / last year removed")
for lab, src, w in [("rc_deepfm w=0.5", RC, 0.5), ("rc_deepfm w=0.25", RC, 0.25), ("rc_gbm w=1", RCG, 1.0), ("rc_gbm w=0.5", RCG, 0.5)]:
    out = []
    for tr in ["full", "noly"]:
        c = f"{lab}|{tr}"; T[c] = WM(base(TREE[tr]["huber"]) + [src[tr]], [1, 1, 1, 1, 1, w]) * mult
        e, e3 = (T[c] - T.y).abs(), (T.R3 - T.y).abs(); out.append(f"{tr}: {e[REC].sum() / e3[REC].sum():.4f} | {e.sum() / e3.sum():.4f} (boot recent vs sub {np.round(sl.boot(T[REC], c, f'sub|{tr}'), 4)})")
    print(f"  {lab:18s} " + "  ||  ".join(out))
