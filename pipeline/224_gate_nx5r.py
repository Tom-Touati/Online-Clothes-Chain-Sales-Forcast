"""Fork 2 (D389i follow-up): the last-year-removed gate on the combination itself, the new nx5 (routed tree + rc at a quarter), on the D358
footing (archive/eda/190): the tree members and the recommender refitted without training origins o - 12 +/- 1 (huber: data/srv4/q_backtest_ls_huber_noly;
huber_new5: data/q_backtest_ls_huber_new5_lyd, server; rc: data/rc_recsys_preds_noly), the other members as stored, x the submitted nx4 correction
multiplier (as D358). nx5 (D361) vs the routed nx5, normal and last year removed, recent / all 13 / held out, style bootstrap."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); K = pd.MultiIndex.from_arrays([D.origin, D.style_color])
rd = lambda f, c="q50": pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"])[c].reindex(K).values
mult = (D.F / D.F0).values; y, R3, O = D.y.values, D.R3.values, D.origin.values; new = D.age_m.fillna(99).values < 6
TREE = {"full": (rd("data/srv4/q_backtest_ls_huber.parquet"), rd("data/q_backtest_ls_huber_new5.parquet")), "noly": (rd("data/srv4/q_backtest_ls_huber_noly.parquet"), rd("data/q_backtest_ls_huber_new5_lyd.parquet"))}
RC = {"full": rd("data/rc_recsys_preds.parquet", "rc_deepfm"), "noly": rd("data/rc_recsys_preds_noly.parquet", "rc_deepfm")}
def sysf(tree, rc):
    X = np.column_stack([D.rules, tree, D.nn, D.pf2c, D.pf2d32, rc]).clip(min=0); W = np.array([1, 1, 1, 1, 1, .25]); ok = np.isfinite(X)
    return np.nansum(np.where(ok, X, 0) * W, 1) / (ok * W).sum(1) * mult
w = lambda f, m: np.abs(f - y)[m].sum() / np.abs(R3 - y)[m].sum(); rng = np.random.default_rng(0); sty = D["style"].values
def ci(a, b, m):
    d = pd.DataFrame(dict(s=sty[m], a=np.abs(a - y)[m], b=np.abs(b - y)[m], r=np.abs(R3 - y)[m])).groupby("s").sum()
    i = rng.integers(0, len(d), size=(400, len(d))); v = (d.a.values[i].sum(1) - d.b.values[i].sum(1)) / d.r.values[i].sum(1); return f"[{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}]"
for tr in ["full", "noly"]:
    h, h5 = TREE[tr]; a = sysf(h, RC[tr]); b = sysf(np.where(new & np.isfinite(h5), h5, h), RC[tr])
    print(f"== {'normal' if tr == 'full' else 'last year removed'}")
    for lab, m in [("recent 9", O >= "2025-06"), ("all 13", O > "0"), ("held out", O >= "2025-10")]: print(f"   {lab:9s} nx5 {w(a, m):.4f} routed {w(b, m):.4f} {ci(b, a, m)}")
