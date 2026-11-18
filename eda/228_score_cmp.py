"""Fork 2 (user: campaign days and discounts per horizon month directly in the models): the routed tree pair of the submission (Huber for
items 6 months and older, Huber with new rows x5 for younger) refitted with the campaign features of eda/227 (variants A: calendar per month,
B: change against last year per month + exposure, C: exposure only), scored inside the submitted system (nx6 = new nx5 backtest: other members
as stored, rc at a quarter, the routed nx5 correction multiplier held fixed) against the current pair; and the last-year-removed gate on the
D358 / D389q footing (trees and rc refitted without o - 12 +/- 1, x the nx4 correction multiplier). Style bootstrap.
stdout -> eda/score_cmp_output.txt"""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
B = pd.read_parquet("data/dec_base_nx5r26.parquet"); B = B[B.y.notna()].reset_index(drop=True)
G = pd.read_parquet("data/geo_nx5r_x_preds.parquet")[["origin", "style_color", "F", "geo"]].rename(columns={"F": "Fg"}); B = B.merge(G, on=["origin", "style_color"])
B["corr"] = (B.geo / B.Fg.clip(lower=1e-9)).fillna(1); K = pd.MultiIndex.from_frame(B[["origin", "style_color"]])
rd = lambda f, c="q50": pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"])[c].reindex(K).values
new = B.age_m.fillna(99).values < 6; O = B.origin.values; y, R3 = B.y.values, B.R3.values; sty = B.style_color.str.split("_").str[0].values; rng = np.random.default_rng(0)
W = np.array([1, 1, 1, 1, 1, .25])
def sysf(tree, rc=None, corr=None):
    X = np.column_stack([B.rules, tree, B.nn, B.pf2c, B.pf2d32, B.rc if rc is None else rc]).clip(min=0); ok = np.isfinite(X)
    return np.nansum(np.where(ok, X, 0) * W, 1) / (ok * W).sum(1) * (B["corr"].values if corr is None else corr)
w = lambda f, m: np.abs(f - y)[m].sum() / np.abs(R3 - y)[m].sum()
def ci(a, b, m):
    d = pd.DataFrame(dict(s=sty[m], a=np.abs(a - y)[m], b=np.abs(b - y)[m], r=np.abs(R3 - y)[m])).groupby("s").sum()
    i = rng.integers(0, len(d), size=(400, len(d))); v = (d.a.values[i].sum(1) - d.b.values[i].sum(1)) / d.r.values[i].sum(1); return f"[{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}]"
route = lambda old, nw: np.where(new & np.isfinite(nw), nw, old)
cur = sysf(route(rd("data/q_backtest_ls_huber.parquet"), rd("data/q_backtest_ls_huber_new5.parquet")))
print("current pair reproduces the submission:", round(w(cur, O >= "2025-06"), 4), round(w(cur, O >= "2025-10"), 4))
SETS = [("recent 9", O >= "2025-06"), ("held out", O >= "2025-10"), ("all 13", O > "0"), ("early 4", O <= "2025-05"), ("Jul-Oct 25", O == "2025-06"), ("Mar-Jun 26", O == "2026-02"), ("new, recent", (O >= "2025-06") & new)]
# last-year-removed footing (D389q): nx4 correction multiplier, rc without last year
E = pd.read_parquet("data/fk_err_master.parquet").set_index(["origin", "style_color"]).reindex(K); mult4 = (E.F / E.F0).values
rc_noly = rd("data/rc_recsys_preds_noly.parquet", "rc_deepfm")
cur_ly = sysf(route(rd("data/srv4/q_backtest_ls_huber_noly.parquet"), rd("data/q_backtest_ls_huber_new5_lyd.parquet")), rc_noly, mult4)
for k in "ABC":
    old, nw = rd(f"data/q_backtest_ls_huber_cmp{k}.parquet"), rd(f"data/q_backtest_ls_huber_new5_cmp{k}.parquet")
    for lab, tree in [(f"cmp{k}: both trees", route(old, nw)), (f"cmp{k}: old-item tree only", route(old, rd("data/q_backtest_ls_huber_new5.parquet")))]:
        s = sysf(tree); print(f"\n== {lab}")
        for nm, m in SETS: print(f"   {nm:12s} {w(cur, m):.4f} -> {w(s, m):.4f}  {ci(s, cur, m)}")
    oly, nly = rd(f"data/q_backtest_ls_huber_cmp{k}_lyd.parquet"), rd(f"data/q_backtest_ls_huber_new5_cmp{k}_lyd.parquet")
    g = sysf(route(oly, nly), rc_noly, mult4)
    print(f"   last year removed: recent {w(cur_ly, O >= '2025-06'):.4f} -> {w(g, O >= '2025-06'):.4f} {ci(g, cur_ly, O >= '2025-06')} | all 13 {w(cur_ly, O > '0'):.4f} -> {w(g, O > '0'):.4f}")
