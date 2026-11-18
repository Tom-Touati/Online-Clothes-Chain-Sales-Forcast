"""Fork 2 (overnight goal): build a routed nx5 base from the tree-search finals (eda/220 on the CPU server) and score it after its corrections.
Step 1 (BUILD=1): data/dec_base_<TAG>26.parquet = nx5's base (data/dec_base_ecnsnnhubrq26.parquet) with the tree column taken from
  NEW=<cand in data/hpo_tree_new_finals.parquet> for items under 6 months at the origin and OLD=<cand in data/hpo_tree_old_finals.parquet>
  for the rest (backtest origins and 2026-06), F = (five + 0.25 rc) / 5.25. Then the corrections run on the server (DEC=_<TAG>26).
Step 2 (SCORE=1): the geo correction (as eda/218c) on the routed base; WAPE / R3 recent / held out / all 13 / early / Jul-Oct 25 / Mar-Jun 26 /
  new items / targets against the submitted new nx5 (data/geo_nx5r_x_preds.parquet), style-bootstrap intervals; writes data/geo_<TAG>_x_preds.parquet
  and data/geo_<TAG>_x_2026.parquet. Usage: TAG=nx6a NEW="trial 32" OLD="trial 61" BUILD=1 python eda/223_combine_routed.py"""
import os, sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
TAG = os.environ["TAG"]
if os.environ.get("BUILD") == "1":
    B = pd.read_parquet("data/dec_base_ecnsnnhubrq26.parquet"); K = pd.MultiIndex.from_frame(B[["origin", "style_color"]])
    def cand(role, nm):
        if nm in ("", "current"): f = {"new": ("data/q_backtest_ls_huber_new5.parquet", "data/q_backtest_ls_huber_new5_2026.parquet"), "old": ("data/q_backtest_ls_huber.parquet", "data/q_backtest_ls_huber_2026.parquet")}[role]; Q = pd.concat([pd.read_parquet(f[0]), pd.read_parquet(f[1]).assign(origin="2026-06")])
        else: Q = pd.read_parquet(f"data/hpo_tree_{role}_finals.parquet"); Q = Q[Q.cand == nm]
        return Q.drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).q50.reindex(K).values
    qn, qo = cand("new", os.environ.get("NEW", "current")), cand("old", os.environ.get("OLD", "current"))
    new = B.age_m.fillna(99).values < 6
    t = np.where(new, qn, qo); print("tree rows missing:", int(np.isnan(t).sum())); B["qdyn"] = np.where(np.isfinite(t), t, B.qdyn.values)
    MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32", "rc"]; W = np.array([1, 1, 1, 1, 1, 0.25])
    B["F"] = (B[MEM].clip(lower=0).values * W).sum(1) / W.sum(); B["spread"] = np.log(B[MEM].clip(lower=1).max(axis=1) / B[MEM].clip(lower=1).min(axis=1))
    bt = B.y.notna(); B.loc[bt, "e"] = np.log((B.y[bt] + 1) / (B.F[bt] + 1)); B.loc[bt, "ae"] = (B.F[bt] - B.y[bt]).abs()
    B.to_parquet(f"data/dec_base_{TAG}26.parquet", index=False); print("wrote", f"data/dec_base_{TAG}26.parquet")
if os.environ.get("SCORE") == "1":
    def geo(base, seg, brk):
        B = pd.read_parquet(base)[["origin", "style_color", "F", "y", "R3", "age_m", "target", "style"]]
        S = pd.read_parquet(seg)[["origin", "style_color", "vol_age_disc_0.5n"]]; K = pd.read_parquet(brk)[["origin", "style_color", "res_all_0.5n"]]
        B = B.merge(S, on=["origin", "style_color"]).merge(K, on=["origin", "style_color"]); B["g"] = np.sqrt(B["vol_age_disc_0.5n"].clip(lower=0) * B["res_all_0.5n"].clip(lower=0))
        B["geo"] = B.g * B.groupby("origin").F.transform("sum") / B.groupby("origin").g.transform("sum"); return B
    G = geo(f"data/dec_base_{TAG}26.parquet", f"data/bias_seg_preds_{TAG}26.parquet", f"data/breakout_preds_{TAG}26_x.parquet")
    bt = G[G.y.notna()].copy(); ref = pd.read_parquet("data/geo_nx5r_x_preds.parquet")[["origin", "style_color", "geo"]].rename(columns={"geo": "ref"})
    bt = bt.merge(ref, on=["origin", "style_color"]); O = bt.origin.values; y, R3 = bt.y.values, bt.R3.values; rng = np.random.default_rng(0)
    w = lambda f, m: np.abs(f - y)[m].sum() / np.abs(R3 - y)[m].sum()
    def ci(m, B_=400):
        df = pd.DataFrame(dict(s=bt["style"].values[m], a=np.abs(bt.geo.values - y)[m], b=np.abs(bt.ref.values - y)[m], r=np.abs(R3 - y)[m])).groupby("s").sum()
        i = rng.integers(0, len(df), size=(B_, len(df))); d = (df.a.values[i].sum(1) - df.b.values[i].sum(1)) / df.r.values[i].sum(1); return f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
    new = bt.age_m.fillna(99).values < 6
    for lab, m in [("recent 9", O >= "2025-06"), ("held out", O >= "2025-10"), ("all 13", O > "0"), ("early 4", O <= "2025-05"), ("Jul-Oct 25", O == "2025-06"), ("Mar-Jun 26", O == "2026-02"),
                   ("new items, recent", (O >= "2025-06") & new), ("old items, recent", (O >= "2025-06") & ~new), ("targets, recent", (O >= "2025-06") & bt.target.astype(bool).values)]:
        print(f"{TAG} {lab:20s} nx5r {w(bt.ref.values, m):.4f} -> {w(bt.geo.values, m):.4f}  {ci(m)}")
    bt.drop(columns=["ref"]).to_parquet(f"data/geo_{TAG}_x_preds.parquet", index=False)
    g26 = G[G.origin == "2026-06"][["style_color", "F", "geo"]]; g26.to_parquet(f"data/geo_{TAG}_x_2026.parquet", index=False)
    print("2026 F on all items", round(g26.F.sum()), "| multipliers 10/50/90%:", np.round(np.percentile((g26.geo / g26.F).dropna(), [10, 50, 90]), 3))
