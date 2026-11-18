"""Fork 2 (user: "another model calibration should be per item, cluster, product type and such based on correlation, weight them together").
Per-item ensemble weights from each member's past skill, measured at four levels and shrunk together:
  skill   correlation between the member's log(forecast / R3) and the actual log(y / R3) (does it call the item's move away from its
          recent rate?), over past backtest rows of the same item / embedding cluster / product type / all items; inverse squared log
          error as an alternative skill.
  weights softmax(skill / tau) per level; then all -> product type -> cluster -> item, each step blending in with n / (n + kappa).
  time    at origin o only rows whose window has ended (origins <= o - 4) are used; the first origins fall back to equal weights.
          (Clusters are the June 2026 embedding's, a grouping only.)
Forecast = weighted arithmetic mean of the five submitted members x the submitted correction multiplier. Settings chosen on the tune
origins 2025-06 .. 2025-09, judged on the held-out 2025-10 .. 2026-02 and on all recent 9, against equal weights, with a style bootstrap.
Output -> eda/hier_weights_output.txt, data/hier_weights.json"""
import sys, json, itertools, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True)
K = pd.MultiIndex.from_arrays([D.origin, D.style_color])
q50 = lambda f: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(K).q50.values
nn_ec = pd.read_parquet("data/nn_backtest_preds_S13ecx6.parquet").groupby(["origin", "style_color"]).q50.sum().reindex(K).values
M = np.column_stack([D.rules.values, q50("data/q_backtest_nx_ec_ns.parquet"), np.where(np.isfinite(nn_ec), nn_ec, D.nn.values),
                     q50("data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet"), q50("data/q_backtest_fktab_tabpfn_v2_dyn_e32.parquet")])
NAMES = ["rules", "tree", "nn", "pf2c", "pf2d32"]; NM = len(NAMES)
g = pd.read_parquet("data/geo_ecnsnn_x_preds.parquet").drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(K)
cm = (g.geo / g.F.clip(lower=1e-9)).fillna(1).values
y, R3 = D.y.values, D.R3.values
cl = pd.read_parquet("data/clusters_2026-06.parquet").set_index("style_color").cluster
D["cluster"] = D.style_color.map(cl).fillna(-1).astype(int).astype(str)
LEV = {"item": D.style_color.values, "cluster": D.cluster.values, "ptype": D.ptype.astype(str).values, "all": np.zeros(len(D), int).astype(str)}
a = np.log((y + 1) / (R3 + 1)); B = np.log((np.maximum(M, 0) + 1) / (R3[:, None] + 1))
oi = pd.PeriodIndex(D.origin, freq="M").asi8; O = D.origin.values
def skills(Hm, lev):
    """per group of level lev, over history rows Hm: n, correlation and inverse squared log error of each member"""
    keys = LEV[lev][Hm]; df = pd.DataFrame({"k": keys, "a": a[Hm], "aa": a[Hm] ** 2})
    for j in range(NM): df[f"b{j}"] = B[Hm, j]; df[f"bb{j}"] = B[Hm, j] ** 2; df[f"ab{j}"] = a[Hm] * B[Hm, j]; df[f"e{j}"] = (B[Hm, j] - a[Hm]) ** 2
    S = df.groupby("k").agg(["sum"]); S.columns = [c[0] for c in S.columns]; n = df.groupby("k").size()
    corr = np.full((len(S), NM), np.nan); inv = np.full((len(S), NM), np.nan)
    va = S.aa - S.a ** 2 / n
    for j in range(NM):
        vb = S[f"bb{j}"] - S[f"b{j}"] ** 2 / n; cov = S[f"ab{j}"] - S.a * S[f"b{j}"] / n
        corr[:, j] = np.where((n >= 3) & (va > 1e-9) & (vb > 1e-9), cov / np.sqrt(np.maximum(va * vb, 1e-18)), np.nan)
        inv[:, j] = 1 / (S[f"e{j}"] / n + 0.01)
    return S.index.values, n.values, corr, inv
def weights_at(o_ord, tau, k_item, k_grp, kind):
    """weights for every row of origin o_ord from history with windows ended (origins <= o_ord - 4)"""
    rows = np.where(oi == o_ord)[0]; Hm = oi <= o_ord - 4
    W = np.full((len(rows), NM), 1 / NM)
    if Hm.sum() == 0: return rows, W
    for lev, kap in [("all", 0), ("ptype", k_grp), ("cluster", k_grp), ("item", k_item)]:
        keys, n, corr, inv = skills(Hm, lev)
        sk = corr if kind == "corr" else np.log(inv)
        z = sk / tau; z = z - np.nanmax(z, axis=1, keepdims=True); e = np.where(np.isfinite(z), np.exp(z), np.nan)
        wl = e / np.nansum(e, axis=1, keepdims=True); ok = np.isfinite(wl).all(axis=1)
        pos = pd.Series(np.arange(len(keys)), index=keys).reindex(LEV[lev][rows]).values
        has = np.isfinite(pos); p_ = np.where(has, pos, 0).astype(int); good = has & ok[p_]
        lam = np.zeros(len(rows)) if lev != "all" else np.ones(len(rows))
        if lev != "all": lam[good] = n[p_[good]] / (n[p_[good]] + kap)
        else: lam = np.where(good, 1.0, 0.0)
        W[good] = (1 - lam[good, None]) * W[good] + lam[good, None] * wl[p_[good]]
    return rows, W
def forecast(tau, k_item, k_grp, kind):
    F = np.empty(len(D)); Wall = np.empty((len(D), NM))
    for o in np.unique(oi):
        rows, W = weights_at(o, tau, k_item, k_grp, kind); F[rows] = (M[rows] * W).sum(1) * cm[rows]; Wall[rows] = W
    return F, Wall
wr3 = lambda f, m: float(np.abs(f - y)[m].sum() / np.abs(R3 - y)[m].sum())
base = M.mean(1) * cm
TUNE = (O >= "2025-06") & (O <= "2025-09"); HOLD = O >= "2025-10"; REC = O >= "2025-06"; ALL = np.ones(len(D), bool); EARLY = O <= "2025-05"
print("equal weights (submitted): tune", round(wr3(base, TUNE), 4), "held out", round(wr3(base, HOLD), 4), "recent 9", round(wr3(base, REC), 4))
grid = []
for kind, tau, ki, kg in itertools.product(["corr", "inverr"], [0.05, 0.1, 0.2, 0.4], [3, 8, 20], [30, 150]):
    if kind == "inverr" and tau < 0.1: continue
    F, _ = forecast(tau if kind == "corr" else tau * 10, ki, kg, kind)
    grid.append(dict(kind=kind, tau=tau, k_item=ki, k_grp=kg, tune=round(wr3(F, TUNE), 4), hold=round(wr3(F, HOLD), 4), rec=round(wr3(F, REC), 4)))
G = pd.DataFrame(grid).sort_values("tune"); print(G.head(12).to_string(index=False)); print("...", len(G), "settings")
rng = np.random.default_rng(0); sty = D["style"].values
def boot(fa, fb, m, B_=400):
    df = pd.DataFrame(dict(s=sty[m], a=np.abs(fa - y)[m], b=np.abs(fb - y)[m], r=np.abs(R3 - y)[m])).groupby("s").sum()
    idx = rng.integers(0, len(df), size=(B_, len(df))); d = (df.a.values[idx].sum(1) - df.b.values[idx].sum(1)) / df.r.values[idx].sum(1)
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
out = {}
for kind in ["corr", "inverr"]:
    b = G[G.kind == kind].iloc[0]; F, W = forecast(b.tau if kind == "corr" else b.tau * 10, int(b.k_item), int(b.k_grp), kind)
    out[kind] = dict(setting=dict(tau=float(b.tau), k_item=int(b.k_item), k_grp=int(b.k_grp)),
                     **{nm: dict(equal=round(wr3(base, m), 4), weighted=round(wr3(F, m), 4), ci=boot(F, base, m)) for nm, m in [("tune", TUNE), ("held out", HOLD), ("recent 9", REC), ("all 13", ALL)]},
                     mean_weights_held_out={n: round(float(W[HOLD, j].mean()), 3) for j, n in enumerate(NAMES)},
                     weight_spread_held_out=round(float(W[HOLD].max(1).mean()), 3))
    print(kind, json.dumps(out[kind]))
# item-level read: the 100 targets on the held-out origins
T = D.target.values.astype(bool) if "target" in D else np.zeros(len(D), bool)
Fc, _ = forecast(G[G.kind == "corr"].iloc[0].tau, int(G[G.kind == "corr"].iloc[0].k_item), int(G[G.kind == "corr"].iloc[0].k_grp), "corr")
print("100 targets, held out: equal", round(wr3(base, HOLD & T), 4), "corr-weighted", round(wr3(Fc, HOLD & T), 4))
json.dump(dict(grid=grid, best=out), open("data/hier_weights.json", "w"), indent=1)
