"""Fork 2 (user: "retrain with 5 times the loss on new data (has less than 6 months, wasnt there from the start)"): the tree member of
the submitted system swapped for eda/187 variants trained with training rows of items under 6 months old weighted x5 (x3), on log L1
(the submitted ec_ns loss) and Huber. System = mean of five members x the submitted correction multiplier. Scores WAPE / R3 and bias,
all items and new items (under 6 months at the origin), with a style bootstrap against the submitted system."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); K = pd.MultiIndex.from_arrays([D.origin, D.style_color])
q = lambda f: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(K).q50.values
y, R3, O, age, sty = D.y.values, D.R3.values, D.origin.values, D.age_m.values, D["style"].values
corr = (D.F / D.F0).values; others = D[["rules", "nn", "pf2c", "pf2d32"]].values.sum(1)
sysf = lambda tree: (others + tree) / 5 * corr
NEW = np.nan_to_num(age, nan=99) < 6
S = {"rec": O >= "2025-06", "all": np.ones(len(D), bool), "hold": O >= "2025-10", "early": O <= "2025-05"}
w = lambda f, m: np.abs(f - y)[m].sum() / np.abs(R3 - y)[m].sum(); b = lambda f, m: f[m].sum() / y[m].sum() - 1
rng = np.random.default_rng(0)
def boot(fa, fb, m, B=400):
    df = pd.DataFrame(dict(s=sty[m], a=np.abs(fa - y)[m], b=np.abs(fb - y)[m], r=np.abs(R3 - y)[m])).groupby("s").sum()
    i = rng.integers(0, len(df), size=(B, len(df))); d = (df.a.values[i].sum(1) - df.b.values[i].sum(1)) / df.r.values[i].sum(1)
    return f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
base = D.F.values
print("new items (under 6 months): share of rows", round(NEW.mean(), 3), "| share of recent units", round(y[NEW & S["rec"]].sum() / y[S["rec"]].sum(), 3))
rows = []
for nm, f in [("submitted (ec_ns, log L1)", "data/q_backtest_nx_ec_ns.parquet"), ("log L1 rerun", "data/q_backtest_ls_l1log.parquet"), ("log L1, new x5", "data/q_backtest_ls_l1log_new5.parquet"),
              ("Huber", "data/q_backtest_ls_huber.parquet"), ("Huber, new x5", "data/q_backtest_ls_huber_new5.parquet"), ("Huber, new x3", "data/q_backtest_ls_huber_new3.parquet")]:
    t = q(f); s = sysf(t); r = dict(variant=nm)
    for k, m in S.items(): r[k] = round(w(s, m), 4)
    r["new rec"] = round(w(s, S["rec"] & NEW), 4); r["new bias"] = f"{b(s, S['rec'] & NEW):+.3f}"; r["old rec"] = round(w(s, S["rec"] & ~NEW), 4)
    r["tree alone new rec"] = round(w(t, S["rec"] & NEW), 4); r["tree new bias"] = f"{b(t, S['rec'] & NEW):+.3f}"
    r["ci rec"] = boot(s, base, S["rec"]); r["ci new rec"] = boot(s, base, S["rec"] & NEW); rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
# routed: the x5 tree only for items under 6 months at the origin, the submitted tree for the rest (after the first read)
print("\nrouted (x5 tree for new items only):")
for nm, f in [("log L1, new x5", "data/q_backtest_ls_l1log_new5.parquet"), ("Huber, new x5", "data/q_backtest_ls_huber_new5.parquet")]:
    s = np.where(NEW, sysf(q(f)), base)
    print(nm, {k: round(w(s, m), 4) for k, m in S.items()}, "| ci rec", boot(s, base, S["rec"]), "ci hold", boot(s, base, S["hold"]), "ci all", boot(s, base, S["all"]), "ci early", boot(s, base, S["early"]))
# last-year-removed gate: both trees refit without training origins o - 12 +/- 1 (LYDROP=1, OSUF=_lyd); the other members as stored
print("\nlast-year-removed gate (tree refit without last year's window; routed = x5 tree for new items only):")
t0, t5 = q("data/q_backtest_ls_l1log_lyd.parquet"), q("data/q_backtest_ls_l1log_new5_lyd.parquet")
g0, g5 = sysf(t0), np.where(NEW, sysf(t5), sysf(t0))
for k in ["rec", "all", "hold"]: print(k, "ec_ns no-LY", round(w(g0, S[k]), 4), "routed no-LY", round(w(g5, S[k]), 4), boot(g5, g0, S[k]))
print("tree alone no-LY, recent: ec_ns", round(w(t0, S["rec"]), 4), "x5 routed", round(w(np.where(NEW, t5, t0), S["rec"]), 4))
