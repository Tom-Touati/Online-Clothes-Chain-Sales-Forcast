"""D276 (fork): is the weak early backtest a matter of season or of training size? The tree member alone (LightGBM median, the 'ecns'
feature set of data/fk_tab_ec_table.parquet, D198 parameters), time-honest pool = origins whose outcome is known at the test origin.
At every test origin: (A) the full pool; (B) random subsets of k = 4, 8, 12, 16 pool origins (3 draws each); (C) the pool without the
origins that forecast the same calendar window a year earlier (test origin - 12 months, +/- 1); (D) only the most recent 6 / 12 pool
origins. Scored as WAPE / R3 of the member alone on that origin. Writes data/fk_learning_curve.json and stdout."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
T = pd.read_parquet("data/fk_tab_ec_table.parquet"); COLS = json.load(open("data/fk_tab_ec_meta.json"))["ecns"]; P = dict(json.load(open("data/fk_explain_meta.json"))["PARAMS"]); P["n_jobs"] = 6
ORIG = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]; rng = np.random.default_rng(0); rows = []
def score(tr, te):
    z = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **P).fit(tr[COLS], tr.z).predict(te[COLS]); f = np.clip((4 * te.level.values + 1) * np.exp(z) - 1, 0, None)
    return float(np.abs(f - te.ytot.values).sum() / np.abs(te.R3.values - te.ytot.values).sum()), float(f.sum() / te.ytot.sum() - 1)
for o in ORIG:
    oo = pd.Period(o).ordinal; pool = sorted(T[(T.o_ord + 4 <= oo) & T.ytot.notna()].o_ord.unique()); te = T[T.origin == o]; trf = T[T.o_ord.isin(pool) & T.ytot.notna()]
    w, b = score(trf, te); rows.append(dict(origin=o, kind="full", k=len(pool), ratio=w, bias=b)); print(o, "pool", len(pool), "full", round(w, 4), flush=True)
    for k in [4, 8, 12, 16]:
        if k >= len(pool): continue
        for d in range(3):
            sub = rng.choice(pool, k, replace=False); w, b = score(T[T.o_ord.isin(sub) & T.ytot.notna()], te); rows.append(dict(origin=o, kind="random", k=k, ratio=w, bias=b))
    same = [x for x in pool if abs(x - (oo - 12)) <= 1]
    if same and len(pool) - len(same) >= 4:
        w, b = score(T[T.o_ord.isin([x for x in pool if x not in same]) & T.ytot.notna()], te); rows.append(dict(origin=o, kind="no_same_season", k=len(pool) - len(same), ratio=w, bias=b))
        ctrl = [rng.choice([x for x in pool if x not in same], len(same), replace=False) for _ in range(3)]   # control: drop as many other origins
        cw = [score(T[T.o_ord.isin([x for x in pool if x not in c]) & T.ytot.notna()], te)[0] for c in ctrl]; rows.append(dict(origin=o, kind="drop_other", k=len(pool) - len(same), ratio=float(np.mean(cw)), bias=np.nan))
    for k in [6, 12]:
        if k < len(pool): w, b = score(T[T.o_ord.isin(pool[-k:]) & T.ytot.notna()], te); rows.append(dict(origin=o, kind="recent", k=k, ratio=w, bias=b))
R = pd.DataFrame(rows); full = R[R.kind == "full"].set_index("origin").ratio
R["vs_full"] = R.ratio - R.origin.map(full); out = dict(rows=R.round(4).to_dict("records"))
s = R.groupby(["kind", "k"]).agg(n_origins=("origin", "nunique"), ratio=("ratio", "mean"), vs_full=("vs_full", "mean")).round(4).reset_index(); print(s.to_string(index=False)); out["summary"] = s.to_dict("records")
a = R[R.kind == "no_same_season"].set_index("origin").vs_full; c = R[R.kind == "drop_other"].set_index("origin").vs_full
print("\nsame-season origins removed vs as many other origins removed, by test origin:"); print(pd.DataFrame({"no_same_season": a, "drop_other": c}).round(4).to_string())
out["season"] = [dict(origin=o, no_same=round(float(a[o]), 4), other=round(float(c[o]), 4)) for o in a.index]
json.dump(out, open("data/fk_learning_curve.json", "w")); print("saved")
