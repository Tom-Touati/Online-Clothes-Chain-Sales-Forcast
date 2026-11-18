"""Learned item embeddings as extra features for the LightGBM ensemble, same 13 rolling origins, everything from data <= origin.
mode=svd : seasonal-shape embedding. Weekly log units per item, minus the item's own mean over its live weeks, low-rank SVD
           over items with >= 26 live weeks; every item (also young ones) is projected onto the components using only the
           weeks it was live (ridge least squares). k components -> features e0..e{k-1}.
mode=nn  : hierarchical entity embedding (macro + type + style + item vector of the eda/57 NN, trained on the same rows),
           reduced to k principal components.
Reference = ens (L1 base + L1 full) of eda/12_round3.py, recomputed here; candidates replace 'full' by 'full + embedding'.
Writes data/emb_{mode}_backtest_preds.parquet."""
import sys, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
ARGS = dict(a.split("=") for a in sys.argv[1:]); MODE = ARGS.get("mode", "svd"); K = int(ARGS.get("k", 6))
F = pd.read_parquet("data/features.parquet")
for c in ["ptype", "macro"] + [c for c in F.columns if c.startswith("a_")]: F[c] = F[c].astype("category")
F["rlog"] = np.log((F.y + 1) / (F.level + 1))
OCC = [c for c in F.columns if c.startswith("occ_")]
BASE = [f"lagr{k}" for k in range(1, 7)] + ["lvl", "age_m", "ptype", "tmonth", "h", "color_num", "own_sea"] + OCC
FULL = BASE + ["f_r7", "f_r28", "trend91", "peak_r", "days_live", "sh_ret", "sh_drop", "sh_tt", "disc_item", "disc_rel", "lfl_type", "lfl_macro", "lfl_occ"] + [c for c in F.columns if c.startswith("a_")]
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=6)
def l1(tr, te, cols):
    m = lgb.LGBMRegressor(objective="l1", **P).fit(tr[cols], tr.rlog); return ((te.level + 1) * np.exp(m.predict(te[cols])) - 1).clip(lower=0)
EC = [f"e{j}" for j in range(K)]

if MODE == "svd":
    d = pd.read_parquet("data/daily_item_chan.parquet", columns=["style_color", "ny_date", "cancelled", "qty"]); d = d[~d.cancelled]
    d["wk"] = pd.to_datetime(d.ny_date).dt.to_period("W-SUN")
    W = d.groupby(["style_color", "wk"]).qty.sum().unstack(fill_value=0); del d
    it = pd.read_parquet("data/items.parquet").set_index("style_color")
    start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
    W = W.reindex(it.index, fill_value=0)
    MEMO = {}
    def embed(o):
        if o in MEMO: return MEMO[o]
        end = pd.Period(o, "M").end_time; cols = [c for c in W.columns if c.end_time <= end]
        X = np.log1p(W[cols].values.astype(float))
        live = np.array([[c.start_time >= s - pd.Timedelta(days=6) for c in cols] for s in start.reindex(W.index).values])
        mu = np.where(live, X, np.nan); mu = np.nanmean(mu, 1, keepdims=True); Z = np.where(live, X - mu, 0.0)
        full = live.sum(1) >= min(26, max(6, int(0.8 * len(cols))))
        U, S, Vt = np.linalg.svd(Z[full], full_matrices=False); V = Vt[:K].T * 1.0          # weeks x K
        if V.shape[1] < K: V = np.hstack([V, np.zeros((V.shape[0], K - V.shape[1]))])
        E = np.zeros((len(W), K))
        for i in range(len(W)):
            m = live[i]
            if m.sum() < 4: continue
            A = V[m]; E[i] = np.linalg.solve(A.T @ A + 1.0 * np.eye(K), A.T @ Z[i, m])
        MEMO[o] = pd.DataFrame(E, index=W.index, columns=EC); return MEMO[o]
else:
    sys.argv = ["x", "inter=0", "seeds=1"]
    import importlib.util; spec = importlib.util.spec_from_file_location("nnm", "eda/57_embed_nn.py"); nnm = importlib.util.module_from_spec(spec); spec.loader.exec_module(nnm)
    nnm.RET_NET = True
    import torch
    def embed(o, tr):
        _, net = nnm.fit_predict(nnm.F[nnm.F.index.isin(tr.index)], nnm.F[nnm.F.origin == o].head(5), 0)
        items = nnm.F.drop_duplicates("style_color").set_index("style_color")
        with torch.no_grad():
            xc = torch.tensor(items[[c + "_i" for c in nnm.CATS]].values).to(nnm.DEV)
            v = sum(net.E[c](xc[:, nnm.CATS.index(c)]) for c in nnm.HIER).cpu().numpy()
        v = v - v.mean(0); U, S, Vt = np.linalg.svd(v, full_matrices=False)
        return pd.DataFrame(v @ Vt[:K].T, index=items.index, columns=EC)

ORIGINS = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
out = []
for o in ORIGINS:
    oo = pd.Period(o).ordinal; tr = F[(F.o_ord + F.h <= oo) & F.y.notna()].copy(); te = F[F.origin == o].copy()
    pb, pf = l1(tr, te, BASE), l1(tr, te, FULL)
    if MODE == "svd":
        # training rows get the embedding as it was at their own origin (no look-ahead inside the training set)
        cache = {oc: embed(oc) for oc in sorted(set(tr.origin) | {o})}
        for D in (tr, te):
            E = pd.concat([cache[oc].reindex(g.style_color).set_index(g.index) for oc, g in D.groupby("origin")])
            D[EC] = E.reindex(D.index).values
    else:
        E = embed(o, tr)
        for D in (tr, te): D[EC] = E.reindex(D.style_color).values
    pe = l1(tr, te, FULL + EC)
    te["ens"] = (pb + pf) / 2; te["full"] = pf; te["full_emb"] = pe; te["ens_emb"] = (pb + pe) / 2; te["ens3"] = (pb + pf + pe) / 3
    out.append(te[["origin", "style_color", "h", "ens", "full", "full_emb", "ens_emb", "ens3", "y"]])
    g = te.groupby("style_color")[["ens", "ens_emb", "ens3", "y"]].sum()
    print(o, {c: round((g[c] - g.y).abs().sum() / g.y.sum(), 3) for c in ["ens", "ens_emb", "ens3"]}, flush=True)
X = pd.concat(out); X.to_parquet(f"data/emb_{MODE}_backtest_preds.parquet")
T = set(pd.read_csv("data/targets.csv").style_color)
for seg, m in [("all", np.ones(len(X), bool)), ("targets", X.style_color.isin(T).values)]:
    R = X[m].groupby("origin").apply(lambda g: pd.Series({c: (lambda t: (t[c] - t.y).abs().sum() / t.y.sum())(g.groupby("style_color")[[c, "y"]].sum()) for c in ["ens", "full", "full_emb", "ens_emb", "ens3"]}))
    print(f"== {seg}: mean WAPE", R.mean().round(4).to_dict(), "| ens_emb beats ens at", int((R.ens_emb < R.ens).sum()), "| ens3 beats ens at", int((R.ens3 < R.ens).sum()), "of", len(R))
