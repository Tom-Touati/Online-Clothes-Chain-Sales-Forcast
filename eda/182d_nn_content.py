"""D205d (fork): the 5 nearest neighbours in Main's v4 content style embedding (D166/D168, data/style_emb_v4.npz; passed the cart gate
on held-out styles; each origin's model trained on carts and sales up to that origin). Same neighbour summaries as the NN block of
eda/182b, computed in the content space: other styles alive in the last 3 months, top 5 by cosine.
  nc_surprise  neighbours' realised surprise (actual o-3 .. o over R3 at o-4)    nc_acc1  their newest-month turn
  nc_mom3 / nc_yoy / nc_sea4   their 3-month momentum, year-on-year growth and last-year ramp into the target months
  nc_lvl_rel   the item's log level minus theirs        nc_sim / nc_age   mean cosine, mean age (months)
Writes data/fk_nnc_feats.parquet and adds "NC" to data/fk_fash_meta.json."""
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
it = pd.read_parquet("data/items.parquet").set_index("style_color"); items = np.sort(it.index.values); n = len(items)
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; M = M.reindex(items).fillna(0); COLS = list(M.columns)
E = np.load("data/style_emb_v4.npz", allow_pickle=True); assert (E["items"].astype(str) == items).all()
sty = pd.Series(items).str.split("_").str[0].values; first = pd.to_datetime(it.reindex(items).first_sale).dt.to_period("M").astype(str).values
Pm = lambda o, k: str(pd.Period(o, "M") + k)
win = lambda o, ks: M.reindex(columns=[Pm(o, k) for k in ks]).fillna(0).values.sum(1)
ok = lambda o, k: Pm(o, k) in COLS
rows = []
for o in [str(p) for p in pd.period_range("2024-03", "2026-06", freq="M")]:
    if o not in E.files: continue
    l3, p3, y3, yt, m0, m3 = win(o, [-2, -1, 0]), win(o, [-5, -4, -3]), win(o, [-14, -13, -12]), win(o, [-11, -10, -9, -8]), win(o, [0]), win(o, [-3, -2, -1]) / 3
    r3p = win(o, [-6, -5, -4]) * 4 / 3; act = win(o, [-3, -2, -1, 0]); okr = ok(o, -6); oky = ok(o, -14)
    surpr = np.where((r3p > 0) & okr, np.log((act + 1) / (r3p + 1)), np.nan)
    mom = np.log((l3 + 1) / (p3 + 1)) if ok(o, -5) else np.full(n, np.nan); yoy = np.where((y3 > 0) & oky, np.log((l3 + 1) / (y3 + 1)), np.nan)
    acc = np.log((m0 + 1) / (m3 + 1)); sea = np.where((y3 > 0) & oky, np.log((yt + 1) / (y3 * 4 / 3 + 1)), np.nan); lvl = np.log1p(l3 / 3)
    age = np.array([(pd.Period(o, "M") - pd.Period(x, "M")).n if x <= o else np.nan for x in first], float)
    V = E[o].astype(np.float32); V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    S = V @ V.T; S[:, l3 <= 0] = -9; S[sty[:, None] == sty[None, :]] = -9
    nb = np.argsort(-S, 1)[:, :5]; sim = np.take_along_axis(S, nb, 1); okn = sim > -1
    with np.errstate(all="ignore"):
        nm = lambda x: np.nanmean(np.where(okn, x[nb], np.nan), 1)
        rows.append(pd.DataFrame({"origin": o, "style_color": items, "nc_surprise": nm(surpr), "nc_acc1": nm(acc), "nc_mom3": nm(mom), "nc_yoy": nm(yoy),
                                  "nc_sea4": nm(sea), "nc_lvl_rel": lvl - nm(lvl), "nc_sim": np.where(okn, sim, np.nan).mean(1), "nc_age": nm(age)}))
X = pd.concat(rows, ignore_index=True); X.to_parquet("data/fk_nnc_feats.parquet", index=False)
NC = [c for c in X.columns if c.startswith("nc_")]; print(X.shape, X.origin.nunique(), "origins")
print(X[X.origin >= "2025-02"][NC].describe().T[["count", "mean", "std"]].round(3).to_string())
meta = json.load(open("data/fk_fash_meta.json")); meta["NC"] = NC; json.dump(meta, open("data/fk_fash_meta.json", "w"), indent=1)
# residual diagnostic against the D198 model (as eda/fash_resid_diag.txt): early origins for selection, recent for reference
T = pd.read_parquet("data/fk_explain_table.parquet")[["origin", "style_color", "ytot"]]; q = pd.read_parquet("data/q_backtest_fkdyn_dyn.parquet")[["origin", "style_color", "q50"]]
D = T.merge(q, on=["origin", "style_color"]).merge(X, on=["origin", "style_color"], how="left"); D = D[D.ytot.notna()]; D["res"] = np.log((D.ytot + 1) / (D.q50 + 1))
for nm_, m in [("early (2025-02..05)", D.origin < "2025-06"), ("recent", D.origin >= "2025-06")]:
    r = {f: D[m].groupby("origin").apply(lambda g: g[[f, "res"]].corr("spearman").iloc[0, 1], include_groups=False) for f in NC}
    print(nm_, {f: (round(v.mean(), 3), round((np.sign(v) == np.sign(v.mean())).mean(), 2)) for f, v in r.items()})
