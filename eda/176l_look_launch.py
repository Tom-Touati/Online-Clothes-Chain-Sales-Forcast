"""D190b-style check (Main D189b; user: group shocks + fashion-feature fill): do launches in the same "look" explain the shared
window error found in D189? For each item at origin o: the new items launched in the window (first sale in o+1 .. o+4, which the
business knows from its launch plan; here the realised launches stand in for the plan) are placed by content similarity (CLIP image +
text, D171; new items have no carts yet). Features:
  la_k     number of window launches among the item's 20 nearest content neighbours (from all items, launches included)
  la_sim   summed positive similarity to window launches with similarity >= the item's 20th-neighbour similarity
  la_sty   a new colour of the same style launches in the window (D144 halo)
Spearman with p2's log residual e (recent origins, items with y >= 20), raw and partial on la_sty; and the residual by la_k band.
A time-honest correction p2 x exp(b1 la_k + b2 la_sty) (b fitted on past origins) shows what the launch plan would be worth."""
import sys, contextlib, io, importlib.util, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); G = lambda *xs: np.exp(np.mean([np.log(np.maximum(x, 0) + 0.5) for x in xs], axis=0)) - 0.5
qv = lambda f: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(key).q50.values
T["p2"] = G(T.rules.values, qv("data/q_backtest_fkdyn_dyn.parquet"), T.nn.values, qv("data/q_backtest_fktab_tabpfn_v2.parquet")); T["e"] = np.log((T.y + 1) / (T.p2 + 1))
items = np.sort(pd.read_parquet("data/items.parquet").style_color.values); n = len(items); sty = pd.Series(items).str.split("_").str[0].values; ix = {s: i for i, s in enumerate(items)}
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; M = M.reindex(items).fillna(0)
first = M.gt(0).idxmax(axis=1).where(M.gt(0).any(axis=1)).values
CE = np.load("data/content_emb.npz", allow_pickle=True); cix = pd.Series(range(len(CE["items"])), index=CE["items"].astype(str)).reindex(items).values
CON = np.hstack([CE["img"][cix], CE["txt"][cix]]); has = np.linalg.norm(CON, axis=1) > 0; CON = CON / np.maximum(np.linalg.norm(CON, axis=1, keepdims=True), 1e-9); SIM = CON @ CON.T
Pm = lambda o, k: str(pd.Period(o, "M") + k); rows = []
for o in sorted(T.origin.unique()):
    launch = np.array([isinstance(f, str) and Pm(o, 1) <= f <= Pm(o, 4) for f in first]) & has
    X = SIM.copy(); X[sty[:, None] == sty[None, :]] = -9; np.fill_diagonal(X, -9); nb = np.argsort(-X, 1)[:, :20]; thr = np.take_along_axis(X, nb[:, 19:20], 1)[:, 0]
    la_k = launch[nb].sum(1); la_sim = np.where(launch[None, :] & (X >= thr[:, None]), np.clip(X, 0, None), 0).sum(1)
    sl_new = pd.Series(launch, index=sty).groupby(level=0).sum(); la_sty = np.array([sl_new.get(s, 0) for s in sty]) > 0
    rows.append(pd.DataFrame({"origin": o, "style_color": items, "la_k": la_k, "la_sim": la_sim, "la_sty": la_sty.astype(int), "n_launch": int(launch.sum())}))
L = pd.concat(rows); T = T.merge(L, on=["origin", "style_color"], how="left")
REC_ = [o for o in ORIGINS if o >= "2025-06"]; d = T[T.origin.isin(REC_) & (T.y >= 20)]
print("window launches with photos per origin:", L.groupby("origin").n_launch.first().to_dict())
for c in ["la_k", "la_sim", "la_sty"]:
    z = d.la_sty.values.astype(float); rz = lambda v: v - np.polyval(np.polyfit(z, v, 1), z)
    print(f"  {c:7s} Spearman with e {spearmanr(d[c], d.e)[0]:+.3f} | partial on la_sty {spearmanr(rz(d[c].values.astype(float)), rz(d.e.values))[0]:+.3f}")
print("\nmean residual (log, actual / p2) by number of look launches among the 20 content neighbours (recent):")
print(d.groupby(pd.cut(d.la_k, [-1, 0, 1, 3, 20])).agg(items=("e", "size"), mean_e=("e", "mean"), units=("y", "sum")).round(3).to_string())
T["p2_la"] = T.p2.copy(); oo = pd.PeriodIndex(T.origin, freq="M").astype(int)
for o in sorted(T.origin.unique()):
    tr = (oo + 4 <= pd.Period(o).ordinal) & (T.y >= 1)
    if T.origin[tr].nunique() < 3: continue
    A = np.column_stack([np.ones(tr.sum()), T.la_k[tr], T.la_sty[tr]]); b = np.linalg.lstsq(A, T.e[tr], rcond=None)[0]
    m = (T.origin == o).values; T.loc[m, "p2_la"] = T.p2[m] * np.exp(b[0] + b[1] * T.la_k[m] + b[2] * T.la_sty[m]); print("  origin", o, "b_look", round(b[1], 3), "b_sibling", round(b[2], 3))
for nm, m in [("nine recent origins", T.origin.isin(REC_)), ("all 13 origins", np.ones(len(T), bool))]:
    print(f"\n######## {nm}"); print(sl.table(T[m], ["p2", "p2_la"], D).round(4).to_string())
print("bootstrap p2_la vs p2, recent:", np.round(sl.boot(T[T.origin.isin(REC_)], "p2_la", "p2"), 4))
