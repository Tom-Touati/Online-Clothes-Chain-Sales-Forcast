"""D189 (user: "identify items that rise together unexpectedly, with their cart co-occurrence, and use that similarity to cluster / embed;
fill items without a clear place from fashion features"). Tests whether the remaining error is group-level and predictable.
Monthly surprise s_im = log((u_im + 1) / (E_im + 1)), E_im = the item's mean of the 3 previous months x its product type's seasonal step
(type units in month m-12 over the type's mean of m-15..m-13, else 1): level- and season-free shocks.
At each origin o (data <= o): co-surprise similarity = correlation of s over months o-11..o (items with >= 6 surprises); cart similarity =
cosine of PPMI-SVD vectors (12 months of carts); content similarity = cosine of v6-style inputs (CLIP image + text, D171).
Neighbourhoods (20 nearest, other styles) under:  surprise only | carts only | content only | combined (0.5 surprise + 0.5 carts; items
with < 6 surprises use 0.5 content + 0.5 carts) | random (same size).
A  Is the error group-level? Correlation of p2's 4-month log residual e_i with its neighbourhood's mean residual (same window, not
   knowable at the origin), recent origins; and the share of residual variance explained.
B  Is it predictable at the origin? g_i = neighbourhood mean of the last 1 and 2 months' surprises (known at o). Spearman of g with e
   (raw and partial on the item's own last surprises); and a time-honest correction p2 x exp(b g), b fitted on past origins'
   residuals (o' + 4 <= o), scored with the standard table.
Writes eda/cosurprise_output.txt (stdout), data/cosurprise_feats.parquet."""
import sys, glob, contextlib, io, importlib.util, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.utils.extmath import randomized_svd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); G = lambda *xs: np.exp(np.mean([np.log(np.maximum(x, 0) + 0.5) for x in xs], axis=0)) - 0.5
qv = lambda f: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(key).q50.values
T["p2"] = G(T.rules.values, qv("data/q_backtest_fkdyn_dyn.parquet"), T.nn.values, qv("data/q_backtest_fktab_tabpfn_v2.parquet"))
T["e"] = np.log((T.y + 1) / (T.p2 + 1))
items = np.sort(pd.read_parquet("data/items.parquet").style_color.values); n = len(items); ix = {s: i for i, s in enumerate(items)}
sty = pd.Series(items).str.split("_").str[0].values
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; M = M.reindex(items).fillna(0); months = list(M.columns)
ptype = pd.read_parquet("data/items.parquet").set_index("style_color").product_type.reindex(items).fillna("na").values
MT = M.groupby(ptype).sum(); U = M.values; first = np.argmax(U > 0, 1); first[U.sum(1) == 0] = len(months)
S = np.full(U.shape, np.nan)
for k in range(3, len(months)):
    base = U[:, k - 3:k].mean(1); live = (first <= k - 3) & (base > 0)
    if k >= 15:
        ty = MT.iloc[:, k - 12].values / np.maximum(MT.iloc[:, k - 15:k - 12].mean(1).values, 1e-9); step = pd.Series(ty, index=MT.index).reindex(ptype).fillna(1).values
    else: step = np.ones(n)
    S[:, k] = np.where(live, np.log((U[:, k] + 1) / (base * step + 1)), np.nan)
P = pd.read_parquet("data/basket_pairs_month.parquet"); CM = {}
for m, g in P.groupby("m"):
    c = np.zeros((n, n), np.float32); c[g.it.values, g.it_b.values] = g.n.values; CM[m] = c + c.T
CE = np.load("data/content_emb.npz", allow_pickle=True); cix = pd.Series(range(len(CE["items"])), index=CE["items"].astype(str)).reindex(items).values
CON = np.hstack([CE["img"][cix], CE["txt"][cix]]); CON = CON / np.maximum(np.linalg.norm(CON, axis=1, keepdims=True), 1e-9); SIM_CON = CON @ CON.T
rown = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9); Pm = lambda o, k: str(pd.Period(o, "M") + k)
rng = np.random.default_rng(0); rows = []
for o in sorted(T.origin.unique()):
    ko = months.index(o); W = S[:, max(ko - 11, 0):ko + 1]; ok = np.isfinite(W).sum(1) >= 6
    Z = np.where(np.isfinite(W), W, np.nan); Z = Z - np.nanmean(Z, 1, keepdims=True); Z = np.nan_to_num(Z / np.maximum(np.nanstd(np.where(np.isfinite(W), W, np.nan), 1, keepdims=True), 1e-6))
    SIM_SUR = (Z @ Z.T) / np.maximum((np.isfinite(W).astype(float) @ np.isfinite(W).T.astype(float)), 1)
    C = sum(CM[m] for m in CM if Pm(o, -11) <= m <= o); r = C.sum(1); tot = max(r.sum(), 1); ctx = r ** 0.75; ctx = ctx / max(ctx.sum(), 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"): pmi = np.nan_to_num(np.maximum(np.log((C / tot) / np.outer(r / tot, ctx)), 0), posinf=0, neginf=0)
    u, s_, _ = randomized_svd(rown(pmi), 24, random_state=0); CV = rown(u * np.sqrt(s_)); SIM_CART = CV @ CV.T
    both = ok[:, None] & ok[None, :]
    SIM_COMB = np.where(both, 0.5 * SIM_SUR + 0.5 * SIM_CART, 0.5 * SIM_CON + 0.5 * SIM_CART)
    live = (U[:, max(ko - 2, 0):ko + 1].sum(1) > 0)
    g1 = S[:, ko]; g2 = np.nanmean(S[:, ko - 1:ko + 1], 1)
    for nm, SM in [("surprise", SIM_SUR), ("carts", SIM_CART), ("content", SIM_CON), ("combined", SIM_COMB), ("random", rng.random((n, n)))]:
        X = SM.copy(); X[:, ~live] = -9; X[sty[:, None] == sty[None, :]] = -9; nb = np.argsort(-X, 1)[:, :20]
        rows.append(pd.DataFrame({"origin": o, "style_color": items, "nbset": nm, "nb": list(nb),
                                  "g1": np.nanmean(np.where(np.isfinite(g1[nb]), g1[nb], np.nan), 1), "g2": np.nanmean(np.where(np.isfinite(g2[nb]), g2[nb], np.nan), 1),
                                  "own1": g1, "own2": g2}))
    print(o, "items with >= 6 surprises", int(ok.sum()), flush=True)
NB = pd.concat(rows, ignore_index=True)
REC_ = [o for o in ORIGINS if o >= "2025-06"]
E = T.set_index(["origin", "style_color"]).e
print("\n== A  is the error group-level? corr(e_i, mean e of its 20 neighbours), recent origins, items with y >= 20")
for nm in ["surprise", "carts", "content", "combined", "random"]:
    d = NB[(NB.nbset == nm) & NB.origin.isin(REC_)]; cs, vs = [], []
    for o, g in d.groupby("origin"):
        eo = E.loc[o]; ei = eo.reindex(g.style_color).values; en = np.array([np.nanmean(eo.reindex(items[x]).values) for x in g.nb])
        m = np.isfinite(ei) & np.isfinite(en) & (T[T.origin == o].set_index("style_color").y.reindex(g.style_color).values >= 20)
        cs.append(np.corrcoef(ei[m], en[m])[0, 1])
    print(f"  {nm:9s} mean over origins r = {np.mean(cs):+.3f}  (per origin {np.round(cs, 2)})")
print("\n== B  is it predictable at the origin? Spearman of neighbours' recent surprise with e (raw | partial on own recent surprise)")
for nm in ["surprise", "carts", "content", "combined", "random"]:
    d = NB[(NB.nbset == nm) & NB.origin.isin(REC_)].merge(T[["origin", "style_color", "e", "y"]], on=["origin", "style_color"]); d = d[d.y >= 20]
    out = []
    for gc in ["g1", "g2"]:
        m = np.isfinite(d[gc]) & np.isfinite(d.own2) & np.isfinite(d.e); a, b, z = d[gc][m].values, d.e[m].values, d.own2[m].values
        rz = lambda v: v - np.polyval(np.polyfit(z, v, 1), z); out += [spearmanr(a, b)[0], spearmanr(rz(a), rz(b))[0]]
    print(f"  {nm:9s} g1 {out[0]:+.3f} | {out[1]:+.3f}    g2 {out[2]:+.3f} | {out[3]:+.3f}")
# time-honest correction with the combined neighbourhood's 2-month surprise
d = NB[NB.nbset == "combined"][["origin", "style_color", "g2"]]; T = T.merge(d, on=["origin", "style_color"], how="left"); T["g2"] = T.g2.fillna(0).clip(-2, 2)
T["p2_cs"] = T.p2.copy(); oo = pd.PeriodIndex(T.origin, freq="M").astype(int)
for o in sorted(T.origin.unique()):
    tr = (oo + 4 <= pd.Period(o).ordinal) & (T.y >= 1)
    if T.origin[tr].nunique() < 3: continue
    b = np.polyfit(T.g2[tr], T.e[tr], 1)[0]; m = (T.origin == o).values; T.loc[m, "p2_cs"] = T.p2[m] * np.exp(b * T.g2[m]); print("  origin", o, "fitted b", round(b, 3))
for nm, m in [("nine recent origins", T.origin.isin(REC_)), ("all 13 origins", np.ones(len(T), bool))]:
    print(f"\n######## {nm}"); print(sl.table(T[m], ["p2", "p2_cs"], D).round(4).to_string())
F_ = T[T.origin.isin(REC_)]; print("bootstrap p2_cs vs p2, recent:", np.round(sl.boot(F_, "p2_cs", "p2"), 4))
NB.drop(columns="nb").to_parquet("data/cosurprise_feats.parquet", index=False)
