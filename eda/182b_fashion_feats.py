"""D205 (fork; user request: cluster history, the 5 nearest neighbours in the embedding, and the fashion element in its temporal
context). At every origin o (2024-03 .. 2026-06), from monthly units up to o only. Three blocks, keyed (origin, style_color):

FA  fashion-attribute trends. Groups: product type, colour, occasion, and the tagger's silhouette, length, print, fabric feel,
    weight, sleeve, occasion tier. For each group value, its share of all units:
      mom   log share, last 3 months over the 3 before;   yoy  last 3 months over the same months a year earlier;
      acc1  share in month o over its mean share in o-3 .. o-1 (the newest turn);   sea4  the value's last-year ramp into the
      target months, units(o-11 .. o-8) / (units(o-14 .. o-12) x 4/3)
    Item features: fa_<g>_mom / fa_<g>_yoy for type, colour, silhouette, occasion tier and print; the mean of mom / yoy / acc1 /
    sea4 over all 10 groups; fa_llm_mom / fa_llm_yoy = the item's standardised LLM scores (14 fields) dotted with the shift in the
    units-weighted mean of those scores (where demand is moving in style space); fa_launch_type / fa_launch_col = the buying
    team's bet: the type's (colour's) share of items launched in the last 6 months over its share of units; fa_in_type = the
    item's own share-of-type momentum.
CH  cluster history (D158 multi-view clusters, labels per origin from eda/163b): the cluster's share mom / yoy / acc1; its 6-month
    log-units slope; its realised surprise in the last fully known window, actual units (o-3 .. o) over R3 at o-4 (pooled, and
    the members' median); its share of units from items under 6 months old; the item's momentum relative to the cluster's.
NN  the 5 nearest neighbours in the dynamic co-occurrence embedding (eda/181, passed the D199 cart gate), other styles alive in
    the last 3 months: their mean realised surprise, acc1, 3-month momentum, year-on-year growth, last-year ramp into the target
    months, the item's log level minus theirs, their mean cosine similarity and age.
"Realised surprise" uses only outcomes known at o: R3 at o-4 = 4/3 x units(o-6 .. o-4), actual = units(o-3 .. o).
Writes data/fk_fash_feats.parquet."""
import sys, glob, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
it = pd.read_parquet("data/items.parquet").set_index("style_color"); items = np.sort(it.index.values); it = it.reindex(items); n = len(items)
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; M = M.reindex(items).fillna(0)
Tx = pd.read_parquet("data/fk_explain_table.parquet"); META = json.load(open("data/fk_explain_meta.json"))
A = Tx.groupby("style_color")[[c for c in META["CAT"] if c.startswith("a_")]].first().reindex(items)
GROUPS = {"type": it.product_type.fillna("na").values, "col": it.color.fillna("na").values, "occ": it.occasion.fillna("na").values,
          **{g[2:]: A[g].astype(str).fillna("na").values for g in A.columns}}
SHOW = ["type", "col", "silhouette", "occasion_tier", "print_pattern"]
L = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("data/llm/scores_part*.csv"))]).drop_duplicates("style_color").set_index("style_color").reindex(items)
Lz = ((L - L.mean()) / L.std()).fillna(0).values
first = pd.to_datetime(it.first_sale).dt.to_period("M").astype(str).values
LAB = pd.read_parquet("data/fk_cluster_labels.parquet"); DYN = np.load("data/fk_cooc_dyn.npz", allow_pickle=True)
assert (DYN["items"].astype(str) == items).all()
sty = pd.Series(items).str.split("_").str[0].values; COLS = list(M.columns)
Pm = lambda o, k: str(pd.Period(o, "M") + k)
def U(o, ks):
    cs = [Pm(o, k) for k in ks]; ok = [c in COLS for c in cs]
    return M.reindex(columns=cs).fillna(0).values, all(ok)
C = 20.0   # pseudo-units in every share


def grp_feats(codes, x):
    """per-item value of x (dict of window -> item units) aggregated to the item's group value."""
    g = pd.Series(codes).astype("category").cat.codes.values; k = g.max() + 1
    return {w: np.bincount(g, weights=v, minlength=k)[g] for w, v in x.items()}


rows = []
for o in [str(p) for p in pd.period_range("2024-03", "2026-06", freq="M")]:
    (l3, _), (p3, okp), (y3, oky), (yt, okt), (m0, _), (m3, _) = U(o, [-2, -1, 0]), U(o, [-5, -4, -3]), U(o, [-14, -13, -12]), U(o, [-11, -10, -9, -8]), U(o, [0]), U(o, [-3, -2, -1])
    W = {"l3": l3.sum(1), "p3": p3.sum(1), "y3": y3.sum(1), "yt": yt.sum(1), "m0": m0.sum(1), "m3": m3.sum(1) / 3}
    TOT = {w: v.sum() for w, v in W.items()}
    share = lambda G, w: (G[w] + C) / max(TOT[w], 1.0)
    f = {"origin": o, "style_color": items}
    mom_all, yoy_all, acc_all, sea_all = [], [], [], []
    for gname, codes in GROUPS.items():
        G = grp_feats(codes, W)
        mom = np.log(share(G, "l3") / share(G, "p3")) if okp else np.full(n, np.nan)
        yoy = np.log(share(G, "l3") / share(G, "y3")) if oky else np.full(n, np.nan)
        acc = np.log(share(G, "m0") / share(G, "m3"))
        sea = np.log((G["yt"] + C) / (G["y3"] * 4 / 3 + C)) if (oky and okt) else np.full(n, np.nan)
        mom_all.append(mom); yoy_all.append(yoy); acc_all.append(acc); sea_all.append(sea)
        if gname in SHOW: f[f"fa_{gname}_mom"] = mom; f[f"fa_{gname}_yoy"] = yoy
    f["fa_mean_mom"], f["fa_mean_yoy"], f["fa_mean_acc1"], f["fa_mean_sea4"] = [np.mean(x, 0) for x in (mom_all, yoy_all, acc_all, sea_all)]
    tilt = lambda w: (W[w][:, None] * Lz).sum(0) / max(W[w].sum(), 1.0)
    f["fa_llm_mom"] = Lz @ (tilt("l3") - tilt("p3")) if okp else np.nan
    f["fa_llm_yoy"] = Lz @ (tilt("l3") - tilt("y3")) if oky else np.nan
    launched = ((first >= Pm(o, -5)) & (first <= o) & (first > "2024-01")).astype(float)
    for gname, key in [("type", "fa_launch_type"), ("col", "fa_launch_col")]:
        G = grp_feats(GROUPS[gname], {"la": launched, "l3": W["l3"]})
        f[key] = np.log(((G["la"] + 1) / max(launched.sum() + 1, 1)) / ((G["l3"] + C) / max(TOT["l3"], 1))) if o >= "2024-07" else np.nan
    G = grp_feats(GROUPS["type"], W); f["fa_in_type"] = np.log(((W["l3"] + 1) / (G["l3"] + C)) / ((W["p3"] + 1) / (G["p3"] + C))) if okp else np.nan
    # realised surprise of the last fully known window (R3 at o-4 vs actual o-3 .. o)
    (r_hist, okr) = U(o, [-6, -5, -4]); r3_prev = r_hist.sum(1) * 4 / 3; act = U(o, [-3, -2, -1, 0])[0].sum(1)
    aliv4 = (r3_prev > 0) & okr; surpr = np.where(aliv4, np.log((act + 1) / (r3_prev + 1)), np.nan)
    item_mom = np.log((W["l3"] + 1) / (W["p3"] + 1)) if okp else np.full(n, np.nan)
    item_yoy = np.where((W["y3"] > 0) & oky, np.log((W["l3"] + 1) / (W["y3"] + 1)), np.nan)
    item_acc = np.log((W["m0"] + 1) / (W["m3"] + 1))
    age = np.array([(pd.Period(o, "M") - pd.Period(x, "M")).n if x <= o else -1 for x in first], float); age[age < 0] = np.nan
    # cluster history
    lab = LAB[LAB.origin == o].set_index("style_color").cluster.reindex(items).fillna(-1).astype(int).values; has = lab >= 0
    if has.any():
        cl = np.where(has, lab, lab.max() + 1); k = cl.max() + 1; bc = lambda v: np.bincount(cl, weights=np.nan_to_num(v), minlength=k)[cl]
        Gc = {w: bc(v) for w, v in W.items()}
        f["ch_mom"] = np.log(share(Gc, "l3") / share(Gc, "p3")) if okp else np.nan
        f["ch_yoy"] = np.log(share(Gc, "l3") / share(Gc, "y3")) if oky else np.nan
        f["ch_acc1"] = np.log(share(Gc, "m0") / share(Gc, "m3"))
        H, _ = U(o, range(-5, 1)); Hc = np.stack([bc(H[:, j]) for j in range(6)], 1); tt = np.arange(6) - 2.5
        f["ch_traj6"] = (np.log1p(Hc) * tt).sum(1) / (tt ** 2).sum()
        f["ch_surprise"] = np.log((bc(np.where(aliv4, act, 0)) + 1) / (bc(np.where(aliv4, r3_prev, 0)) + 1)) if okr else np.nan
        med = pd.Series(surpr).groupby(cl).median(); f["ch_surprise_med"] = med.reindex(cl).values
        f["ch_young"] = bc(np.where(age < 6, W["l3"], 0)) / np.maximum(Gc["l3"], 1)
        f["ch_rel_mom"] = item_mom - np.log((Gc["l3"] + 1) / (Gc["p3"] + 1)) if okp else np.nan
        for c in [c for c in f if c.startswith("ch_")]:
            if np.ndim(f[c]): f[c] = np.where(has, f[c], np.nan)
    # 5 nearest neighbours in the dynamic co-occurrence embedding
    if o in DYN.files:
        V = DYN[o].astype(np.float32); V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9); live = W["l3"] > 0
        S = V @ V.T; S[:, ~live] = -9; S[sty[:, None] == sty[None, :]] = -9
        nb = np.argsort(-S, 1)[:, :5]; sim = np.take_along_axis(S, nb, 1); ok = sim > -1
        nm = lambda x: np.where(ok.any(1), np.nanmean(np.where(ok, x[nb], np.nan), 1), np.nan)
        lvl = np.log1p(W["l3"] / 3); sea_i = np.where((W["y3"] > 0) & oky & okt, np.log((W["yt"] + 1) / (W["y3"] * 4 / 3 + 1)), np.nan)
        with np.errstate(all="ignore"):
            f.update({"nn_surprise": nm(surpr), "nn_acc1": nm(item_acc), "nn_mom3": nm(item_mom), "nn_yoy": nm(item_yoy), "nn_sea4": nm(sea_i),
                      "nn_lvl_rel": lvl - nm(lvl), "nn_sim": np.where(ok, sim, np.nan).mean(1), "nn_age": nm(age)})
    rows.append(pd.DataFrame(f))
    print(o, "clusters", int(has.sum()), "| dyn", o in DYN.files, flush=True)
X = pd.concat(rows, ignore_index=True); X.to_parquet("data/fk_fash_feats.parquet", index=False)
FEATS = [c for c in X.columns if c[:3] in ("fa_", "ch_", "nn_")]
print(X.shape); print(X[X.origin >= "2025-02"][FEATS].describe().T[["count", "mean", "std", "min", "max"]].round(3).to_string())
json.dump({"FA": [c for c in FEATS if c.startswith("fa_")], "CH": [c for c in FEATS if c.startswith("ch_")], "NN": [c for c in FEATS if c.startswith("nn_")]}, open("data/fk_fash_meta.json", "w"), indent=1)
