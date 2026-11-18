"""D205b (fork): a second fashion block, the trend's position in its cycle and its spread. At every origin o, from data up to o only:
  fl_peak_<g>    the value's share of units in the last 3 months over its highest 3-month share in the last 12 months (log; 0 = at
                 its peak, negative = past it) for type, colour, silhouette, print
  fl_trough_type the type's share over its lowest 3-month share in the last 12 months (log; large = a rising trend)
  fl_combo_mom / fl_combo_yoy   share momentum / year-on-year of the item's type x colour and type x silhouette combinations (mean)
  fl_reg_type    regional diffusion: for each state, the 3-month share momentum of the item's type within the state; weighted by the
                 item's own state mix (last 12 months), minus the national momentum: is its type growing where it sells?
  fl_reg_all     the same for all units (is the item's region growing faster than the country)
  fl_new_type    share of the type's last-3-month units that come from items launched in the last 3 months (how fast the type turns over)
Writes data/fk_fash2_feats.parquet and adds "FL" to data/fk_fash_meta.json."""
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
it = pd.read_parquet("data/items.parquet").set_index("style_color"); items = np.sort(it.index.values); it = it.reindex(items); n = len(items)
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; M = M.reindex(items).fillna(0); COLS = list(M.columns)
Tx = pd.read_parquet("data/fk_explain_table.parquet"); A = Tx.groupby("style_color")[["a_silhouette", "a_print_pattern"]].first().reindex(items)
G = {"type": it.product_type.fillna("na").values, "col": it.color.fillna("na").values, "silhouette": A.a_silhouette.astype(str).values, "print": A.a_print_pattern.astype(str).values}
G["type_col"] = pd.Series(G["type"], dtype=object).astype(str).values + "|" + pd.Series(G["col"], dtype=object).astype(str).values; G["type_sil"] = pd.Series(G["type"], dtype=object).astype(str).values + "|" + pd.Series(G["silhouette"], dtype=object).astype(str).values
code = {g: pd.factorize(pd.Series(np.asarray(v, dtype=object)).fillna("na").astype(str))[0] for g, v in G.items()}
S = pd.read_parquet("data/state_item_month.parquet"); states = np.sort(S.state.unique()); si = {s: k for k, s in enumerate(states)}
chk = S.groupby("it").units.sum().reindex(range(n)).fillna(0).values; print("state file vs monthly totals, corr", np.corrcoef(chk, M.sum(1).values)[0, 1].round(4))
SM = {m: np.zeros((n, len(states))) for m in S.m.unique()}
for m, g in S.groupby("m"): SM[m][g.it.values, g.state.map(si).values] = g.units.values
first = pd.to_datetime(it.first_sale).dt.to_period("M").astype(str).values
Pm = lambda o, k: str(pd.Period(o, "M") + k)
win = lambda o, ks: M.reindex(columns=[Pm(o, k) for k in ks]).fillna(0).values.sum(1)
ok = lambda o, k: Pm(o, k) in COLS
C = 20.0
gsum = lambda c, v: np.bincount(c, weights=v, minlength=c.max() + 1)
rows = []
for o in [str(p) for p in pd.period_range("2024-03", "2026-06", freq="M")]:
    f = {"origin": o, "style_color": items}
    l3, p3 = win(o, [-2, -1, 0]), win(o, [-5, -4, -3]); y3 = win(o, [-14, -13, -12]); okp, oky = ok(o, -5), ok(o, -14)
    tot = lambda v: max(v.sum(), 1.0)
    for g in ["type", "col", "silhouette", "print"]:
        c = code[g]; sh_now = (gsum(c, l3) + C) / tot(l3)
        hist = [(gsum(c, win(o, [k - 2, k - 1, k])) + C) / tot(win(o, [k - 2, k - 1, k])) for k in range(-11, 1) if ok(o, k - 2)]
        f[f"fl_peak_{g}"] = np.log(sh_now / np.max(hist, 0))[c] if len(hist) >= 6 else np.nan
        if g == "type": f["fl_trough_type"] = np.log(sh_now / np.min(hist, 0))[c] if len(hist) >= 6 else np.nan
    moms, yoys = [], []
    for g in ["type_col", "type_sil"]:
        c = code[g]; sh = lambda v: (gsum(c, v) + C) / tot(v)
        moms.append(np.log(sh(l3) / sh(p3))[c] if okp else np.full(n, np.nan)); yoys.append(np.log(sh(l3) / sh(y3))[c] if oky else np.full(n, np.nan))
    f["fl_combo_mom"], f["fl_combo_yoy"] = np.mean(moms, 0), np.mean(yoys, 0)
    # regional diffusion
    ms = lambda ks: sum(SM.get(Pm(o, k), 0) for k in ks)
    if okp:
        Sl, Sp = ms([-2, -1, 0]), ms([-5, -4, -3]); mix = ms(range(-11, 1)); mix = mix / np.maximum(mix.sum(1, keepdims=True), 1)
        c = code["type"]; k = c.max() + 1
        tl = np.zeros((k, len(states))); tp = np.zeros((k, len(states))); np.add.at(tl, c, Sl); np.add.at(tp, c, Sp)
        st_mom = np.log(((tl + C / 10) / np.maximum(Sl.sum(0), 1)) / ((tp + C / 10) / np.maximum(Sp.sum(0), 1)))     # type share momentum in each state
        nat = np.log(((gsum(c, l3) + C) / tot(l3)) / ((gsum(c, p3) + C) / tot(p3)))
        f["fl_reg_type"] = (mix * st_mom[c]).sum(1) - nat[c]
        all_mom = np.log((Sl.sum(0) + 50) / (Sp.sum(0) + 50)); f["fl_reg_all"] = mix @ all_mom - np.log((Sl.sum() + 50) / (Sp.sum() + 50))
    else: f["fl_reg_type"] = f["fl_reg_all"] = np.nan
    new = ((first >= Pm(o, -2)) & (first <= o) & (first > "2024-01")).astype(float)
    c = code["type"]; f["fl_new_type"] = (gsum(c, l3 * new) / np.maximum(gsum(c, l3), 1))[c] if o >= "2024-04" else np.nan
    rows.append(pd.DataFrame(f))
X = pd.concat(rows, ignore_index=True); X.to_parquet("data/fk_fash2_feats.parquet", index=False)
FL = [c for c in X.columns if c.startswith("fl_")]
print(X.shape); print(X[X.origin >= "2025-02"][FL].describe().T[["count", "mean", "std", "min", "max"]].round(3).to_string())
meta = json.load(open("data/fk_fash_meta.json")); meta["FL"] = FL; json.dump(meta, open("data/fk_fash_meta.json", "w"), indent=1)
