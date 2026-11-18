"""Fork 2 (goal, 18 Nov: "optimize any other bits on the meaningful months with the new metrics"): the monthly split of the 4-month total.
The workbook splits each item's total by the geometric mean of the rule agents' and the NN's monthly profiles (archive/eda/181k, prof). Never scored.
Backtest: the new nx5's corrected total (data/geo_nx5r_x_preds.parquet, time-honest) split by
  cur      geometric mean of the rules' (rule_agents/<origin>/predictions_verified.csv) and the NN's (S13ecx6, h = 1..4) profiles   [as submitted]
  rules / nn   one of them
  type_ly  the product type's own monthly shares in the same four months a year earlier (like-for-like items live then)
  item_ly  the item's own shares a year earlier when it sold then (>= 20 units), else cur
  blends   cur shrunk toward flat (0.25 each month) by s; cur x type_ly geometric
Metrics: monthly WAPE (sum |f_m - y_m| over items and months / sum y), its ratio to the flat split, and the new per-item metric (median over
items of the correlation between forecast and actual monthly shares). Origins: recent 9, focus windows 2025-06 (Jul-Oct 25) and 2026-02.
stdout -> pipeline/month_split_output.txt"""
import sys, os, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
G = pd.read_parquet("data/geo_nx5r_x_preds.parquet")[["origin", "style_color", "geo", "y", "age_m"]]
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
it = pd.read_parquet("data/items.parquet").set_index("style_color"); PT = it.product_type.reindex(M.index).fillna("na")
NN = pd.read_parquet("data/nn_backtest_preds_S13ecx6.parquet").pivot_table(index=["origin", "style_color"], columns="h", values="q50")
mon = lambda o, h: str(pd.Period(o, "M") + h)
rows = []
for o, g in G.groupby("origin"):
    ms = [mon(o, h) for h in (1, 2, 3, 4)]
    if not all(m in M.columns for m in ms): continue
    f = f"rule_agents/{o}/predictions_verified.csv"
    RU = pd.read_csv(f, dtype={"month": str}).pivot(index="style_color", columns="month", values="pred").reindex(columns=ms) if os.path.exists(f) else None
    nn = NN.loc[o] if o in NN.index.get_level_values(0) else None
    ly = [mon(o, h - 12) for h in (1, 2, 3, 4)]; lyok = all(m in M.columns for m in ly + [mon(o, -12 - k) for k in (0, 1, 2)])
    TY = None
    if lyok:
        live = M[[mon(o, -12 - k) for k in (0, 1, 2)]].min(1) > 0   # like-for-like: items selling in the 3 months before last year's window
        TY = M[live][ly].groupby(PT[live]).sum(); TY = TY.div(TY.sum(1), axis=0)
    for r in g.itertuples():
        y = M.loc[r.style_color, ms].values.astype(float) if r.style_color in M.index else None
        if y is None or y.sum() <= 0: continue
        pr = lambda v: (v / v.sum()) if v is not None and np.all(np.isfinite(v)) and v.sum() > 0 else None
        p_ru = pr(RU.loc[r.style_color].values.astype(float)) if RU is not None and r.style_color in RU.index else None
        p_nn = pr(nn.loc[r.style_color].values.astype(float)) if nn is not None and r.style_color in nn.index else None
        p_cur = pr(np.exp((np.log(np.clip(p_ru, 1e-4, None)) + np.log(np.clip(p_nn, 1e-4, None))) / 2)) if p_ru is not None and p_nn is not None else (p_ru if p_ru is not None else p_nn)
        if p_cur is None: p_cur = np.full(4, 0.25)
        p_ty = TY.loc[PT[r.style_color]].values if TY is not None and PT[r.style_color] in TY.index else None
        yl = M.loc[r.style_color, ly].values.astype(float) if lyok else None
        p_it = pr(yl) if yl is not None and yl.sum() >= 20 else None
        rows.append(dict(origin=o, style_color=r.style_color, tot=r.geo, y=y, cur=p_cur, rules=p_ru if p_ru is not None else p_cur, nn=p_nn if p_nn is not None else p_cur,
                         type_ly=p_ty if p_ty is not None else p_cur, item_ly=p_it if p_it is not None else p_cur, new=(r.age_m if pd.notna(r.age_m) else 99) < 6))
R = pd.DataFrame(rows); print("item-origins", len(R))
def variants(r):
    c = r.cur; out = {"flat": np.full(4, .25), "cur": c, "rules": r.rules, "nn": r.nn, "type_ly": r.type_ly, "item_ly": r.item_ly}
    for s in (0.25, 0.5): out[f"cur shrunk {s}"] = (1 - s) * c + s * 0.25
    gm = np.exp((np.log(np.clip(c, 1e-4, None)) + np.log(np.clip(r.type_ly, 1e-4, None))) / 2); out["cur x type_ly"] = gm / gm.sum()
    gm3 = np.exp((np.log(np.clip(r.rules, 1e-4, None)) + np.log(np.clip(r.nn, 1e-4, None)) + np.log(np.clip(r.type_ly, 1e-4, None))) / 3); out["rules x nn x type_ly"] = gm3 / gm3.sum()
    return out
V = [variants(r) for r in R.itertuples()]; names = list(V[0])
def report(mask, lab):
    m = np.where(mask)[0]; Y = np.stack(R.y.values[m]); tot = R.tot.values[m]; res = {}
    for nm in names:
        Pm = np.stack([V[i][nm] for i in m]) * tot[:, None]
        wape = np.abs(Pm - Y).sum() / Y.sum()
        sh = Pm / Pm.sum(1, keepdims=True); ysh = Y / Y.sum(1, keepdims=True)
        cc = [np.corrcoef(a, b)[0, 1] for a, b in zip(sh, ysh) if np.std(a) > 1e-9 and np.std(b) > 1e-9]
        res[nm] = (wape, float(np.median(cc)) if cc else np.nan)
    base = res["flat"][0]
    print(f"\n{lab} (n {len(m)}): monthly WAPE, ratio to flat split, median per-item correlation of monthly shares")
    for nm in names: print(f"   {nm:22s} {res[nm][0]:.4f}  {res[nm][0] / base:.4f}  {res[nm][1]:+.3f}")
O = R.origin.values
for lab, mk in [("recent 9", O >= "2025-06"), ("Jul-Oct 25 (2025-06)", O == "2025-06"), ("Mar-Jun 26 (2026-02)", O == "2026-02"), ("all 13", O > "0"), ("new items, recent", (O >= "2025-06") & R.new.values)]:
    report(mk, lab)
