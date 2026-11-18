"""D255: monthly split of the 4-month forecast (D244: September +28% over, November -19% under, January -14%, June / July -12%).
Pure shape error: every item's 4-month forecast is scaled to its ACTUAL 4-month total, so only the split across the 4 months is scored:
shape WAPE = sum |s_hat x Y - y_m| / sum y_m. Recent origins, items of data/dec_base.parquet with y >= 20. Shapes compared:
  lgb      the LightGBM monthly profile used for the 4-month members (data/best_backtest_preds.parquet)
  type_ly  the item's product type's like-for-like monthly units a year earlier (items live then), as shares
  own_ly   the item's own units a year earlier in the same 4 months (items with >= 40 units then), else type_ly
  site_ly  the whole site's units a year earlier
  blends   geometric means of lgb with type_ly / own_ly (weights 0.5, and 0.3 on last year), renormalised
Also the share bias per target calendar month for each shape. Writes eda/shape_output.txt (stdout) and data/shape_alt.parquet
(origin, style_color, h, share for each shape)."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 200)
B = pd.read_parquet("data/dec_base.parquet"); B = B[B.origin >= "2025-06"]
BT = pd.read_parquet("data/best_backtest_preds.parquet")[["origin", "style_color", "h", "pred"]]
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
it = pd.read_parquet("data/items.parquet").set_index("style_color"); pt = it.product_type.reindex(M.index).fillna("na")
first = M.gt(0).idxmax(axis=1)
rows = []
for o, g in B.groupby("origin"):
    P = pd.Period(o, "M"); tm = [str(P + h) for h in range(1, 5)]; ly = [str(P + h - 12) for h in range(1, 5)]; base_ly = str(P - 14)
    lb = BT[BT.origin == o].pivot(index="style_color", columns="h", values="pred").reindex(g.style_color)
    S = {"lgb": lb.values / lb.values.sum(1, keepdims=True)}
    coh = first < base_ly                                                    # like-for-like: live 2 months before last year's window
    MT = M[coh].groupby(pt[coh]).sum()[ly]; ts = (MT + 1).div((MT + 1).sum(1), axis=0)
    S["type_ly"] = ts.reindex(pt.reindex(g.style_color).values).fillna(0.25).values
    own = M.reindex(g.style_color)[ly].fillna(0).values; ok = own.sum(1) >= 40
    S["own_ly"] = np.where(ok[:, None], (own + 1) / (own + 1).sum(1, keepdims=True), S["type_ly"])
    st = M[ly].sum(); S["site_ly"] = np.repeat((st / st.sum()).values[None, :], len(g), 0)
    gm = lambda a, b, w: (lambda x: x / x.sum(1, keepdims=True))(np.exp((1 - w) * np.log(np.clip(a, 1e-6, None)) + w * np.log(np.clip(b, 1e-6, None))))
    S["lgb+type_ly"] = gm(S["lgb"], S["type_ly"], 0.5); S["lgb+own_ly"] = gm(S["lgb"], S["own_ly"], 0.5); S["lgb+own_ly.3"] = gm(S["lgb"], S["own_ly"], 0.3)
    Y = M.reindex(g.style_color)[tm].fillna(0).values
    for k, s in S.items():
        for h in range(4): rows.append(pd.DataFrame({"origin": o, "style_color": g.style_color.values, "h": h + 1, "cal": tm[h][-2:], "shape": k, "s": s[:, h], "ym": Y[:, h], "Y": Y.sum(1), "y4": g.y.values}))
R = pd.concat(rows); R = R[R.y4 >= 20]; R["fm"] = R.s * R.Y
print("== shape WAPE (forecast scaled to the actual 4-month total; recent origins, items >= 20 units)")
print(R.groupby("shape").apply(lambda d: (d.fm - d.ym).abs().sum() / d.ym.sum()).round(4).sort_values().to_string())
print("\n== share bias by target calendar month (sum of shape-split units / actual - 1)")
print(R.groupby(["cal", "shape"]).apply(lambda d: d.fm.sum() / d.ym.sum() - 1).unstack().round(3).to_string())
print("\n== shape WAPE by origin")
print(R.groupby(["origin", "shape"]).apply(lambda d: (d.fm - d.ym).abs().sum() / d.ym.sum()).unstack().round(3).to_string())
R[["origin", "style_color", "h", "shape", "s"]].to_parquet("data/shape_alt.parquet", index=False)
