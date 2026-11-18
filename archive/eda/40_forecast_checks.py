"""Forecast checks for the last open questions, all at the June 2025 origin (the analogue of the real task).
a July: the production method (quantile P50 total x monthly profile of the monthly model) - predicted July and July/June vs actual, by group.
b mature items: the production procedure (50/50 blend of model and own trend x 0.843) - bias and hit rates at June and July 2025, and the
  implied forecast / last-year ratio vs the actual ratio the same items achieved.
c floors: floor level = q-th percentile of the 2024 item-level Jul-Oct/June distribution (q = 25..75), applied to young Homecoming/Halloween items in 2025."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 230)
M = pd.read_parquet("data/monthly_item.parquet"); cols = list(M.columns)
it = pd.read_parquet("data/items.parquet").set_index("style_color")
start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
occ = it.occasion.fillna(""); mo = lambda y, ms: [f"{y}-{m:02d}" for m in ms]
def met(f, a):
    f, a = np.asarray(f, float), np.asarray(a, float); ape = np.abs(f - a) / np.clip(a, 1, None)
    return dict(n=len(a), bias=round(f.sum() / a.sum() - 1, 3), wape=round(np.abs(f - a).sum() / a.sum(), 3), w20=round((ape <= .2).mean(), 3), w50=round((ape <= .5).mean(), 3))
def group(ids, y):
    age = (pd.Timestamp(f"{y}-06-30") - start.reindex(ids)).dt.days.values / 30.44
    g = np.where(age < 3, "launch <3m", np.where(age < 12, "3-12 months", "12m+"))
    g = np.where(it.product_type.reindex(ids).isin(["Swimwear Top", "Swimwear Bottom"]).values, "swimwear", g)
    return pd.Series(g, index=ids)

print("== a  July at the June 2025 origin, production method")
Q = pd.read_parquet("data/q_backtest.parquet"); q = Q[Q.origin == "2025-06"].set_index("style_color")
B = pd.read_parquet("data/best_backtest_preds.parquet"); B = B[B.origin == "2025-06"].pivot(index="style_color", columns="h", values="pred")
ids = q.index.intersection(B.index); prof = B.loc[ids].div(B.loc[ids].sum(axis=1), axis=0)
jul_pred = q.loc[ids, "q50"] * prof[1]; jun = M.loc[ids, "2025-06"]; jul_act = M.loc[ids, "2025-07"]
G = pd.DataFrame({"grp": group(ids, 2025).values, "jun": jun.values, "jul_pred": jul_pred.values, "jul_act": jul_act.values, "tot_pred": q.loc[ids, "q50"].values, "tot_act": q.loc[ids, "ytot"].values,
                  "share_pred": prof[1].values}, index=ids)
G["share_act"] = G.jul_act / G.tot_act.clip(lower=1)
rows = []
for k, d in list(G.groupby("grp")) + [("ALL", G)]:
    rows.append(dict(group=k, n=len(d), julJun_pred=round(d.jul_pred.sum() / d.jun.sum(), 3), julJun_act=round(d.jul_act.sum() / d.jun.sum(), 3), july_bias=round(d.jul_pred.sum() / d.jul_act.sum() - 1, 3),
                     total_bias=round(d.tot_pred.sum() / d.tot_act.sum() - 1, 3), jul_share_pred=round(d.jul_pred.sum() / d.tot_pred.sum(), 3), jul_share_act=round(d.jul_act.sum() / d.tot_act.sum(), 3)))
A = pd.DataFrame(rows).set_index("group"); print(A.to_string())

print("\n== b  mature items: the production blend at the summer origins")
src = open("eda/31_yoy_test.py", encoding="utf-8").read(); src = src[:src.index("def met(")]; ns = {}; exec(src, ns); D = ns["D"]
for o in ["2025-06", "2025-07"]:
    d = D[D.origin == o]; b = np.exp(.5 * np.log(d.p50 + 1) + .5 * np.log(d.yn * 0.843 + 1)) - 1
    print(o, "model alone", met(d.p50, d.a), "| trend x 0.843", met(d.yn * 0.843, d.a), "| blend (used)", met(b, d.a))
d = D[D.origin == "2025-06"]; b = np.exp(.5 * np.log(d.p50 + 1) + .5 * np.log(d.yn * 0.843 + 1)) - 1
print(f"June 2025 origin, same items: blend forecast / last Jul-Oct = {b.sum() / d.ly.sum():.3f} (item median {(b / d.ly).median():.3f}); actual / last Jul-Oct = {d.a.sum() / d.ly.sum():.3f} (item median {(d.a / d.ly).median():.3f})")
O = pd.read_csv("data/final_forecast_q3.csv", index_col=0); m = O[O.mature].index; ly26 = M.loc[m, mo(2025, [7, 8, 9, 10])].sum(axis=1)
print(f"2026: blend forecast / last Jul-Oct = {O.loc[m, 'after_yoy_blend'].sum() / ly26.sum():.3f} (item median {(O.loc[m, 'after_yoy_blend'] / ly26).median():.3f}); final incl. campaign = {O.loc[m, 'final_total'].sum() / ly26.sum():.3f}")

print("\n== c  floor level sensitivity (2024 distribution -> 2025 young items)")
src = open("eda/29_segment_rule_test.py", encoding="utf-8").read(); src = src[:src.index("def met(")]; ns = {}; exec(src, ns); g29, seg = ns["g"], ns["seg"]
age = (pd.Timestamp("2025-06-30") - start.reindex(g29.style_color)).dt.days.values / 30.44
for s in ["Homecoming", "Halloween"]:
    live = start[start < pd.Timestamp("2024-05-15")].index.intersection(M.index); live = live[M.loc[live, "2024-06"] >= 50]
    r24 = (M.loc[live, mo(2024, [7, 8, 9, 10])].sum(axis=1) / M.loc[live, "2024-06"])[seg(live).values == s]
    d = g29[(age < 15) & (g29.seg == s).values]
    out = [dict(level="model only", mult=np.nan, **met(d.p50, d.ytot))]
    for qq in [0.25, 0.4, 0.5, 0.6, 0.75]:
        mult = r24.quantile(qq); f = np.maximum(d.p50, d.jun * mult); out.append(dict(level=f"floor at 2024 p{int(qq*100)}", mult=round(mult, 2), **met(f, d.ytot)))
    print(f"{s}: 2024 n={len(r24)}, young 2025 items n={len(d)}"); print(pd.DataFrame(out).set_index("level").to_string())
